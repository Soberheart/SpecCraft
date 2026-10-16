"""Two fixed prospective adapter checks; never release model candidate pool."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
try:
    from prepare_level4_shared_input import assemble_runtime,digest,write
    from resume_level4_shared_input import environment,check_files
except ModuleNotFoundError:
    from scripts.prepare_level4_shared_input import assemble_runtime,digest,write
    from scripts.resume_level4_shared_input import environment,check_files


def run(request,tools,output):
    if (request['kind']!='two_fixed_transaction_v3' or request['candidate_ids']!=['candidate_000','candidate_001']
            or request['new_simulations_max']!=2 or request['new_Dyn_rollouts']!=0
            or request['candidate_pool_released'] is not False):
        raise ValueError('Scope differs from two fixed interface checks')
    write(output/'environment.json',environment())
    if environment()!=request['expected_compute_environment']:raise ValueError('Compute environment changed')
    check_files(request['required_files'])
    if shutil.disk_usage(output).free<2*1024**3:raise ValueError('At least2GiB free required')
    runtime=assemble_runtime(request,tools,output.parent)
    from level4.artifacts import runtime_commit,checked_pool,read_json,write_json,utc_now
    from level4.shared_input import exact_numeric_packet
    from level4.clock_review import review_clock_schedule
    from level4.candidate_interface import diagnose_tool_schedule
    runtime_commit(runtime)
    pool=Path(request['source_pool']);manifest=checked_pool(pool)
    if digest(pool/'pool.json')!=request['source_pool_sha256']:raise ValueError('Frozen pool differs')
    provenance=output/'runtime_provenance';provenance.mkdir()
    shutil.copyfile(runtime.parent/'manifest.json',provenance/'sealed_runtime_manifest.json')
    shutil.copyfile(pool/'pool.json',output/'source_pool_manifest.json')
    for name in ('shared_source_state.npz','static_material_info.npz','source_scene.yaml','planner_input.npz'):
        shutil.copyfile(pool/name,output/name)
    ledger={r['candidate_id']:{'prediction':'not_run','execution':'not_executed'} for r in manifest['candidates']}
    write_json(output/'candidate_status_before.json',ledger)
    reports={}
    for candidate in request['candidate_ids']:
        record=next(r for r in manifest['candidates'] if r['candidate_id']==candidate)
        target=output/'execution'/candidate
        shutil.copyfile(pool/record['scene'],output/(candidate+'_scene.yaml'))
        command=[sys.executable,'-B','-u',str(tools/'probe_level4_candidate_interface.py'),'capture',
                 '--runtime',str(runtime),'--config',str(pool/record['scene']),'--output',str(target),
                 '--shared-state',str(pool/'shared_source_state.npz'),
                 '--topology-source-sha256',request['topology_source_sha256']]
        if request.get('gpu_binding_policy') is not None:
            if request['gpu_binding_policy']!='preserve_single_PBS_allocation_v1':
                raise ValueError('Unknown prospective GPU binding policy')
            from level4.artifacts import sha256
            expected=request['gpu_builder_source_sha256']
            if sha256(runtime/'sim/generate/builder.py')!=expected:
                raise ValueError('Reviewed GPU builder changed')
            command.extend(['--gpu-builder-source-sha256',expected])
        ledger[candidate]['execution']='running'
        write_json(output/(candidate+'_status_before_execution.json'),ledger)
        started=time.monotonic()
        with (output/(candidate+'.log')).open('xb') as log:
            result=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
        write_json(output/(candidate+'_exit.json'),{'command':command,'exit_code':result.returncode,
                   'elapsed_s':time.monotonic()-started,'completed_utc':utc_now()})
        ledger[candidate]['execution']='completed' if result.returncode==0 else 'failed'
        write_json(output/(candidate+'_status_after_execution.json'),ledger)
        if result.returncode:raise RuntimeError('Capture failed: '+candidate)
        if request.get('gpu_binding_policy') is not None:
            binding=read_json(target/'gpu_binding_after_init.json')
            if binding['status']!='passed_simulator_initialization_only' or binding['binding_unchanged'] is not True:
                raise RuntimeError('Candidate GPU binding guard did not pass')
        exact_numeric_packet(pool/'shared_source_state.npz',target/'reset_returned.npz')
        exact_numeric_packet(pool/'static_material_info.npz',target/'static_material_info.npz')
        reports[candidate]=review_clock_schedule(target/'clock_events.jsonl',target/'actual_tools')
        write_json(output/(candidate+'_clock.json'),reports[candidate])
    write_json(output/'candidate_status_final.json',ledger)
    from omegaconf import OmegaConf
    import numpy as np
    nominal=output/'execution/candidate_000'
    logs=list((nominal/'scenes').rglob('log.pkl'))
    if len(logs)!=1:raise ValueError('Unique prospective nominal log missing')
    center=np.asarray(OmegaConf.load(pool/'source_scene.yaml').actions.grasp.to_pos,dtype=float)
    candidate=next(r for r in manifest['candidates'] if r['candidate_id']=='candidate_001')
    tools_report=diagnose_tool_schedule(output/'execution/candidate_001',logs[0],
                   nominal/'actual_tools',center,candidate['parameters_mm_mm_deg'])
    write_json(output/'prospective_tool_diagnosis.json',tools_report)
    check_files(request['required_files'])
    return {'status':'prospective_pair_collected_requires_review','clock_labels_match':all(
        row['frozen_replay']['all_elapsed_times_equal_original_labels'] for row in reports.values()),
        'tool_transform_status':tools_report['tool_transform_status'],'candidate_pool_released':False,
        'model_alignment':'unverified','task_success':'not_evaluated','primary_frame':60,'primary_label_s':1.920}


def collect(request_path,output):
    output.mkdir(exist_ok=False);request=json.loads(request_path.read_text());status=0;started=time.monotonic()
    try:report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1;report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False}
        write(output/'probe_failure.json',report)
    write(output/'diagnostic_report.json',report);write(output/'request.json',request)
    shutil.copyfile(output.parent/'toolkit.zip',output/'submitted_toolkit.zip')
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-started,
        'scientific_scope':'prospective_interface_not_task_or_model','candidate_pool_released':False,
        'execution_subprocess_exits_recorded':len(list(output.glob('candidate_*_exit.json'))),
        'candidate_status_snapshots':sorted(p.name for p in output.glob('*status*.json')),
        'failed_or_interrupted_evidence_retained':True})
    write(output/'evidence_manifest.json',{'files_sha256':{p.relative_to(output).as_posix():digest(p)
        for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
    with zipfile.ZipFile(output.parent/'diagnostic_evidence.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:archive.write(p,p.relative_to(output).as_posix())
    print(json.dumps(report));return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.exit(collect(args.request.resolve(),args.output.resolve()))
