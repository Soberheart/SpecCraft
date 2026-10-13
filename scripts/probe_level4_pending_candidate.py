"""Collect only unexecuted candidate001; retain failure and exact tool indices.

The old nominal execution is reused as an offline physical diagnostic reference.
No model input is changed and no prediction/online action selection takes place.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
try:
    from prepare_level4_shared_input import assemble_runtime,digest,write
    from resume_level4_shared_input import environment,check_files
except ModuleNotFoundError as exc:
    if exc.name not in ('prepare_level4_shared_input','resume_level4_shared_input'):raise
    from scripts.prepare_level4_shared_input import assemble_runtime,digest,write
    from scripts.resume_level4_shared_input import environment,check_files


def run(request,tools,output):
    if (request.get('kind')!='one_pending_candidate_physical_diagnostic_v1' or request.get('candidate_ids')!=['candidate_001']
            or request.get('new_simulations_max')!=1 or request.get('new_Dyn_rollouts')!=0
            or request.get('candidate_pool_released') is not False):
        raise ValueError('Only the one still-unexecuted candidate interface diagnostic is allowed')
    actual=environment();write(output/'environment.json',actual)
    if actual!=request['expected_compute_environment']:raise ValueError('Reviewed compute environment changed')
    check_files(request['required_files'])
    if shutil.disk_usage(output).free<2*1024**3:raise ValueError('At least2GiB free required')
    for name,h in json.loads((tools/'tool_manifest.json').read_text())['files_sha256'].items():
        if Path(name).name!=name or digest(tools/name)!=h:raise ValueError('Delivered tool source changed')
    runtime=assemble_runtime(request,tools,output.parent)
    from level4.artifacts import runtime_commit,read_json,write_json,utc_now,checked_pool
    from level4.shared_input import exact_numeric_packet
    runtime_commit(runtime)
    pool=Path(request['source_pool']);manifest=checked_pool(pool)
    if digest(pool/'pool.json')!=request['source_pool_sha256']:raise ValueError('Frozen source pool changed')
    nominal=Path(request['nominal_execution']);source_state=pool/'shared_source_state.npz'
    exact_numeric_packet(source_state,nominal/'reset_returned.npz')
    exact_numeric_packet(source_state,nominal/'after_first_action.npz')
    exact_numeric_packet(pool/'static_material_info.npz',nominal/'static_material_info.npz')
    if read_json(request['nominal_exit'])['exit_code']!=0:raise ValueError('Nominal simulation was not completed')
    if read_json(request['old_stage_exit'])['exit_code']!=1:raise ValueError('Historical failed probe state changed')
    provenance=output/'runtime_provenance';provenance.mkdir()
    shutil.copyfile(runtime.parent/'manifest.json',provenance/'sealed_runtime_manifest.json')
    shutil.copyfile(request['runtime_manifest'],provenance/'parent_runtime_manifest.json')
    sealed=read_json(runtime.parent/'manifest.json')
    shutil.copyfile(sealed['execution_environment_evidence']['path'],provenance/'execution_environment_probe.json')
    shutil.copyfile(pool/'pool.json',output/'source_pool_manifest.json')
    # Retain the nominal reference actual bodies, not just an NSCC path.
    reference=output/'nominal_reference';reference.mkdir();traces=reference/'actual_tools';traces.mkdir()
    for name in ('reset_returned.npz','after_first_action.npz','static_material_info.npz','capture_report.json'):
        shutil.copyfile(nominal/name,reference/name)
    for index in range(63):shutil.copyfile(nominal/'actual_tools'/('frame_%03d.npz'%index),traces/('frame_%03d.npz'%index))
    shutil.copyfile(request['nominal_exit'],reference/'original_nominal_exit.json')
    shutil.copyfile(request['old_stage_exit'],reference/'original_failed_stage_exit.json')
    shutil.copyfile(request['legacy_source_log'],reference/'legacy_log.pkl')
    shutil.copyfile(pool/'planner_input.npz',reference/'original_planner_input.npz')
    record=next(r for r in manifest['candidates'] if r['candidate_id']=='candidate_001')
    ledger={r['candidate_id']:{'prediction':'not_run','this_job_execution':'not_executed'} for r in manifest['candidates']}
    ledger['candidate_000']['prior_execution']='completed_prior_failed_probe';ledger['candidate_000']['this_job_execution']='reused_not_rerun'
    write_json(output/'candidate_status_before.json',ledger)
    target=output/'execution/candidate_001'
    command=[sys.executable,'-B','-u',str(tools/'probe_level4_candidate_interface.py'),'capture',
        '--runtime',str(runtime),'--config',str(pool/record['scene']),'--output',str(target),'--shared-state',str(source_state)]
    started=time.monotonic()
    with (output/'candidate_001.log').open('xb') as log:result=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
    write_json(output/'candidate_001_exit.json',{'command':command,'exit_code':result.returncode,'elapsed_s':time.monotonic()-started,
        'completed_utc':utc_now(),'candidate_id':'candidate_001','source_pool_sha256':request['source_pool_sha256']})
    ledger['candidate_001']['this_job_execution']='completed' if not result.returncode else 'failed'
    write_json(output/'candidate_status_after_execution.json',ledger)
    if result.returncode:raise RuntimeError('Candidate001 independent capture failed; raw partial data retained')
    from omegaconf import OmegaConf
    center=np.asarray(OmegaConf.load(pool/'source_scene.yaml').actions.grasp.to_pos,dtype=float)
    from level4.candidate_interface import audit_candidate
    try:
        report=audit_candidate(target,request['legacy_source_log'],source_state,pool/'static_material_info.npz',pool/'planner_input.npz',
            center,record['parameters_mm_mm_deg'],physical_reference=traces)
    except Exception as exc:
        write_json(target/'physical_interface_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'diagnostic_position_limit_mm_unchanged':.001,'legacy_model_input_alignment':'not_certified','candidate_pool_released':False})
        raise
    report.update(created_utc=utc_now(),candidate_id='candidate_001',
        status='physical_candidate001_diagnostic_passed_legacy_model_input_not_certified',
        nominal_reference_prior_job='25787714.pbs101',nominal_reference_prior_job_status='failed',
        source_pool_sha256=request['source_pool_sha256'],new_simulations=1,new_Dyn_rollouts=0,
        scientific_scope='offline_interface_diagnostic_no_task_evaluation_no_online_truth_selection',
        main_frame=60,main_s=1.920,task_satisfaction='not_evaluated',model_input_changed=False,candidate_pool_released=False)
    write_json(output/'physical_interface_report.json',report)
    check_files(request['required_files'])
    return report


def collect(request_path,output):
    output.mkdir(exist_ok=False);request=json.loads(request_path.read_text(encoding='utf-8'));started=time.monotonic();status=0
    try:report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1;report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False}
        write(output/'probe_failure.json',report)
    write(output/'request.json',request)
    toolkit=output.parent/'toolkit.zip'
    if toolkit.is_file():shutil.copyfile(toolkit,output/'submitted_toolkit.zip')
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-started,'status':report['status'],
        'execution_subprocess_exits_recorded':len(list(output.glob('candidate_*_exit.json'))),'new_simulations_max':1,
        'model_prediction_run':False,'legacy_model_input_alignment':'not_certified','candidate_pool_released':False})
    write(output/'evidence_manifest.json',{'files_sha256':{p.relative_to(output).as_posix():digest(p)
        for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        'scope':'true_pending_candidate_physical_interface_and_exact_tool_index_diagnostic_not_task_result'})
    archive=output.parent/'diagnostic_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(output).as_posix())
    print(json.dumps({'exit_code':status,'status':report['status'],'archive':str(archive),'archive_sha256':digest(archive)}),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();sys.exit(collect(args.request.resolve(),args.output.resolve()))
