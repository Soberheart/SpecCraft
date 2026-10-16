"""One explicitly repeated candidate001 to record previously missing events.

Completion means diagnostic collection, never certification of the old run,
model tools, material tasks, protocol timestamps, or the rest of the pool.
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
    if (request.get('kind')!='one_repeat_candidate001_clock_diagnostic_v1' or request.get('candidate_ids')!=['candidate_001']
            or request.get('new_simulations_max')!=1 or request.get('new_Dyn_rollouts')!=0
            or request.get('candidate_pool_released') is not False or request.get('explicit_repeat_for_missing_events') is not True):
        raise ValueError('Only one explicit candidate001 repeat for missing clock events is permitted')
    actual=environment();write(output/'environment.json',actual)
    if actual!=request['expected_compute_environment']:raise ValueError('Reviewed compute environment changed')
    check_files(request['required_files'])
    if shutil.disk_usage(output).free<2*1024**3:raise ValueError('At least2GiB free required')
    for name,h in json.loads((tools/'tool_manifest.json').read_text())['files_sha256'].items():
        if Path(name).name!=name or digest(tools/name)!=h:raise ValueError('Delivered clock diagnostic source changed')
    runtime=assemble_runtime(request,tools,output.parent)
    from level4.artifacts import runtime_commit,read_json,write_json,utc_now,checked_pool
    from level4.shared_input import exact_numeric_packet
    from level4.candidate_interface import diagnose_tool_schedule,reduce_clock_events
    runtime_commit(runtime);pool=Path(request['source_pool']);manifest=checked_pool(pool)
    if digest(pool/'pool.json')!=request['source_pool_sha256']:raise ValueError('Frozen source pool changed')
    if read_json(request['prior_candidate_exit'])['exit_code']!=0 or read_json(request['prior_stage_exit'])['exit_code']!=1:
        raise ValueError('Previous execution/failure evidence changed')
    provenance=output/'runtime_provenance';provenance.mkdir()
    shutil.copyfile(runtime.parent/'manifest.json',provenance/'sealed_runtime_manifest.json')
    shutil.copyfile(request['runtime_manifest'],provenance/'parent_runtime_manifest.json')
    sealed=read_json(runtime.parent/'manifest.json')
    shutil.copyfile(sealed['execution_environment_evidence']['path'],provenance/'execution_environment_probe.json')
    shutil.copyfile(pool/'pool.json',output/'source_pool_manifest.json')
    reference=output/'nominal_reference';reference.mkdir();traces=reference/'actual_tools';traces.mkdir()
    nominal=Path(request['nominal_execution'])
    for name in ('reset_returned.npz','after_first_action.npz','static_material_info.npz','capture_report.json'):
        shutil.copyfile(nominal/name,reference/name)
    for index in range(63):shutil.copyfile(nominal/'actual_tools'/('frame_%03d.npz'%index),traces/('frame_%03d.npz'%index))
    shutil.copyfile(request['legacy_source_log'],reference/'legacy_log.pkl')
    shutil.copyfile(pool/'planner_input.npz',reference/'original_planner_input.npz')
    shutil.copyfile(request['prior_candidate_exit'],reference/'prior_candidate001_exit.json')
    shutil.copyfile(request['prior_stage_exit'],reference/'prior_failed_stage_exit.json')
    record=next(r for r in manifest['candidates'] if r['candidate_id']=='candidate_001')
    ledger={r['candidate_id']:{'prediction':'not_run','this_job_execution':'not_executed'} for r in manifest['candidates']}
    ledger['candidate_000']['this_job_execution']='reused_not_rerun'
    ledger['candidate_001']['prior_execution']='completed_prior_failed_job;explicit_repeat_for_missing_clock_events'
    write_json(output/'candidate_status_before.json',ledger)
    target=output/'execution/candidate_001'
    command=[sys.executable,'-B','-u',str(tools/'probe_level4_candidate_interface.py'),'capture',
        '--runtime',str(runtime),'--config',str(pool/record['scene']),'--output',str(target),'--shared-state',str(pool/'shared_source_state.npz')]
    started=time.monotonic()
    with (output/'candidate_001.log').open('xb') as log:result=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
    write_json(output/'candidate_001_exit.json',{'command':command,'exit_code':result.returncode,'elapsed_s':time.monotonic()-started,
        'completed_utc':utc_now(),'candidate_id':'candidate_001','explicit_new_repeat_not_old_run_repair':True})
    ledger['candidate_001']['this_job_execution']='completed' if not result.returncode else 'failed'
    write_json(output/'candidate_status_after_execution.json',ledger)
    if result.returncode:raise RuntimeError('Clock capture failed; partial evidence retained')
    exact_numeric_packet(pool/'shared_source_state.npz',target/'reset_returned.npz')
    exact_numeric_packet(pool/'static_material_info.npz',target/'static_material_info.npz')
    with np.load(target/'after_first_action.npz',allow_pickle=False) as first,np.load(pool/'shared_source_state.npz',allow_pickle=False) as source:
        for key in ('mpm_pos','mpm_vel','mpm_C','mpm_F','mpm_active','tool_000_pos','tool_000_quat','tool_000_vel','tool_000_ang'):
            if not np.array_equal(first[key],source[key]):raise ValueError('First action changed material or ground')
    from omegaconf import OmegaConf
    center=np.asarray(OmegaConf.load(pool/'source_scene.yaml').actions.grasp.to_pos,dtype=float)
    diagnosis=diagnose_tool_schedule(target,reference/'legacy_log.pkl',traces,center,record['parameters_mm_mm_deg'])
    write_json(output/'tool_diagnosis.json',diagnosis)
    clock=reduce_clock_events(target/'clock_events.jsonl',target/'actual_tools')
    write_json(output/'clock_integration_report.json',clock)
    check_files(request['required_files'])
    return {'status':'clock_diagnostic_collection_completed_task_and_model_tools_not_certified',
        'created_utc':utc_now(),'explicit_repeat_for_missing_events':True,'new_simulations':1,'new_Dyn_rollouts':0,
        'clock_replay':clock['status'],'elapsed_times_equal_original_labels':clock['all_elapsed_times_equal_original_labels'],
        'tool_transform_status':diagnosis['tool_transform_status'],'original_runs_remain_failed':True,
        'legacy_model_tool_input_alignment':'not_certified','task_satisfaction':'not_evaluated',
        'candidate_pool_released':False,'main_frame_unchanged':60,'main_label_s_unchanged':1.920}


def collect(request_path,output):
    output.mkdir(exist_ok=False);request=json.loads(request_path.read_text(encoding='utf-8'));started=time.monotonic();status=0
    try:report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1;report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False}
        write(output/'probe_failure.json',report)
    write(output/'diagnostic_report.json',report);write(output/'request.json',request)
    toolkit=output.parent/'toolkit.zip'
    if toolkit.is_file():shutil.copyfile(toolkit,output/'submitted_toolkit.zip')
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-started,'status':report['status'],
        'scope':'event_collection_only_never_task_or_interface_certification','new_simulations_max':1,
        'execution_subprocess_exits_recorded':len(list(output.glob('candidate_*_exit.json'))),'model_prediction_run':False,
        'legacy_model_input_alignment':'not_certified','candidate_pool_released':False})
    write(output/'evidence_manifest.json',{'files_sha256':{p.relative_to(output).as_posix():digest(p)
        for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        'scope':'explicit_new_clock_diagnostic_repeat_not_historical_repair_or_task_result'})
    archive=output.parent/'diagnostic_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(output).as_posix())
    print(json.dumps({'exit_code':status,'status':report['status'],'archive':str(archive),'archive_sha256':digest(archive)}),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();sys.exit(collect(args.request.resolve(),args.output.resolve()))
