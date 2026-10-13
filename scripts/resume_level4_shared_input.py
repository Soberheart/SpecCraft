"""CPU-only preparation from immutable saved Renderer/AE input; no GPU imports."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np
try:
    from prepare_level4_shared_input import digest, write, assemble_runtime
except ModuleNotFoundError as exc:
    if exc.name!='prepare_level4_shared_input': raise
    from scripts.prepare_level4_shared_input import digest, write, assemble_runtime


def environment():
    return {'python':sys.version,'packages':dict(sorted((d.metadata['Name'].lower(),d.version)
        for d in importlib.metadata.distributions() if d.metadata.get('Name')))}


def check_files(entries):
    for name,row in entries.items():
        source=Path(name)
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing source or symlink: '+name)
        before=source.stat()
        if before.st_size!=row['size'] or digest(source)!=row['sha256']:
            raise ValueError('Frozen CPU prerequisite changed: '+name)
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Source changed while checking: '+name)


def run(request,tools,output):
    if (request.get('kind')!='cpu_saved_initial_input_resume_v1'
            or request.get('new_simulations')!=0 or request.get('new_Dyn_rollouts')!=0
            or request.get('GPU_calls')!=0 or request.get('candidate_pool_released') is not False):
        raise ValueError('Only CPU saved-input preparation is authorized by this request')
    actual=environment(); write(output/'cpu_environment.json',actual)
    if actual!=request['expected_cpu_environment']:
        raise ValueError('Frozen CPU preparation environment changed')
    check_files(request['required_files'])
    manifest=json.loads((tools/'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name,h in manifest.items():
        if Path(name).name!=name or digest(tools/name)!=h:
            raise ValueError('Delivered CPU tool changed: '+name)
    source=Path(request['saved_input'])
    original=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    if digest(source/'metadata.json')!=request['source_metadata_sha256']:
        raise ValueError('Saved model input metadata changed')
    if set(original['input_sha256'])!={'model_config.yaml','observations.npz','shared_source_state.npz',
                                      'static_material_info.npz','logged_frame0.npz','initialization_evidence.json'}:
        raise ValueError('Unexpected saved-input members; future material excluded')
    files=assemble_runtime(request,tools,output.parent,cpu_preparation_environment=actual)
    from level4.artifacts import runtime_commit,read_json,write_json,sha256,utc_now,checked_pool
    from level4.shared_input import checked_initial_input
    runtime_commit(files)
    arrays=checked_initial_input(source,original)
    scene_source=Path(request['scene_config']); checkpoint=Path(request['checkpoint'])
    if sha256(scene_source)!=original['scene_config_sha256'] or sha256(checkpoint)!=original['checkpoint_sha256']:
        raise ValueError('Saved encoding scene/checkpoint bodies differ')
    if (output.parent==source or source in output.parent.parents
            or output.parent==files or files in output.parent.parents):
        raise ValueError('CPU output must be separate from source')
    provenance=output/'provenance';provenance.mkdir()
    shutil.copyfile(source/'metadata.json',provenance/'original_model_input_metadata.json')
    shutil.copyfile(request['source_evidence_manifest'],provenance/'source_evidence_manifest.json')
    shutil.copyfile(request['runtime_manifest'],provenance/'parent_runtime_manifest.json')
    shutil.copyfile(files.parent/'manifest.json',provenance/'cpu_sealed_runtime_manifest.json')
    inputs=output/'model_input';inputs.mkdir();scene=output/'source_scene';scene.mkdir()
    shutil.copyfile(scene_source,scene/'config.yaml')
    for name in original['input_sha256']:
        shutil.copyfile(source/name,inputs/name)
        if sha256(inputs/name)!=original['input_sha256'][name]:
            raise ValueError('Saved input copy differs: '+name)
    derived=dict(original,scene_dir=str(scene),checkpoint=str(checkpoint),cpu_resume_provenance={
        'created_utc':utc_now(),'source_run_id':request['source_run_id'],
        'original_metadata_sha256':sha256(source/'metadata.json'),
        'original_evidence_manifest_sha256':sha256(request['source_evidence_manifest']),
        'source_paths':{'input':str(source),'scene':original['scene_dir'],'checkpoint':original['checkpoint']},
        'only_path_fields_derived':True,'model_or_Renderer_recomputed':False,
        'historical_failed_job_reclassified':False})
    write_json(inputs/'metadata.json',derived)
    checked_initial_input(inputs,derived)
    from level4.pipeline import prepare
    prepared=prepare(inputs,output/'pool',tools/'pool_config.json')
    pool=checked_pool(output/'pool')
    with np.load(output/'pool/planner_input.npz',allow_pickle=False) as p:
        for key,value in arrays.items():
            if not np.array_equal(p[key],value) or p[key].dtype!=value.dtype:
                raise ValueError('Prepared input changed saved encoding/state: '+key)
    prohibited=[n for n in sys.modules if n=='torch' or n.startswith(('torch.','net.model','net.pipeline.renderer','taichi','sim.mpm'))]
    if prohibited: raise ValueError('CPU preparation imported a GPU/model/simulator dependency: '+str(prohibited))
    check_files(request['required_files'])
    report={'status':'prepared_not_predicted_or_executed','created_utc':utc_now(),
        'source_run_id':request['source_run_id'],'source_job_remains_failed':True,'pool':prepared,
        'saved_encoding_and_material_preserved_exactly':True,'initial_array_shapes':{k:list(v.shape) for k,v in arrays.items()},
        'query_cell_count':pool['query_cell_count'],'initial_reference_mm':pool['initial_reference_mm'],
        'candidate_prediction':'not_run','candidate_execution':'not_executed','candidate_pool_released':False,
        'task_thresholds_calibrated':False,'new_Dyn_rollouts':0,'new_simulations':0,'GPU_calls':0,
        'runtime_imports_GPU_model_simulator':prohibited,'pool_sha256':sha256(output/'pool/pool.json'),
        'CPU_environment_only':'GPU_prediction_execution_require_separate_reviewed_environment_seal',
        'versions':{'numpy':np.__version__,'source_metadata_sha256':request['source_metadata_sha256']}}
    write_json(output/'cpu_preparation_report.json',report)
    return report


def collect(request_path,output):
    output.mkdir(parents=True,exist_ok=False)
    request=json.loads(request_path.read_text(encoding='utf-8'))
    started=time.monotonic();status=0
    try:
        report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1;report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc()}
        write(output/'cpu_failure.json',report)
    write(output/'request.json',request)
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-started,
        'scientific_status':report['status'],'new_simulation_run':False,'model_prediction_run':False,'GPU_calls':0})
    write(output/'evidence_manifest.json',{'files_sha256':{p.relative_to(output).as_posix():digest(p)
        for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        'scope':'CPU_prepared_candidates_only_no_scientific_task_result'})
    archive=output.parent/'cpu_preparation_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(output).as_posix())
    print(json.dumps({'exit_code':status,'status':report['status'],'archive':str(archive),
        'archive_sha256':digest(archive),'GPU_calls':0,'candidate_pool_released':False}),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.exit(collect(args.request.resolve(),args.output.resolve()))
