"""Render/encode only new frame0; prepare an uncalibrated pool, no Dyn or sim."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as out: json.dump(value,out,indent=2,allow_nan=False)


def validate_preprocess_input(tools,report_path):
    """CPU pre-qsub/compute check of numeric transport and native pickle roundtrip."""
    tools=Path(tools)
    manifest=json.loads((tools/'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name in ('initial_frame_source.py','nominal_first_frame.npz','nominal_first_frame.json','logged_frame0.npz'):
        if name not in manifest or digest(tools/name)!=manifest[name]:
            raise ValueError('Numeric frame transport source changed: '+name)
    spec=importlib.util.spec_from_file_location('_level4_numeric_initial_frame',tools/'initial_frame_source.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frame=module.load_frame(tools/'nominal_first_frame.npz',tools/'nominal_first_frame.json')
    reference=module.validate_material_reference(frame,tools/'logged_frame0.npz')
    native=pickle.dumps([frame],protocol=4)
    decoded=pickle.loads(native)
    if len(decoded)!=1: raise ValueError('Native initial-frame pickle must contain exactly one frame')
    module.assert_same_frame(frame,decoded[0])
    write(report_path,dict(reference,status='passed_CPU_initial_frame_native_roundtrip',
        python=sys.version,numpy=np.__version__,transport='NPZ_allow_pickle_false_and_JSON',
        native_pickle_producer_numpy=np.__version__,native_pickle_protocol=4,
        all_frame_fields_roundtrip_exactly=True,new_simulation_run=False,model_prediction_run=False,
        transported_npz_sha256=digest(tools/'nominal_first_frame.npz'),
        transported_json_sha256=digest(tools/'nominal_first_frame.json')))
    return frame


def validate_level4_closure(files):
    """Check local imports in the frozen package, including deferred imports.

    An import of pipeline alone cannot expose dependencies inside stage functions.
    This check runs before Renderer/AE and never imports CUDA/simulator libraries.
    """
    import ast
    files=Path(files)
    for source in (files/'level4').glob('*.py'):
        tree=ast.parse(source.read_text(encoding='utf-8'),feature_version=(3,9))
        for node in ast.walk(tree):
            if not isinstance(node,ast.ImportFrom): continue
            if node.level==1 and node.module:
                module='level4.'+node.module
            elif node.level==0 and node.module and node.module.startswith('level4.'):
                module=node.module
            else: continue
            target=files.joinpath(*module.split('.'))
            if not target.with_suffix('.py').is_file() and not (target/'__init__.py').is_file():
                raise ValueError('Frozen Level4 dependency missing: '+module+' (from '+source.name+')')


def assemble_runtime(request,tools,parent,*,cpu_preparation_environment=None):
    """Copy the frozen archived runtime, then apply reviewed Level4 overlays."""
    from datetime import datetime,timezone
    original=Path(request['runtime'])
    runtime=parent/'runtime'; runtime.mkdir(); files=runtime/'files'; files.mkdir()
    manifest=json.loads(Path(request['runtime_manifest']).read_text(encoding='utf-8'))
    original_hashes=manifest['files_sha256'].copy()
    for name,h in original_hashes.items():
        relative=Path(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or relative.as_posix()!=name):
            raise ValueError('Unexpected runtime source path')
        source=original/name
        if source.is_symlink() or digest(source)!=h: raise ValueError('Runtime source changed: '+name)
        dest=files/name; dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as inp,dest.open('xb') as out: shutil.copyfileobj(inp,out)
        if digest(dest)!=h or digest(source)!=h: raise ValueError('Runtime source changed during copy: '+name)
    for name,tool_name in request['runtime_overlays'].items():
        # Copy into a new runtime only. Old runtime/experiments stay untouched.
        dest=files/name
        if (not name.startswith(('level4/','scripts/')) or '..' in Path(name).parts
                or '\\' in name or ':' in name or Path(name).as_posix()!=name
                or Path(tool_name).name!=tool_name):
            raise ValueError('Unexpected overlay path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if name in original_hashes:
            saved=runtime/('original_'+Path(name).name)
            with dest.open('rb') as inp,saved.open('xb') as out: shutil.copyfileobj(inp,out)
        with (tools/tool_name).open('rb') as inp,dest.open('wb' if name in original_hashes else 'xb') as out:
            shutil.copyfileobj(inp,out)
        manifest['files_sha256'][name]=digest(dest)
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_runtime_manifest_sha256=digest(request['runtime_manifest']),
        new_level4_overlay_hashes={n:manifest['files_sha256'][n] for n in request['runtime_overlays']},
        scope='Archived runtime copied; new shared initial input and guarded prepare overlay, no historical data mutation')
    if cpu_preparation_environment is not None:
        if request.get('kind')!='cpu_saved_initial_input_resume_v1':
            raise ValueError('CPU resealing is restricted to saved-input preparation')
        manifest['parent_environment_provenance']={k:manifest[k] for k in
            ('environment','execution_environment','execution_environment_evidence') if k in manifest}
        manifest['environment']=cpu_preparation_environment
        manifest.pop('execution_environment',None)
        manifest.pop('execution_environment_evidence',None)
        manifest['stage_scope']='CPU_saved_input_preparation_only_GPU_execution_requires_new_seal'
    validate_level4_closure(files)
    write(runtime/'manifest.json',manifest)
    os.environ['LEVEL4_SUBMISSION_MANIFEST']=str(runtime/'manifest.json')
    os.environ['LEVEL4_SUBMISSION_MANIFEST_SHA256']=digest(runtime/'manifest.json')
    sys.path.insert(0,str(files))
    os.chdir(files)
    return files


def run(request,tools,output):
    actual={'python':sys.version,'packages':dict(sorted((d.metadata['Name'].lower(),d.version)
            for d in importlib.metadata.distributions() if d.metadata.get('Name')))}
    write(output/'environment.json',actual)
    if actual!=request['expected_compute_environment']: raise ValueError('Reviewed compute environment changed')
    for name,row in request['required_files'].items():
        p=Path(name)
        if p.is_symlink() or p.stat().st_size!=row['size'] or digest(p)!=row['sha256']:
            raise ValueError('Frozen prerequisite changed: '+name)
    tool_manifest=json.loads((tools/'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name,h in tool_manifest.items():
        if digest(tools/name)!=h: raise ValueError('Delivered source/input changed: '+name)
    frame=validate_preprocess_input(tools,output/'compute_initial_frame_check.json')
    runtime=assemble_runtime(request,tools,output.parent)
    from level4.artifacts import runtime_commit,write_json,utc_now,sha256
    runtime_commit(runtime)  # Exact source/assets/environment guard, no compute-node Git.
    provenance=output/'runtime_provenance'; provenance.mkdir()
    shutil.copyfile(runtime.parent/'manifest.json',provenance/'sealed_runtime_manifest.json')
    shutil.copyfile(request['runtime_manifest'],provenance/'parent_runtime_manifest.json')
    sealed=json.loads((runtime.parent/'manifest.json').read_text(encoding='utf-8'))
    shutil.copyfile(sealed['execution_environment_evidence']['path'],provenance/'execution_environment_probe.json')
    for path in runtime.parent.glob('original_*.py'):
        shutil.copyfile(path,provenance/path.name)
    from omegaconf import OmegaConf
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA needed for the original Renderer/AE; no simulation required')
    torch.cuda.set_device(0)
    from net.pipeline.random import manual_seed
    manual_seed(request['observation_seed'])

    # One explicitly derived frame only. It is a preprocessing input, never
    # presented as a newly executed trajectory or a completed scientific run.
    scene=output/'initial_preprocess'; scene.mkdir()
    shutil.copyfile(tools/'scene_config.yaml',scene/'config.yaml')
    # Create the derived preprocessing pickle using THIS host's NumPy. Never
    # transport NumPy2 pickle references into the frozen NumPy1 environment.
    with (scene/'log.pkl').open('xb') as out: pickle.dump([frame],out,protocol=4)
    shutil.copyfile(tools/'nominal_first_frame.npz',output/'transported_initial_frame.npz')
    shutil.copyfile(tools/'nominal_first_frame.json',output/'transported_initial_frame.json')
    spec=importlib.util.spec_from_file_location('_level4_first_frame_processor',runtime/'sim/process.py')
    processor=importlib.util.module_from_spec(spec); spec.loader.exec_module(processor)
    process_cfg=OmegaConf.load(runtime/'sim/process/config/common.yaml')
    process_cfg.num_frames=1; process_cfg.num_processes=1
    OmegaConf.resolve(process_cfg)
    OmegaConf.save(process_cfg,output/'process_config.yaml')
    class Writer:
        def __init__(self): self.items=[]
        def write(self,lock,item,scene_idx,frame_idx):
            if scene_idx!=0 or frame_idx!=0: raise ValueError('Only initial frame may enter observation preprocessing')
            self.items.append(copy.deepcopy(item))
    writer=Writer()
    processor.process_sequence(0,str(scene),writer,process_cfg,DEBUG=True,total=1)
    if len(writer.items)!=1: raise ValueError('Initial mesh missing or padded')
    mesh=writer.items[0]
    with (output/'processed_initial_mesh.npz').open('xb') as out:
        np.savez_compressed(out,**mesh)
    cfg=OmegaConf.load(request['model_config'])
    OmegaConf.resolve(cfg)
    cfg.settings.ddp=False; cfg.settings.test_only=True; cfg.log_wandb=False; cfg.cuda_id=0
    if cfg.model.condition.ee_shape!='both' or cfg.dimensions.input_dim!=4:
        raise ValueError('Unexpected frozen model interface')
    if float(cfg.augmentation.render.unit_to_meter)!=.2:
        raise ValueError('Frozen model and new physical input scale differ')
    from scripts.check_level3_dyn import canonical_model_state
    from net.model.model import Predictor
    from net.pipeline.renderer import Renderer
    device=torch.device('cuda:0'); model=Predictor(cfg).to(device).eval()
    model.load_state_dict(canonical_model_state(torch.load(request['checkpoint'],map_location='cpu')['model_state_dict']),strict=True)
    render_cfg=OmegaConf.create(OmegaConf.to_container(cfg.augmentation.render,resolve=True))
    renderer=Renderer(render_cfg,device=device).eval()
    fields=('obj_verts','obj_faces','obj_vert_labels','obj_face_labels','ee_verts','ee_faces','genus')
    data={k:torch.as_tensor(mesh[k],device=device)[None] for k in fields}
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    manual_seed(request['observation_seed'])
    with torch.no_grad():
        observation=renderer(data,postfix='',num_samples=cfg.dataset.n_points)
        if (tuple(observation.shape)!=(1,cfg.dataset.n_points,4) or not torch.isfinite(observation).all()
                or not (observation[...,:3]!=0).any()): raise ValueError('Invalid new initial observation')
        _,_,latent=model.reconstruct(observation,None,decode=False)
        if not torch.isfinite(latent).all(): raise ValueError('Invalid new initial encoding')
        _,_,repeated=model.reconstruct(observation,None,decode=False)
        if not torch.allclose(latent,repeated,rtol=1e-4,atol=1e-5): raise ValueError('Initial encoding not repeatable')
    # Editable third-party installs may still resolve outside the snapshot.
    # Preserve actual import locations instead of claiming full environment recovery.
    modules={}
    for name in ('torch','open3d','sdftoolbox','nvdiffrast','nvdiffrast.torch',
                 'net.pipeline.renderer','net.model.model','sim.util'):
        module=sys.modules.get(name)
        path=getattr(module,'__file__',None)
        if path:
            modules[name]={'path':str(Path(path).resolve()),'sha256':sha256(path)}
    write_json(provenance/'imported_module_paths.json',{
        'modules':modules,'full_installed_environment_recovery':'pending',
        'scope':'Actual loaded module bodies referenced by path/hash; installed package bodies are not all bundled'})
    inputs=output/'model_input'; inputs.mkdir()
    for name in ('shared_source_state.npz','static_material_info.npz','logged_frame0.npz','initialization_evidence.json'):
        shutil.copyfile(tools/name,inputs/name)
    OmegaConf.save(cfg,inputs/'model_config.yaml')
    with np.load(tools/'logged_frame0.npz',allow_pickle=False) as state:
        particles=state['mpm_pos'][state['mpm_active'].astype(bool)]
    with np.load(tools/'legacy_action_input.npz',allow_pickle=False) as old:
        tools_array,times,goal=(old[k] for k in ('tools','times_s','z_goal'))
    with (inputs/'observations.npz').open('xb') as out:
        np.savez_compressed(out,observed_initial=observation[0].cpu().numpy(),z_initial=latent[0].cpu().numpy(),
                            z_goal=goal,tools=tools_array,times_s=times,initial_particles=particles)
    inputs_used=('model_config.yaml','observations.npz','shared_source_state.npz','static_material_info.npz',
                 'logged_frame0.npz','initialization_evidence.json')
    metadata={'kind':'shared_initial_observation_input_v1','created_utc':utc_now(),
        'scene_dir':str(scene),'scene_config_sha256':sha256(scene/'config.yaml'),
        'checkpoint':request['checkpoint'],'checkpoint_sha256':request['required_files'][request['checkpoint']]['sha256'],
        'seed':request['observation_seed'],'unit_to_meter':.2,'rollout_frames':request['rollout_frames'],
        'execution_required_s':(np.arange(61)*.032).tolist(),'historical_initial_observation_reused':False,
        'observation_source':'original preprocessing and Renderer from new captured nominal frame0; only this material frame rendered',
        'tool_schedule_source':'whitelisted existing nominal tool pointcloud schedule; candidate interface still needs execution audit',
        'point_goal_source':'existing Level3 frame60 latent, extra representative target; not certified A/B feasible',
        'time_origin':'legacy logged frame0 after first solver step; primary frame60 is +1.920 s; auxiliary62 +1.984 s',
        'reset_pre_solver_time_relative_to_logged_origin_s':-.002,
        'input_sha256':{n:sha256(inputs/n) for n in inputs_used}}
    write_json(inputs/'metadata.json',metadata)
    write_json(output/'encoding_report.json',{'status':'initial_Renderer_AE_completed_before_pool_preparation',
        'checkpoint_loaded_strictly':True,'reencoding_check':'passed',
        'initial_observation_shape':list(observation.shape),'initial_latent_shape':list(latent.shape),
        'input_metadata_sha256':sha256(inputs/'metadata.json'),'new_Dyn_rollouts':0,'new_simulations':0})
    from level4.shared_input import checked_initial_input
    checked_initial_input(inputs,metadata)
    del model,renderer; torch.cuda.empty_cache()
    from level4.pipeline import prepare
    prepared=prepare(inputs,output/'pool',tools/'pool_config.json')
    report={'created_utc':utc_now(),'kind':'new_shared_initial_observation_and_pool_preparation',
        'status':'prepared_not_predicted_or_executed','initial_observation_shape':list(observation.shape),
        'initial_latent_shape':list(latent.shape),'checkpoint_loaded_strictly':True,
        'reencoding_check':'passed','new_initial_geometry_source':'25770711.pbs101 attempt1 recorded frame0',
        'historical_initial_observation_reused':False,'pool':prepared,'candidate_prediction':'not_run',
        'candidate_execution':'not_run','new_simulation_run':False,'task_thresholds_calibrated':False,
        'candidate_pool_released':False,'gpu':torch.cuda.get_device_name(0)}
    write_json(output/'input_preparation_report.json',report)
    return report


def collect(request_path,output):
    output.mkdir(exist_ok=False)
    start=time.monotonic(); request=json.loads(request_path.read_text(encoding='utf-8'))
    status=0
    try:
        report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1
        report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False}
        write(output/'input_failure.json',report)
    write(output/'request.json',request)
    toolkit=output.parent/'toolkit.zip'
    if toolkit.is_file(): shutil.copyfile(toolkit,output/'submitted_toolkit.zip')
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-start,'scientific_status':report['status'],
        'model_prediction_run':False,'new_simulation_run':False})
    # Hash all completed/partial artifacts. Preserve failures as well as success.
    hashes={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}
    write(output/'evidence_manifest.json',{'files_sha256':hashes,'scope':'initial observation/prepared pool only; no task result'})
    archive=output.parent/'diagnostic_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(output).as_posix())
    print(json.dumps({'status':report['status'],'archive':str(archive),'archive_sha256':digest(archive),'exit_code':status}),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-initial-input',type=Path); parser.add_argument('--validation-report',type=Path)
    args=parser.parse_args()
    if args.validate_initial_input is not None:
        if args.validation_report is None: parser.error('--validation-report required for CPU input validation')
        validate_preprocess_input(args.validate_initial_input.resolve(),args.validation_report.resolve())
        print('CPU_INITIAL_FRAME_NATIVE_ROUNDTRIP_PASSED; NO_GPU_OR_SIMULATION',flush=True)
        sys.exit(0)
    if args.request is None or args.output is None: parser.error('--request and --output required for preparation')
    sys.exit(collect(args.request.resolve(),args.output.resolve()))
