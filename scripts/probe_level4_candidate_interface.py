"""Two fixed independent executions to audit shared material/tool interfaces.

No Renderer, AE or Dyn; no task success claim or release of the other candidates.
"""
import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import weakref
import zipfile
import numpy as np
try:
    from prepare_level4_shared_input import assemble_runtime,digest,write
    from probe_level4_initialization import Recorder,array,attach_recorder
    from resume_level4_shared_input import check_files,environment
except ModuleNotFoundError as exc:
    if exc.name not in ('prepare_level4_shared_input','probe_level4_initialization','resume_level4_shared_input'):raise
    from scripts.prepare_level4_shared_input import assemble_runtime,digest,write
    from scripts.probe_level4_initialization import Recorder,array,attach_recorder
    from scripts.resume_level4_shared_input import check_files,environment


def attach_clock_capture(scene,output):
    """Observe active-branch integration and state rollback; never step it.

    A topology probe can integrate a temporary branch, restore a saved state,
    then integrate an additional retained step. Scene.t counts both branches;
    the solver global counter is reset on restore. Neither is elapsed time.
    Keep weak references to states, so observations do not retain GPU tensors.
    Unknown restore sources make the logical integration clock unknown.
    """
    clock={'active_substeps':0,'event_count':0,'next_snapshot':0}
    snapshots={};path=output/'clock_events.jsonl'
    path.touch(exist_ok=False)
    def event(kind,**fields):
        clock['event_count']+=1
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=clock['event_count'],kind=kind,
                active_substeps=clock['active_substeps'],**fields),allow_nan=False)+'\n')
    original_get=scene.get_state
    def get_state(*args,**kwargs):
        state=original_get(*args,**kwargs);identity=id(state)
        clock['next_snapshot']+=1;snapshot=clock['next_snapshot']
        entry={'id':snapshot,'active_substeps':clock['active_substeps']}
        try:
            entry['reference']=weakref.ref(state,lambda ref,ident=identity:snapshots.pop(ident,None))
        except TypeError:
            event('untrackable_snapshot',snapshot=snapshot)
        else:
            snapshots[identity]=entry;event('get_state',snapshot=snapshot)
        return state
    scene.get_state=get_state
    original_set=scene.set_state
    def set_state(state,*args,**kwargs):
        before=clock['active_substeps'];entry=snapshots.get(id(state))
        known=entry is not None and entry['reference']() is state
        result=original_set(state,*args,**kwargs)
        clock['active_substeps']=entry['active_substeps'] if known else None
        event('set_state',source_snapshot=entry['id'] if known else None,
              before_active_substeps=before,solver_counter_after=int(scene.sim.cur_substep_global))
        return result
    scene.set_state=set_state
    original_reset=scene.reset
    def reset(*args,**kwargs):
        result=original_reset(*args,**kwargs);clock['active_substeps']=0
        event('reset_origin',solver_counter_after=int(scene.sim.cur_substep_global))
        return result
    scene.reset=reset
    original_step=scene.step
    def step(*args,**kwargs):
        before=int(scene.sim.cur_substep_global);n=int(scene.sim.n_substeps)
        result=original_step(*args,**kwargs)
        if clock['active_substeps'] is not None:clock['active_substeps']+=n
        event('step',n_substeps=n,solver_counter_before=before,solver_counter_after=int(scene.sim.cur_substep_global))
        return result
    scene.step=step
    return clock


def attach_action_capture(workspace,recorder,tools,indices,output):
    """Observe existing transitions; never process input or advance a solver."""
    scene=getattr(workspace,'scene',None)
    clock=attach_clock_capture(scene,output) if scene is not None and all(
        hasattr(scene,name) for name in ('get_state','set_state','reset','step','sim')) else None
    original=workspace.planner.step;first=[True]
    def step(goals):
        result=original(goals)
        if first[0]:
            first[0]=False;recorder.save(workspace,'after_first_action')
            targets={}
            for i in indices:
                for key in ('pos','quat','vel','ang'):
                    target=tools[i]._tgt.get(key)
                    if target is None:raise ValueError('First action has incomplete tool target: '+key)
                    targets['tool_%03d_%s'%(i,key)]=array(target)
            np.savez_compressed(output/'first_action_targets.npz',**targets)
        return result
    workspace.planner.step=step
    original_log=workspace.log;trace_dir=output/'actual_tools';trace_dir.mkdir()
    raw_dir=output/'raw_frames';raw_dir.mkdir()
    def log(step,*args,**kwargs):
        packet={'step':np.asarray(step,dtype=np.int64)}
        sim=getattr(workspace.scene,'sim',None) if hasattr(workspace,'scene') else None
        if sim is not None:
            previous=(int(sim.cur_substep_global)-1)%int(sim.max_substeps_local)
            packet.update(sample_global_substep=np.asarray(sim.cur_substep_global,dtype=np.int64),
                current_local_substep=np.asarray(sim.cur_substep_local,dtype=np.int64),
                previous_local_substep=np.asarray(previous,dtype=np.int64),substep_dt=np.asarray(sim.substep_dt))
        if clock is not None:
            known=clock['active_substeps'] is not None
            packet.update(active_branch_clock_known=np.asarray(known),
                active_branch_substeps=np.asarray(clock['active_substeps'] if known else -1,dtype=np.int64),
                clock_event_count=np.asarray(clock['event_count'],dtype=np.int64))
        for i,tool in enumerate(tools):
            state=tool.get_state()
            for key in ('pos','quat','vel','ang'):
                packet['tool_%03d_%s'%(i,key)]=array(getattr(state,key))
            if sim is not None:
                previous_state=tool.get_state(previous)
                for key in ('pos','quat','vel','ang'):
                    packet['previous_tool_%03d_%s'%(i,key)]=array(getattr(previous_state,key))
                packet['latest_tool_%03d_pos'%i]=array(tool.latest_pos.to_numpy()[0])
                packet['latest_tool_%03d_quat'%i]=array(tool.latest_quat.to_numpy()[0])
        with (trace_dir/('frame_%03d.npz'%(step//16))).open('xb') as out:
            np.savez_compressed(out,**packet)
        row=original_log(step,*args,**kwargs)
        with (raw_dir/('frame_%03d.pkl'%(step//16))).open('xb') as out:pickle.dump(row,out,protocol=4)
        return row
    workspace.log=log


def capture(runtime,config,output,shared_state):
    output.mkdir(parents=True,exist_ok=False);recorder=Recorder(output)
    try:
        os.chdir(runtime);sys.path.insert(0,str(runtime))
        from level4.artifacts import runtime_commit,write_json,utc_now
        runtime_commit(runtime)
        from omegaconf import OmegaConf
        spec=importlib.util.spec_from_file_location('_level4_candidate_generator',runtime/'sim/generate.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if module.DEBUG:raise ValueError('Candidate audit requires normal simulator execution')
        cfg=OmegaConf.load(config);cfg.log.base_dir=str(output/'scenes');(output/'scenes').mkdir()
        OmegaConf.save(cfg,output/'requested_config.yaml')
        workspace=module.GenerationWorkspace(cfg)
        static=workspace.scene.sim.mpm_solver.particles_info.to_numpy()
        np.savez_compressed(output/'static_material_info.npz',**{k:array(static[k]) for k in ('mat_idx','mu','lam','mass')})
        tools=workspace.scene.sim.tool_solver.entities
        indices=[tools.index(entity) for entity in workspace.planner.ee.entities]
        write_json(output/'tool_entity_mapping.json',{'names':[e.name for e in tools],'gripper_indices':indices,
            'note':'Actual scene entities; never infer order from a placeholder material ID'})
        attach_recorder(recorder,workspace,shared_state)
        attach_action_capture(workspace,recorder,tools,indices,output)
        workspace.run()
        write_json(output/'capture_report.json',{'status':'captured_not_task_evaluated','created_utc':utc_now(),
            'states':recorder.rows,'shared_state_sha256':digest(shared_state),'no_warmup_replayed':True,
            'first_action_targets_recorded_before_solver_step':True,'candidate_pool_released':False,
            'primary_s':1.920,'auxiliary_s':1.984})
    except BaseException as exc:
        write(output/'capture_failure.json',{'status':'failed','error':repr(exc),
            'traceback':traceback.format_exc(),'states':recorder.rows,'candidate_pool_released':False})
        raise


def run(request,tools,output):
    if (request.get('kind')!='two_candidate_shared_interface_probe_v1' or request['candidate_ids']!=['candidate_000','candidate_001']
            or request['new_Dyn_rollouts']!=0 or request['new_simulations_max']!=2
            or request.get('candidate_pool_released') is not False):
        raise ValueError('Only two frozen candidate interface diagnostics are allowed')
    actual=environment();write(output/'environment.json',actual)
    if actual!=request['expected_compute_environment']:raise ValueError('Reviewed compute environment changed')
    check_files(request['required_files'])
    if shutil.disk_usage(output).free<2*1024**3:raise ValueError('At least2GiB server free needed')
    toolkit=json.loads((tools/'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name,h in toolkit.items():
        if Path(name).name!=name or digest(tools/name)!=h:raise ValueError('Delivered candidate audit source changed')
    runtime=assemble_runtime(request,tools,output.parent)
    from level4.artifacts import runtime_commit,checked_pool,read_json,write_json,utc_now
    runtime_commit(runtime)
    provenance=output/'runtime_provenance';provenance.mkdir()
    shutil.copyfile(runtime.parent/'manifest.json',provenance/'sealed_runtime_manifest.json')
    shutil.copyfile(request['runtime_manifest'],provenance/'parent_runtime_manifest.json')
    sealed=read_json(runtime.parent/'manifest.json')
    shutil.copyfile(sealed['execution_environment_evidence']['path'],provenance/'execution_environment_probe.json')
    from level4.pipeline import prepare
    prepared=prepare(request['saved_input'],output/'pool',tools/'pool_config.json')
    pool=output/'pool';manifest=checked_pool(pool)
    original=checked_pool(request['source_pool'])
    if digest(Path(request['source_pool'])/'pool.json')!=request['source_pool_sha256']:
        raise ValueError('Source CPU pool hash changed')
    for key in ('config_sha256','source_observations_sha256','checkpoint_sha256','initial_reference_mm','query_cell_count'):
        if manifest[key]!=original[key]:raise ValueError('Fresh GPU-sealed pool differs from frozenCPU pool: '+key)
    from level4.candidate_interface import compare_prepared_candidates
    consistency=compare_prepared_candidates(original['candidates'],manifest['candidates'],original['bounds_mm_mm_deg'])
    for name in ('grid.npz','planner_input.npz','shared_source_state.npz','static_material_info.npz'):
        with np.load(pool/name,allow_pickle=False) as a,np.load(Path(request['source_pool'])/name,allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('Prepared numeric input/grid changed on compute node: '+name)
    write_json(output/'prepared_pool_consistency.json',consistency)
    from omegaconf import OmegaConf
    center=np.asarray(OmegaConf.load(pool/'source_scene.yaml').actions.grasp.to_pos,dtype=float)
    results=[]
    ledger={r['candidate_id']:{'execution':'not_executed','interface_audit':'not_run','prediction':'not_run'} for r in manifest['candidates']}
    write_json(output/'candidate_execution_status.json',ledger)
    events=output/'candidate_status_events';events.mkdir();event=[0]
    def save_status():
        write_json(events/('status_%03d.json'%event[0]),ledger);event[0]+=1
    for cid in request['candidate_ids']:
        record=next(r for r in manifest['candidates'] if r['candidate_id']==cid)
        target=pool/'executions'/cid
        command=[sys.executable,'-B','-u',str(tools/'probe_level4_candidate_interface.py'),'capture',
            '--runtime',str(runtime),'--config',str(pool/record['scene']),'--output',str(target),
            '--shared-state',str(pool/'shared_source_state.npz')]
        started=time.monotonic()
        ledger[cid]['execution']='running';save_status()
        with (output/(cid+'.log')).open('xb') as log:
            result=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT)
        exit_record={'command':command,'exit_code':result.returncode,'elapsed_s':time.monotonic()-started,
            'pool_sha256':digest(pool/'pool.json'),'candidate_id':cid,'completed_utc':utc_now(),
            'scientific_scope':'interface_diagnostic_not_planner_selection'}
        write_json(output/(cid+'_exit.json'),exit_record)
        ledger[cid]['execution']='completed' if result.returncode==0 else 'failed'
        save_status()
        if result.returncode:raise RuntimeError('Candidate capture failed: '+cid+'; second probe stopped if not yet started')
        from level4.candidate_interface import audit_candidate
        try:
            report=audit_candidate(target,request['legacy_source_log'],pool/'shared_source_state.npz',pool/'static_material_info.npz',
                pool/'planner_input.npz',center,record['parameters_mm_mm_deg'])
        except Exception as exc:
            ledger[cid]['interface_audit']='failed';save_status()
            write_json(target/'interface_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False})
            raise
        ledger[cid]['interface_audit']='passed';save_status()
        report.update(candidate_id=cid,pool_sha256=digest(pool/'pool.json'),created_utc=utc_now())
        write_json(target/'interface_audit.json',report);results.append(report)
    check_files(request['required_files'])
    report={'status':'two_candidate_interface_passed_no_full_pool_release','created_utc':utc_now(),
        'source_pool_sha256':request['source_pool_sha256'],'pool':prepared,'candidate_results':results,
        'executed_candidates':request['candidate_ids'],'other_candidates':'not_executed','final_candidate_statuses':ledger,
        'new_simulations':len(results),'new_Dyn_rollouts':0,'model_prediction':'not_run','task_success':'not_evaluated',
        'candidate_pool_released':False,'thresholds_calibrated':False,
        'next_step':'review_actual_diagnostics_then_predict_small_pilot; each remaining candidate requires same per-run checks'}
    write_json(output/'candidate_interface_report.json',report)
    return report


def collect(request_path,output):
    output.mkdir(exist_ok=False);started=time.monotonic();status=0
    request=json.loads(request_path.read_text(encoding='utf-8'))
    try:report=run(request,request_path.parent,output)
    except Exception as exc:
        status=1;report={'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'candidate_pool_released':False}
        write(output/'probe_failure.json',report)
    write(output/'request.json',request)
    toolkit=output.parent/'toolkit.zip'
    if toolkit.is_file():shutil.copyfile(toolkit,output/'submitted_toolkit.zip')
    write(output/'stage_exit.json',{'exit_code':status,'elapsed_s':time.monotonic()-started,'status':report['status'],
        'scope':'interface_diagnostic','model_prediction_run':False,'candidate_pool_released':False,
        'execution_subprocess_exits_recorded':len(list(output.glob('candidate_*_exit.json'))),
        'interface_audits_passed':len(list(output.glob('pool/executions/*/interface_audit.json'))),
        'requested_candidates':request.get('candidate_ids',[]),'new_simulations_max':2,
        'failure_or_interruption_raw_frames_preserved':True})
    write(output/'evidence_manifest.json',{'files_sha256':{p.relative_to(output).as_posix():digest(p)
        for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        'scope':'real_two_candidate_execution_interface_diagnostic_not_task_results'})
    archive=output.parent/'diagnostic_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(output).as_posix())
    print(json.dumps({'exit_code':status,'status':report['status'],'archive':str(archive),'archive_sha256':digest(archive)}),flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    child=sub.add_parser('capture')
    for name in ('runtime','config','output','shared-state'):child.add_argument('--'+name,type=Path,required=True)
    outer=sub.add_parser('collect');outer.add_argument('--request',type=Path,required=True);outer.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='capture':capture(args.runtime.resolve(),args.config.resolve(),args.output.resolve(),args.shared_state.resolve())
    else:sys.exit(collect(args.request.resolve(),args.output.resolve()))
