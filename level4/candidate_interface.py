"""CPU audit of candidate-independent material and candidate-specific tools.

This diagnoses an execution interface, never selects an action using truth and
never certifies the other candidates or any task threshold.
"""
from pathlib import Path
import json
import math
import pickle
import numpy as np
from .artifacts import read_json,sha256
from .shared_input import exact_numeric_packet
from scripts.plan_level3_pose_cem import compose_yaw,transform_tools


def numeric(value,shape,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid numeric '+name)
    return value


def inspect_tool_clock(packet,frame):
    """Validate a solver-buffer index, without treating it as an elapsed clock.

    Topology.check can call Scene.set_state and reset cur_substep_global. The
    captured counter is still useful for exact previous/latest comparisons.
    It cannot establish accumulated physical time across those state restores.
    """
    def integer(key):
        value=np.asarray(packet[key])
        if value.shape!=() or value.dtype.kind not in 'iu' or int(value)<0:
            raise ValueError('Invalid captured tool clock: '+key)
        return int(value)
    step=integer('step');global_index=integer('sample_global_substep')
    current=integer('current_local_substep');previous=integer('previous_local_substep')
    dt=np.asarray(packet['substep_dt'])
    if (step!=frame*16 or dt.shape!=() or float(dt)!=.0002
            or current!=global_index%50 or previous!=(global_index-1)%50):
        raise ValueError('Inconsistent captured current/previous tool indices')
    expected=(frame*16+1)*10
    return {'frame':frame,'global_solver_counter':global_index,'current_local_substep':current,
            'previous_local_substep':previous,'local_indices_consistent':True,
            'matches_no_reset_counter':global_index==expected,
            'no_reset_counter_reference':expected,'nominal_label_s':frame*.032,
            'elapsed_physical_time_certified':False,
            'clock_scope':'resettable_solver_counter; state-restore event history required for elapsed time'}


def reduce_clock_events(event_path,tool_directory,frame_count=63):
    """Recompute active-branch integration from observed calls and rollbacks.

    This certifies the *recorded integration count*, not task satisfaction or
    the protocol's time labels. Counter/state edits or untracked restores fail.
    """
    events=[json.loads(line) for line in Path(event_path).read_text(encoding='utf-8').splitlines()]
    if not events or frame_count<1:raise ValueError('Missing scene clock event history')
    active=0;counter=0;snapshots={};states={0:0};reset_count=restore_count=step_count=0
    def integer(value):
        if type(value) is not int or value<0:raise ValueError('Invalid clock event integer')
        return value
    for index,event in enumerate(events,1):
        if integer(event['event'])!=index:raise ValueError('Clock event sequence changed')
        kind=event['kind']
        if kind=='reset_origin':
            if integer(event['solver_counter_after'])!=0:raise ValueError('Scene reset did not reset solver counter')
            active=0;counter=0;reset_count+=1
        elif kind=='get_state':
            snapshot=integer(event['snapshot'])
            if snapshot==0 or snapshot in snapshots:raise ValueError('Duplicate scene clock snapshot')
            snapshots[snapshot]=active
        elif kind=='step':
            n=integer(event['n_substeps'])
            if n!=10 or integer(event['solver_counter_before'])!=counter or integer(event['solver_counter_after'])!=counter+n:
                raise ValueError('Scene step count or solver counter changed')
            active+=n;counter+=n;step_count+=1
        elif kind=='set_state':
            snapshot=event['source_snapshot']
            if snapshot is None or integer(snapshot) not in snapshots:
                raise ValueError('Unknown scene restore source; active-branch time is not certified')
            if integer(event['before_active_substeps'])!=active or integer(event['solver_counter_after'])!=0:
                raise ValueError('Scene restore clock changed')
            active=snapshots[snapshot];counter=0;restore_count+=1
        else:raise ValueError('Unknown/untrackable clock event')
        if integer(event['active_substeps'])!=active:raise ValueError('Recorded active-branch clock differs from event replay')
        states[index]=active
    if reset_count!=1:raise ValueError('Expected exactly one shared reset origin')
    frames=[]
    for frame in range(frame_count):
        with np.load(Path(tool_directory)/('frame_%03d.npz'%frame),allow_pickle=False) as packet:
            inspect_tool_clock(packet,frame)
            known=np.asarray(packet['active_branch_clock_known'])
            if known.shape!=() or known.dtype.kind!='b' or not bool(known):raise ValueError('Per-frame active-branch clock unknown')
            event_index=np.asarray(packet['clock_event_count']);value=np.asarray(packet['active_branch_substeps'])
            if (event_index.shape!=() or value.shape!=() or event_index.dtype.kind not in 'iu' or value.dtype.kind not in 'iu'
                    or int(event_index) not in states or states[int(event_index)]!=int(value)):
                raise ValueError('Per-frame clock disagrees with event prefix')
            frames.append({'frame':frame,'nominal_label_s':frame*.032,'active_branch_substeps':int(value),'event_prefix':int(event_index)})
    origin=frames[0]['active_branch_substeps']
    for row in frames:
        row['active_branch_elapsed_s']=(row['active_branch_substeps']-origin)*.0002
        row['elapsed_minus_label_s']=row['active_branch_elapsed_s']-row['nominal_label_s']
    if any(b['active_branch_substeps']<=a['active_branch_substeps'] for a,b in zip(frames,frames[1:])):
        raise ValueError('Retained material sequence clock is not increasing')
    return {'kind':'observed_scene_active_branch_clock_v1','status':'recorded_integration_count_verified',
        'events':len(events),'step_calls_including_temporary_branches':step_count,'state_restore_calls':restore_count,
        'scene_reset_calls':reset_count,'substep_dt_s':.0002,'records':frames,
        'all_elapsed_times_equal_original_labels':all(abs(r['elapsed_minus_label_s'])<=1e-8 for r in frames),
        'main_frame_unchanged':60,'original_main_label_s':1.920,'task_satisfaction':'not_evaluated',
        'candidate_pool_released':False,'scope':'Observed integration/rollback only; no task protocol change or action-selection input'}


def diagnose_tool_schedule(output,source_log,physical_reference,nominal_center,parameters):
    """Collect every discrepancy in saved traces; never certify/repair a run.

    Unlike a fail-fast gate, this exposes downstream numeric mismatches even
    when an earlier clock check fails. This is an offline development report;
    no data from it enters prediction, action selection or task thresholds.
    """
    output=Path(output);nominal_center=numeric(nominal_center,(3,),'nominal center')
    parameters=numeric(parameters,(3,),'candidate parameters')
    with (output/'scenes/0000/log.pkl').open('rb') as stream:actual=pickle.load(stream)
    with Path(source_log).open('rb') as stream:nominal=pickle.load(stream)
    if len(actual)!=63 or len(nominal)!=63:raise ValueError('Expected63 saved actual and nominal frames')
    records=[]
    for frame,(row,base) in enumerate(zip(actual,nominal)):
        if row['step']!=frame*16 or row['frame_idx']!=frame or abs(float(row['t'])-frame*.032)>1e-8:
            raise ValueError('Saved execution frame/label time changed')
        record={'frame':frame,'nominal_label_s':float(row['t'])}
        with np.load(output/'actual_tools'/('frame_%03d.npz'%frame),allow_pickle=False) as packet, np.load(
                Path(physical_reference)/('frame_%03d.npz'%frame),allow_pickle=False) as reference:
            if int(reference['step'])!=frame*16:raise ValueError('Nominal actual tool reference step changed')
            record.update(inspect_tool_clock(packet,frame))
            for key in ('pos','pos_left','pos_right'):
                point=numeric(row['ee'][key],(3,),'actual summary '+key)
                old=numeric(base['ee'][key],(3,),'nominal summary '+key)
                expected=transform_tools(old[None,None,:],nominal_center,[parameters],.2)[0,0,0,:3]
                record['summary_'+key+'_transform_error_mm']=float(np.max(np.abs(point-expected))*200)
            for i,side in ((1,'pos_left'),(2,'pos_right')):
                prefix='finger_%d_'%i
                pos=numeric(packet['tool_%03d_pos'%i],(3,),'current actual finger')
                old=numeric(reference['tool_%03d_pos'%i],(3,),'nominal actual finger')
                expected=transform_tools(old[None,None,:],nominal_center,[parameters],.2)[0,0,0,:3]
                latest=numeric(packet['latest_tool_%03d_pos'%i],(3,),'latest actual finger')
                previous=numeric(packet['previous_tool_%03d_pos'%i],(3,),'previous actual finger')
                record[prefix+'physical_transform_error_mm']=float(np.max(np.abs(pos-expected))*200)
                record[prefix+'latest_previous_error_mm']=float(np.max(np.abs(latest-previous))*200)
                record[prefix+'latest_previous_exact']=bool(np.array_equal(latest,previous))
                record[prefix+'summary_latest_error_mm']=float(np.max(np.abs(latest-row['ee'][side]))*200)
                record[prefix+'current_latest_error_mm']=float(np.max(np.abs(pos-latest))*200)
                q=numeric(packet['tool_%03d_quat'%i],(4,),'actual quaternion')
                expected_q=np.asarray(compose_yaw(base['ee']['quat'],parameters[2]))
                record[prefix+'quaternion_error']=float(min(np.max(np.abs(q-expected_q)),np.max(np.abs(q+expected_q))))
                lq=numeric(packet['latest_tool_%03d_quat'%i],(4,),'latest quaternion')
                pq=numeric(packet['previous_tool_%03d_quat'%i],(4,),'previous quaternion')
                record[prefix+'latest_previous_quaternion_exact']=bool(np.array_equal(lq,pq))
            record['summary_opening_error_mm']=abs(float(row['ee']['open'])-float(base['ee']['open']))*200
            if not np.isfinite(record['summary_opening_error_mm']):raise ValueError('Nonfinite opening')
        records.append(record)
    metrics={}
    for key in records[0]:
        if key.endswith(('_error_mm','quaternion_error')):
            worst=max(records,key=lambda r:r[key])
            limit=1e-6 if key.endswith('quaternion_error') else .001
            metrics[key]={'max':worst[key],'frame':worst['frame'],
                          'comparison_limit':limit,'exceeds_limit':worst[key]>limit}
    # Diagnostic current/latest separation and reconstruction error are also
    # recorded; the acceptance mismatch concerns requested vs executed tools.
    transform_keys=[key for key in metrics if 'transform_error' in key or key=='summary_opening_error_mm' or key.endswith('quaternion_error')]
    violated=any(metrics[key]['exceeds_limit'] for key in transform_keys)
    return {'kind':'offline_saved_tool_schedule_diagnosis_v1',
            'status':'violated' if violated else 'unknown',
            'tool_transform_status':'violated' if violated else 'within_original_numeric_limits',
            'metrics':metrics,'clock_status':'unknown_elapsed_physical_time',
            'frames_not_matching_no_reset_counter':[r['frame'] for r in records if not r['matches_no_reset_counter']],
            'all_local_indices_consistent':True,
            'latest_previous_exact_all_frames':all(r['finger_%d_latest_previous_exact'%i] and
                r['finger_%d_latest_previous_quaternion_exact'%i] for r in records for i in (1,2)),
            'records':records,'legacy_model_tool_input_alignment':'not_certified',
            'candidate_pool_released':False,'task_satisfaction':'not_evaluated',
            'position_limit_mm_unchanged':.001,'main_frame_unchanged':60,
            'main_nominal_label_s_unchanged':1.920,'new_simulations':0,
            'comparison_scope':'matched saved frame labels; equal cumulative physical time is not certified'}


def compare_prepared_candidates(reference,fresh,bounds):
    """Actions are exact; scalar costs allow documented arithmetic rounding.

    This comparison only gates an interface probe with fixed candidate IDs.
    It never selects actions or changes scientific thresholds or reset limits.
    """
    bounds=numeric(bounds,(3,),'action bounds')
    if (bounds<=0).any() or len(reference)!=len(fresh):raise ValueError('Candidate pool bounds/count changed')
    differences=[]
    for a,b in zip(reference,fresh):
        if {k:v for k,v in a.items() if k!='action_cost'}!={k:v for k,v in b.items() if k!='action_cost'}:
            raise ValueError('Frozen candidate action/metadata changed')
        action=numeric(a['parameters_mm_mm_deg'],(3,),'candidate action')
        scalar=math.sqrt(math.fsum((float(x)/float(y))**2 for x,y in zip(action,bounds)))
        limit=2*math.ulp(scalar)
        for row in (a,b):
            cost=float(row['action_cost'])
            if not math.isfinite(cost) or abs(cost-scalar)>limit:raise ValueError('Candidate cost is not the frozen normalized L2 cost')
        if a['action_cost']!=b['action_cost']:
            differences.append({'candidate_id':a['candidate_id'],'source_cost':a['action_cost'],
                'fresh_cost':b['action_cost'],'independent_scalar_cost':scalar,'arithmetic_limit_ULP':2})
    return {'action_and_metadata_exactly_equal':True,'cost_rounding_differences':differences,
            'diagnostic_only_fixed_IDs_cost_not_used_for_selection':True}


def audit_candidate(output,source_log,source_state,material_reference,initial_reference,
                    nominal_center,parameters,scale=.2,position_limit_mm=.001,quaternion_limit=1e-6,
                    *,physical_reference=None):
    output=Path(output)
    if (scale!=.2 or position_limit_mm!=.001 or quaternion_limit!=1e-6
            or np.asarray(parameters).shape!=(3,) or not np.isfinite(parameters).all()):
        raise ValueError('Candidate diagnostic units/limits or parameters changed')
    nominal_center=numeric(nominal_center,(3,),'nominal center')
    exact_numeric_packet(source_state,output/'reset_returned.npz')
    exact_numeric_packet(material_reference,output/'static_material_info.npz')
    with np.load(initial_reference,allow_pickle=False) as packet:
        reference=packet['initial_particles'].astype(float)
    if reference.ndim!=2 or reference.shape[1]!=3 or not len(reference) or not np.isfinite(reference).all():
        raise ValueError('Invalid initial material reference')
    with (output/'scenes/0000/log.pkl').open('rb') as inp:actual=pickle.load(inp)
    with Path(source_log).open('rb') as inp:nominal=pickle.load(inp)
    if len(actual)!=63 or len(nominal)!=63:raise ValueError('Expected unchanged63-frame logs through auxiliary62')
    times=np.asarray([r['t'] for r in actual])
    if not np.allclose(times,np.arange(63)*.032,rtol=0,atol=1e-8):
        raise ValueError('Candidate execution dense physical times changed')
    if any(r['step']!=i*16 or r['frame_idx']!=i for i,r in enumerate(actual)):
        raise ValueError('Candidate frame/step mapping changed')
    positions=np.asarray(actual[0]['obj']['particles']['pos'],dtype=float)
    if positions.shape!=reference.shape or not np.isfinite(positions).all():
        raise ValueError('Candidate initial material count/nonfinite mismatch')
    initial_error=float(np.max(np.abs(positions-reference))*200)
    if initial_error>position_limit_mm:raise ValueError('Candidate logged0 material exceeds unchanged reset tolerance')
    if int(np.sum(actual[0]['obj']['particles']['colliding']))!=0:
        raise ValueError('Candidate starts in tool collision')
    maximum_pos=maximum_quat=maximum_open=summary_solver_max=0.
    trajectory=[]
    # Check every raw logging time, including the two auxiliary tail frames.
    for index,(row,base) in enumerate(zip(actual,nominal)):
        numeric(row['obj']['particles']['pos'],reference.shape,'raw material frame '+str(index))
        ee,old=row['ee'],base['ee']
        if ee.get('is_gripper') is not True or old.get('is_gripper') is not True:
            raise ValueError('Candidate requires the same two-finger gripper')
        record={'frame':index,'t':float(times[index]),'positions_max_error_mm':0.}
        for key in ('pos','pos_left','pos_right'):
            point=np.asarray(ee[key],dtype=float)
            nominal_point=np.asarray(old[key],dtype=float)
            if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid tool position: '+key)
            expected=transform_tools(nominal_point[None,None,:],nominal_center,[parameters],scale)[0,0,0,:3]
            error=float(np.max(np.abs(point-expected))*200)
            record['positions_max_error_mm']=max(record['positions_max_error_mm'],error)
            maximum_pos=max(maximum_pos,error)
        expected_quat=np.asarray(compose_yaw(old['quat'],parameters[2]))
        q=np.asarray(ee['quat'],dtype=float)
        if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1,rtol=0,atol=1e-6):
            raise ValueError('Invalid executed tool quaternion')
        error=float(min(np.max(np.abs(q-expected_quat)),np.max(np.abs(q+expected_quat))))
        maximum_quat=max(maximum_quat,error);record['quat_sign_invariant_max_abs_error']=error
        gap_error=abs(float(ee['open'])-float(old['open']))*200
        if not np.isfinite(gap_error):raise ValueError('Nonfinite tool opening')
        maximum_open=max(maximum_open,gap_error);record['opening_error_mm']=gap_error
        trajectory.append(record)
        # Read actual solver entity poses as well as the reconstructed ee log.
        # A rigid-gripper summary alone cannot certify both tool entities.
        with np.load(output/'actual_tools'/('frame_%03d.npz'%index),allow_pickle=False) as packet,np.load(source_state,allow_pickle=False) as source:
            if int(packet['step'])!=index*16:raise ValueError('Actual solver tool frame/step mismatch')
            if physical_reference is not None:
                clock=inspect_tool_clock(packet,index)
                if not clock['matches_no_reset_counter']:
                    raise ValueError('Solver clock reset observed; elapsed physical time requires state-restore event provenance')
            for i,key in ((1,'pos_left'),(2,'pos_right')):
                point=numeric(packet['tool_%03d_pos'%i],(3,),'actual finger position')
                error=float(np.max(np.abs(point-numeric(ee[key],(3,),'logged finger position')))*200)
                summary_solver_max=max(summary_solver_max,error)
                if physical_reference is None:
                    maximum_pos=max(maximum_pos,error)
                else:
                    with np.load(Path(physical_reference)/('frame_%03d.npz'%index),allow_pickle=False) as nominal_packet:
                        if int(nominal_packet['step'])!=index*16:raise ValueError('Nominal actual tool reference step changed')
                        physical=numeric(nominal_packet['tool_%03d_pos'%i],(3,),'nominal actual solver position')
                        physical_expected=transform_tools(physical[None,None,:],nominal_center,[parameters],scale)[0,0,0,:3]
                        physical_error=float(np.max(np.abs(point-physical_expected))*200)
                        maximum_pos=max(maximum_pos,physical_error)
                        record['actual_solver_to_nominal_solver_max_error_mm']=max(record.get('actual_solver_to_nominal_solver_max_error_mm',0.),physical_error)
                    for field in ('pos','quat'):
                        latest=numeric(packet['latest_tool_%03d_%s'%(i,field)],(3,) if field=='pos' else (4,),'actual latest tool '+field)
                        previous=numeric(packet['previous_tool_%03d_%s'%(i,field)],latest.shape,'actual previous tool '+field)
                        if not np.array_equal(latest,previous):raise ValueError('Latest tool field is not the exact last integrated solver index')
                tq=numeric(packet['tool_%03d_quat'%i],(4,),'actual finger quaternion')
                maximum_quat=max(maximum_quat,float(min(np.max(np.abs(tq-expected_quat)),np.max(np.abs(tq+expected_quat)))))
            for key in ('pos','quat','vel','ang'):
                if not np.array_equal(packet['tool_000_'+key],source['tool_000_'+key]):
                    raise ValueError('Candidate changed actual ground trajectory')
    # Inspect the actual solver fields before first integration, independently
    # of the summarized ee log. Ground is intentionally excluded from rotation.
    mapping=read_json(output/'tool_entity_mapping.json')
    if mapping['names']!=['ground','finger_left','finger_right'] or mapping['gripper_indices']!=[1,2]:
        raise ValueError('Actual tool entity order differs from frozen ground/two-finger contract')
    with np.load(output/'after_first_action.npz',allow_pickle=False) as packet,np.load(source_state,allow_pickle=False) as source:
        for key in ('mpm_pos','mpm_vel','mpm_C','mpm_F','mpm_active'):
            if not np.array_equal(packet[key],source[key]):raise ValueError('First tool action changed material before solver step')
        for key in ('pos','quat','vel','ang'):
            if not np.array_equal(packet['tool_000_'+key],source['tool_000_'+key]):
                raise ValueError('Candidate action changed ground state')
    with np.load(output/'first_action_targets.npz',allow_pickle=False) as packet:
        for i,key in ((1,'pos_left'),(2,'pos_right')):
            expected=transform_tools(np.asarray(nominal[0]['ee'][key])[None,None,:],nominal_center,[parameters],scale)[0,0,0,:3]
            point=numeric(packet['tool_%03d_pos'%i],(3,),'first action target position')
            err=float(np.max(np.abs(point-expected))*200)
            maximum_pos=max(maximum_pos,err)
            if err>position_limit_mm:raise ValueError('First action tool target position differs')
            q=numeric(packet['tool_%03d_quat'%i],(4,),'first action target quaternion')
            for field in ('vel','ang'):
                if np.any(numeric(packet['tool_%03d_%s'%(i,field)],(3,),'first action '+field)!=0):
                    raise ValueError('Initial set action should have zero tool velocities')
            expected_q=np.asarray(compose_yaw(nominal[0]['ee']['quat'],parameters[2]))
            qerr=float(min(np.max(np.abs(q-expected_q)),np.max(np.abs(q+expected_q))))
            maximum_quat=max(maximum_quat,qerr)
    if maximum_pos>position_limit_mm or maximum_open>position_limit_mm or maximum_quat>quaternion_limit:
        raise ValueError('Candidate full tool schedule disagrees with model action transform: position_mm=%r opening_mm=%r quat=%r'%
                         (maximum_pos,maximum_open,maximum_quat))
    return {'kind':'two_candidate_interface_diagnostic_not_full_pool_certification','status':'passed',
        'parameters_mm_mm_deg':list(map(float,parameters)),'same_complete_pre_action_state':True,
        'same_actual_static_material_map':True,'logged0_max_abs_material_error_mm':initial_error,
        'tool_position_max_abs_error_mm':maximum_pos,'tool_opening_max_error_mm':maximum_open,
        'tool_quaternion_sign_invariant_max_abs_error':maximum_quat,
        'solver_vs_reconstructed_summary_max_error_mm':summary_solver_max,
        'physical_reference_basis':'same_index_actual_solver_development_nominal' if physical_reference is not None else 'legacy_reconstructed_ee_summary',
        'legacy_model_tool_input_alignment':'not_certified' if physical_reference is not None else 'passed',
        'diagnostic_position_limit_mm_unchanged':position_limit_mm,'diagnostic_quaternion_limit':quaternion_limit,
        'frames_checked':63,'primary_frame':60,'primary_s':1.920,'auxiliary_frame':62,'auxiliary_s':1.984,
        'model_tool_pointcloud_basis':'legacy_model_input_still_requires_alignment' if physical_reference is not None else 'same frozen mesh/opening; worldXY rigid transform of archived action schedule',
        'initial_observation_source':'new_shared_nominal_frame0_Renderer_AE_already_saved',
        'source_log_sha256':sha256(source_log),'actual_log_sha256':sha256(output/'scenes/0000/log.pkl'),
        'trajectory_errors':trajectory,'candidate_pool_released':False,
        'scope':'Only explicitly checked candidate IDs; no material task satisfaction or prediction claim'}
