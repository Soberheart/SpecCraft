import copy
import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from level4.artifacts import read_json,write_json
from level4.archive import verify_zip
from level4.candidate_interface import audit_candidate,inspect_tool_clock,diagnose_tool_schedule
from tests.test_level4_shared_reset import make_packet


def fixture(root):
    # Independently specified 90deg XY rotation and 1mm/-2mm translation.
    # Expected positions/quats below do not call either transform helper.
    packet=make_packet()
    for i in (1,2):
        for key in ('pos','quat','vel','ang'):packet['tool_%03d_%s'%(i,key)]=packet['tool_000_'+key].copy()
    source=root/'source.npz';np.savez(source,**packet)
    material=root/'static.npz';np.savez(material,mat_idx=np.array([0,0],dtype=np.int32),mu=np.array([357.14,357.14],dtype=np.float32))
    nominal=[];actual=[]
    q=np.array([2**-.5,0,0,2**-.5])
    for index in range(63):
        gap=.5-.002*index
        row={'step':index*16,'frame_idx':index,'t':index*.032,
            'obj':{'particles':{'pos':packet['mpm_pos'].copy(),'colliding':np.zeros(2,dtype=int)}},
            'ee':{'is_gripper':True,'pos':np.array([0.,0.,0.]),'pos_left':np.array([-gap/2,0.,0.]),
                  'pos_right':np.array([gap/2,0.,0.]),'quat':np.array([1.,0.,0.,0.]),'open':gap}}
        nominal.append(row)
        changed=copy.deepcopy(row)
        changed['ee'].update(pos=np.array([.005,-.01,0]),pos_left=np.array([.005,-.01-gap/2,0.]),
            pos_right=np.array([.005,-.01+gap/2,0.]),quat=q.copy())
        actual.append(changed)
    reference=root/'initial.npz';np.savez(reference,initial_particles=packet['mpm_pos'])
    log=root/'nominal.pkl'
    with log.open('wb') as out:pickle.dump(nominal,out)
    target=root/'candidate';(target/'scenes/0000').mkdir(parents=True)
    np.savez(target/'reset_returned.npz',**packet);np.savez(target/'after_first_action.npz',**packet)
    np.savez(target/'static_material_info.npz',mat_idx=np.array([0,0],dtype=np.int32),mu=np.array([357.14,357.14],dtype=np.float32))
    targets={'tool_%03d_%s'%(i,key):value for i,side in ((1,'pos_left'),(2,'pos_right'))
             for key,value in (('pos',actual[0]['ee'][side]),('quat',q),('vel',np.zeros(3)),('ang',np.zeros(3)))}
    np.savez(target/'first_action_targets.npz',**targets)
    (target/'actual_tools').mkdir()
    for index,row in enumerate(actual):
        trace={'step':np.asarray(index*16)}
        for key in ('pos','quat','vel','ang'):trace['tool_000_'+key]=packet['tool_000_'+key]
        for i,side in ((1,'pos_left'),(2,'pos_right')):
            trace['tool_%03d_pos'%i]=row['ee'][side];trace['tool_%03d_quat'%i]=q
        np.savez(target/'actual_tools'/('frame_%03d.npz'%index),**trace)
    write_json(target/'tool_entity_mapping.json',{'names':['ground','finger_left','finger_right'],'gripper_indices':[1,2]})
    with (target/'scenes/0000/log.pkl').open('wb') as out:pickle.dump(actual,out)
    args=[target,log,source,material,reference,[0,0,0],[1,-2,90]]
    return args,actual


class CandidateInterfaceTests(unittest.TestCase):
    def test_clock_observer_rolls_back_probe_steps_and_retains_extra_step(self):
        from scripts.probe_level4_candidate_interface import attach_clock_capture
        class State:pass
        events=[]
        sim=SimpleNamespace(cur_substep_global=0,n_substeps=10)
        def original_step():sim.cur_substep_global+=10;events.append('step')
        def original_set(state):sim.cur_substep_global=0;events.append('restore')
        scene=SimpleNamespace(sim=sim,step=original_step,set_state=original_set,
            reset=lambda:events.append('reset'),get_state=lambda:State())
        with tempfile.TemporaryDirectory() as d:
            clock=attach_clock_capture(scene,Path(d));scene.reset();scene.step()
            saved=scene.get_state()
            for _ in range(160):scene.step()  # existing temporary probe
            scene.set_state(saved);scene.step()  # existing retained extra step
            self.assertEqual(clock['active_substeps'],20)
            self.assertEqual(sim.cur_substep_global,10)
            self.assertEqual(events.count('step'),162)  # observer added none
            self.assertEqual(events.count('restore'),1)
            unknown=State();scene.set_state(unknown);scene.step()
            self.assertIsNone(clock['active_substeps'])
            self.assertEqual(events.count('step'),163)
            self.assertEqual(len((Path(d)/'clock_events.jsonl').read_text(encoding='utf-8').splitlines()),clock['event_count'])

    def test_event_replay_matches_frame_prefixes_and_rejects_tampering(self):
        from scripts.probe_level4_candidate_interface import attach_clock_capture
        from level4.candidate_interface import reduce_clock_events
        class State:pass
        sim=SimpleNamespace(cur_substep_global=0,n_substeps=10)
        def step():sim.cur_substep_global+=10
        def restore(state):sim.cur_substep_global=0
        scene=SimpleNamespace(sim=sim,step=step,set_state=restore,reset=lambda:None,get_state=lambda:State())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);clock=attach_clock_capture(scene,root);scene.reset();tools=root/'tools';tools.mkdir()
            def save(frame):
                np.savez(tools/('frame_%03d.npz'%frame),step=np.asarray(frame*16),
                    sample_global_substep=np.asarray(sim.cur_substep_global),current_local_substep=np.asarray(sim.cur_substep_global%50),
                    previous_local_substep=np.asarray((sim.cur_substep_global-1)%50),substep_dt=np.asarray(.0002),
                    active_branch_clock_known=np.asarray(True),active_branch_substeps=np.asarray(clock['active_substeps']),
                    clock_event_count=np.asarray(clock['event_count']))
            scene.step();save(0);state=scene.get_state()
            for _ in range(160):scene.step()
            scene.set_state(state);scene.step();save(1)
            result=reduce_clock_events(root/'clock_events.jsonl',tools,frame_count=2)
            self.assertAlmostEqual(result['records'][1]['active_branch_elapsed_s'],.002)
            self.assertFalse(result['all_elapsed_times_equal_original_labels'])
            self.assertFalse(result['candidate_pool_released'])
            original=(root/'clock_events.jsonl').read_text(encoding='utf-8')
            import json
            events=[json.loads(s) for s in original.splitlines()];events[3]['active_substeps']+=10
            (root/'clock_events.jsonl').write_text('\n'.join(json.dumps(s) for s in events), encoding='utf-8')
            with self.assertRaises(ValueError):reduce_clock_events(root/'clock_events.jsonl',tools,frame_count=2)

    def test_clock_probe_guard_retains_failure_without_simulation(self):
        from scripts.probe_level4_clock_diagnostic import collect
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);request=root/'request.json';write_json(request,{'kind':'invalid'})
            with patch('sys.stdout',io.StringIO()):self.assertEqual(collect(request,root/'data'),1)
            self.assertEqual(read_json(root/'data/stage_exit.json')['execution_subprocess_exits_recorded'],0)
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_resettable_clock_consistency_is_separate_from_elapsed_time(self):
        packet={'step':np.asarray(400),'sample_global_substep':np.asarray(10),
                'current_local_substep':np.asarray(10),'previous_local_substep':np.asarray(9),
                'substep_dt':np.asarray(.0002)}
        clock=inspect_tool_clock(packet,25)
        self.assertTrue(clock['local_indices_consistent'])
        self.assertFalse(clock['matches_no_reset_counter'])
        self.assertFalse(clock['elapsed_physical_time_certified'])
        for key,value in (('current_local_substep',11),('previous_local_substep',10),
                          ('sample_global_substep',10.5),('sample_global_substep',-1),('step',399)):
            bad=dict(packet);bad[key]=np.asarray(value)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):inspect_tool_clock(bad,25)

    def test_offline_diagnosis_keeps_numeric_violation_after_clock_reset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,actual=fixture(root);reference=root/'physical';reference.mkdir()
            with args[1].open('rb') as stream:nominal=pickle.load(stream)
            for frame,row in enumerate(actual):
                path=args[0]/'actual_tools'/('frame_%03d.npz'%frame)
                with np.load(path) as packet:values={k:packet[k] for k in packet.files}
                g=10 if frame>=25 else (frame*16+1)*10
                values.update(sample_global_substep=np.asarray(g),current_local_substep=np.asarray(g%50),
                              previous_local_substep=np.asarray((g-1)%50),substep_dt=np.asarray(.0002))
                baseline={'step':np.asarray(frame*16)}
                for i,key in ((1,'pos_left'),(2,'pos_right')):
                    baseline['tool_%03d_pos'%i]=nominal[frame]['ee'][key]
                    for field in ('pos','quat'):
                        values['previous_tool_%03d_%s'%(i,field)]=values['tool_%03d_%s'%(i,field)].copy()
                        values['latest_tool_%03d_%s'%(i,field)]=values['tool_%03d_%s'%(i,field)].copy()
                if frame==60:values['tool_001_pos']=values['tool_001_pos']+np.array([.001,0,0])
                np.savez(path,**values);np.savez(reference/path.name,**baseline)
            with self.assertRaisesRegex(ValueError,'state-restore event provenance'):
                audit_candidate(*args,physical_reference=reference)
            report=diagnose_tool_schedule(args[0],args[1],reference,args[5],args[6])
            self.assertEqual(report['tool_transform_status'],'violated')
            self.assertEqual(report['clock_status'],'unknown_elapsed_physical_time')
            self.assertAlmostEqual(report['metrics']['finger_1_physical_transform_error_mm']['max'],.2,places=5)
            self.assertEqual(report['metrics']['finger_1_physical_transform_error_mm']['frame'],60)
            self.assertTrue(report['latest_previous_exact_all_frames'])
            self.assertFalse(report['candidate_pool_released'])

    def test_candidate_cost_rounding_without_action_change_and_bad_cost_rejected(self):
        from level4.candidate_interface import compare_prepared_candidates
        action={'candidate_id':'candidate_001','parameters_mm_mm_deg':[1.,0.,0.],'action_cost':.025,'legal':True}
        fresh=dict(action,action_cost=float(np.nextafter(.025,np.inf)))
        self.assertEqual(len(compare_prepared_candidates([action],[fresh],[40,40,10])['cost_rounding_differences']),1)
        for changed in (dict(fresh,parameters_mm_mm_deg=[1.000001,0,0]),dict(fresh,action_cost=.025001),dict(fresh,action_cost=float('nan'))):
            with self.assertRaises(ValueError):compare_prepared_candidates([action],[changed],[40,40,10])

    def test_analytic_rotation_translation_opening_and_times(self):
        with tempfile.TemporaryDirectory() as d:
            args,actual=fixture(Path(d));report=audit_candidate(*args)
            self.assertEqual(report['status'],'passed');self.assertEqual(report['frames_checked'],63)
            self.assertLess(report['tool_position_max_abs_error_mm'],.001)
            self.assertFalse(report['candidate_pool_released'])

    def mutate_log(self,args,rows):
        with (args[0]/'scenes/0000/log.pkl').open('wb') as out:pickle.dump(rows,out)

    def test_late_tool_schedule_drift_and_missing_frame_rejected(self):
        for mode in ('pose','time','missing'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                args,rows=fixture(Path(d))
                if mode=='pose':rows[60]['ee']['pos_left'][0]+=.01
                elif mode=='time':rows[60]['t']=1.984
                else:rows.pop()
                self.mutate_log(args,rows)
                with self.assertRaises(ValueError):audit_candidate(*args)

    def test_quaternion_sign_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            args,rows=fixture(Path(d))
            for row in rows:row['ee']['quat']*=-1
            self.mutate_log(args,rows);self.assertEqual(audit_candidate(*args)['status'],'passed')

    def test_static_material_and_hidden_history_tampering_rejected(self):
        for path,key in (('static_material_info.npz','mu'),('reset_returned.npz','mpm_F'),('after_first_action.npz','mpm_C')):
            with self.subTest(path=path),tempfile.TemporaryDirectory() as d:
                args,rows=fixture(Path(d));target=args[0]/path
                with np.load(target,allow_pickle=False) as p:values={k:p[k] for k in p.files}
                values[key].flat[0]+=1;np.savez(target,**values)
                with self.assertRaises(ValueError):audit_candidate(*args)

    def test_stale_nominal_targets_and_ground_change_rejected(self):
        for mode in ('targets','ground'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                args,rows=fixture(Path(d));target=args[0]/('first_action_targets.npz' if mode=='targets' else 'after_first_action.npz')
                with np.load(target,allow_pickle=False) as p:values={k:p[k] for k in p.files}
                values['tool_001_pos' if mode=='targets' else 'tool_000_pos'][0]+=1
                np.savez(target,**values)
                with self.assertRaises(ValueError):audit_candidate(*args)

    def test_start_collision_and_wrong_mapping_rejected(self):
        for mode in ('collision','mapping'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                args,rows=fixture(Path(d))
                if mode=='collision':rows[0]['obj']['particles']['colliding'][0]=1;self.mutate_log(args,rows)
                else:(args[0]/'tool_entity_mapping.json').write_text('{"names":["finger_left","ground","finger_right"],"gripper_indices":[0,2]}', encoding='utf-8')
                with self.assertRaises(ValueError):audit_candidate(*args)

    def test_guard_failure_retains_stage_and_verifiable_bundle(self):
        from scripts.probe_level4_candidate_interface import collect
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);request=root/'request.json';write_json(request,{'kind':'invalid'})
            with patch('sys.stdout',io.StringIO()):self.assertEqual(collect(request,root/'data'),1)
            self.assertEqual(read_json(root/'data/stage_exit.json')['exit_code'],1)
            self.assertFalse(read_json(root/'data/stage_exit.json')['model_prediction_run'])
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_same_solver_time_reference_does_not_certify_legacy_summary_input(self):
        for changed in ('none','bad_latest','bad_time','wrong_reference'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as d:
                root=Path(d);args,rows=fixture(root);ref=root/'nominal_physical';ref.mkdir()
                with args[1].open('rb') as inp:nominal=pickle.load(inp)
                for index,row in enumerate(rows):
                    path=args[0]/'actual_tools'/('frame_%03d.npz'%index)
                    with np.load(path,allow_pickle=False) as p:values={k:p[k] for k in p.files}
                    # Independent 90deg rotation maps (0,-.00002,0) to
                    # (.00002,0,0); current and historical summary differ .004mm.
                    reference={'step':np.asarray(index*16)}
                    for i,key in ((1,'pos_left'),(2,'pos_right')):
                        reference['tool_%03d_pos'%i]=nominal[index]['ee'][key]+np.array([0,-.00002,0])
                        values['previous_tool_%03d_pos'%i]=row['ee'][key]
                        values['latest_tool_%03d_pos'%i]=row['ee'][key]
                        values['previous_tool_%03d_quat'%i]=row['ee']['quat']
                        values['latest_tool_%03d_quat'%i]=row['ee']['quat']
                        values['tool_%03d_pos'%i]=row['ee'][key]+np.array([.00002,0,0])
                    global_index=(index*16+1)*10
                    values.update(sample_global_substep=np.asarray(global_index),current_local_substep=np.asarray(global_index%50),
                        previous_local_substep=np.asarray((global_index-1)%50),substep_dt=np.asarray(.0002))
                    if index==60:
                        if changed=='bad_latest':values['latest_tool_001_pos']=values['latest_tool_001_pos']+1
                        elif changed=='bad_time':values['sample_global_substep']=np.asarray(global_index-1)
                        elif changed=='wrong_reference':reference['tool_001_pos']=reference['tool_001_pos']+1
                    np.savez(path,**values);np.savez(ref/path.name,**reference)
                if changed!='none':
                    with self.assertRaises(ValueError):audit_candidate(*args,physical_reference=ref)
                else:
                    with self.assertRaises(ValueError):audit_candidate(*args)
                    report=audit_candidate(*args,physical_reference=ref)
                    self.assertEqual(report['legacy_model_tool_input_alignment'],'not_certified')
                    self.assertFalse(report['candidate_pool_released'])
                    self.assertGreater(report['solver_vs_reconstructed_summary_max_error_mm'],.001)

    def test_pending_probe_scope_failure_records_zero_execution_and_hash_bundle(self):
        from scripts.probe_level4_pending_candidate import collect
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);request=root/'request.json';write_json(request,{'kind':'invalid'})
            with patch('sys.stdout',io.StringIO()):self.assertEqual(collect(request,root/'data'),1)
            record=read_json(root/'data/stage_exit.json')
            self.assertEqual(record['execution_subprocess_exits_recorded'],0)
            self.assertFalse(record['candidate_pool_released'])
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_invalid_targets_reference_and_actual_tool_drift_rejected(self):
        for mode in ('target_nan','target_shape','target_velocity','reference_nan','actual_finger','actual_ground'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                args,rows=fixture(Path(d))
                if mode=='reference_nan':path=args[4];key='initial_particles'
                elif mode.startswith('target'):path=args[0]/'first_action_targets.npz';key='tool_001_vel' if mode=='target_velocity' else 'tool_001_pos'
                else:path=args[0]/'actual_tools/frame_060.npz';key='tool_001_pos' if mode=='actual_finger' else 'tool_000_pos'
                with np.load(path,allow_pickle=False) as p:values={k:p[k] for k in p.files}
                if mode=='target_shape':values[key]=np.zeros((1,3))
                else:values[key].flat[0]=np.nan if mode.endswith('nan') else values[key].flat[0]+1
                np.savez(path,**values)
                with self.assertRaises(ValueError):audit_candidate(*args)

    def test_capture_observes_targets_and_actual_states_without_extra_solver_steps(self):
        from scripts.probe_level4_candidate_interface import attach_action_capture
        events=[];tools=[]
        for i in range(3):
            state=SimpleNamespace(pos=np.zeros(3),quat=np.array([1.,0,0,0]),vel=np.zeros(3),ang=np.zeros(3))
            tool=SimpleNamespace(_tgt={k:getattr(state,k).copy() for k in ('pos','quat','vel','ang')},get_state=lambda s=state:s)
            tools.append(tool)
        def original_step(goals):
            events.append('planner');tools[1]._tgt['pos'][0]=4;return False
        workspace=SimpleNamespace(planner=SimpleNamespace(step=original_step),log=lambda *a,**k:events.append('log'))
        recorder=SimpleNamespace(save=lambda w,n:events.append(n))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);attach_action_capture(workspace,recorder,tools,[1,2],root)
            self.assertFalse(workspace.planner.step([]));workspace.log(0);workspace.planner.step([]);workspace.log(16)
            self.assertEqual(events,['planner','after_first_action','log','planner','log'])
            with np.load(root/'first_action_targets.npz',allow_pickle=False) as p:self.assertEqual(p['tool_001_pos'][0],4)
            self.assertEqual(len(list((root/'actual_tools').glob('*.npz'))),2)

    def test_capture_preserves_both_solver_indices_and_latest_without_advancing(self):
        from scripts.probe_level4_candidate_interface import attach_action_capture
        events=[];tools=[]
        sim=SimpleNamespace(cur_substep_global=170,cur_substep_local=20,max_substeps_local=50,substep_dt=.0002)
        for i in range(3):
            state=SimpleNamespace(pos=np.array([1.,2,3]),quat=np.array([1.,0,0,0]),vel=np.zeros(3),ang=np.zeros(3))
            latest=SimpleNamespace(to_numpy=lambda:np.array([[4.,5,6]]))
            def get_state(f=None,s=state):events.append(('get',f));return s
            tools.append(SimpleNamespace(_tgt={k:getattr(state,k).copy() for k in ('pos','quat','vel','ang')},get_state=get_state,
                latest_pos=latest,latest_quat=SimpleNamespace(to_numpy=lambda:np.array([[1.,0,0,0]]))))
        workspace=SimpleNamespace(scene=SimpleNamespace(sim=sim),planner=SimpleNamespace(step=lambda g:False),log=lambda *a:None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);attach_action_capture(workspace,SimpleNamespace(save=lambda *a:None),tools,[1,2],root)
            workspace.log(16)
            self.assertEqual(events,[('get',None),('get',19)]*3)
            self.assertEqual(sim.cur_substep_global,170)
            with np.load(root/'actual_tools/frame_001.npz',allow_pickle=False) as p:
                self.assertEqual(int(p['sample_global_substep']),170)
                self.assertEqual(int(p['previous_local_substep']),19)
                np.testing.assert_array_equal(p['latest_tool_001_pos'],[4,5,6])


if __name__=='__main__':unittest.main()
