"""CPU orchestration regression: synthetic captures, no simulator or scheduler.

Exercise real exclusive artifact writers, reset equality checks, clock replay,
tool comparison and ZIP manifest validation across a complete two-case run.
Only environment/runtime setup, YAML loading and child capture are mocked.
"""
import io
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from level4.artifacts import read_json, sha256, write_json
from level4.archive import verify_zip
from scripts import probe_level4_transaction as runner


def make_fixture(root):
    pool = root/'pool'; pool.mkdir()
    np.savez(pool/'shared_source_state.npz', material=np.array([1.,2.]))
    np.savez(pool/'static_material_info.npz', mat_idx=np.array([0,0], dtype=np.int32))
    np.savez(pool/'planner_input.npz', synthetic=np.array([True]))
    (pool/'source_scene.yaml').write_text('synthetic CPU fixture only\n', encoding='utf-8')
    candidates=[]
    for index in range(16):
        identifier='candidate_%03d'%index
        (pool/(identifier+'.yaml')).write_text('synthetic CPU fixture only\n', encoding='utf-8')
        candidates.append({'candidate_id':identifier,'scene':identifier+'.yaml',
                           'parameters_mm_mm_deg':[float(index),0.,0.]})
    hashes={p.name:sha256(p) for p in pool.iterdir()}
    write_json(pool/'pool.json',{'candidates':candidates,'input_sha256':hashes})
    runtime=root/'runtime/files';runtime.mkdir(parents=True)
    write_json(runtime.parent/'manifest.json',{'kind':'synthetic_CPU_runtime_fixture'})
    (root/'toolkit.zip').write_bytes(b'synthetic_CPU_toolkit_fixture')
    request={'kind':'two_fixed_transaction_v3','candidate_ids':['candidate_000','candidate_001'],
             'new_simulations_max':2,'new_Dyn_rollouts':0,'candidate_pool_released':False,
             'expected_compute_environment':{'synthetic_CPU':True},'required_files':{},
             'source_pool':str(pool),'source_pool_sha256':sha256(pool/'pool.json'),
             'topology_source_sha256':'a'*64}
    path=root/'request.json';write_json(path,request)
    return path,runtime,pool


def synthetic_capture(command,pool,exit_code=0,reset_bad=False,extra_step=False,tool_bad=False):
    """Hand-defined translated grippers and integer clock, never model output."""
    target=Path(command[command.index('--output')+1]);target.mkdir(parents=True)
    (target/'partial_capture.txt').write_text('synthetic fixture; not actual simulation', encoding='utf-8')
    if exit_code:return NS(returncode=exit_code)
    shift=0.005 if target.name=='candidate_001' else 0.
    for source,destination in [('shared_source_state.npz','reset_returned.npz'),
                               ('static_material_info.npz','static_material_info.npz')]:
        shutil.copyfile(pool/source,target/destination)
    if reset_bad:np.savez(target/'reset_returned.npz',material=np.array([99.,2.]))
    traces=target/'actual_tools';traces.mkdir()
    scenes=target/'scenes/0000';scenes.mkdir(parents=True)
    events=[];rows=[];active=counter=0
    def event(kind,**fields):
        events.append(dict(event=len(events)+1,kind=kind,active_substeps=active,**fields))
    def step():
        nonlocal active,counter
        before=counter;active+=10;counter+=10
        event('step',n_substeps=10,solver_counter_before=before,solver_counter_after=counter)
    event('reset_origin',solver_counter_after=0)
    for frame in range(63):
        for _ in range(1 if frame==0 else 16):step()
        if extra_step and frame==25:step()
        packet={'step':np.asarray(frame*16),'sample_global_substep':np.asarray(counter),
                'current_local_substep':np.asarray(counter%50),
                'previous_local_substep':np.asarray((counter-1)%50),
                'substep_dt':np.asarray(.0002),'active_branch_clock_known':np.asarray(True),
                'active_branch_substeps':np.asarray(active),'clock_event_count':np.asarray(len(events))}
        left=np.array([shift,-.1,0.]);right=np.array([shift,.1,0.]);quat=np.array([1.,0.,0.,0.])
        for i,pos in ((1,left),(2,right)):
            packet['tool_%03d_pos'%i]=pos.copy()
            packet['previous_tool_%03d_pos'%i]=pos.copy()
            packet['latest_tool_%03d_pos'%i]=pos.copy()
            for key in ('tool_%03d_quat'%i,'previous_tool_%03d_quat'%i,'latest_tool_%03d_quat'%i):
                packet[key]=quat.copy()
        if tool_bad and target.name=='candidate_001' and frame==60:packet['tool_001_pos'][0]+=.01
        np.savez(traces/('frame_%03d.npz'%frame),**packet)
        rows.append({'step':frame*16,'frame_idx':frame,'t':frame*.032,
                     'ee':{'pos':np.array([shift,0.,0.]),'pos_left':left,'pos_right':right,
                           'quat':quat,'open':.2}})
    (target/'clock_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events),encoding='utf-8')
    with (scenes/'log.pkl').open('xb') as stream:pickle.dump(rows,stream,protocol=4)
    return NS(returncode=0)


class TransactionRunnerTests(unittest.TestCase):
    def exercise(self,root,binding_guard=False,binding_pass=True,**faults):
        request,runtime,pool=make_fixture(root);calls=[]
        if binding_guard:
            builder=runtime/'sim/generate/builder.py';builder.parent.mkdir(parents=True)
            builder.write_text('synthetic CPU builder body only\n', encoding='utf-8')
            value=read_json(request)
            value.update(gpu_binding_policy='preserve_single_PBS_allocation_v1',gpu_builder_source_sha256=sha256(builder))
            request.write_text(json.dumps(value), encoding='utf-8')
        yaml=NS(OmegaConf=NS(load=lambda path:NS(actions=NS(grasp=NS(to_pos=[0.,0.,0.])))))
        def execute(command,**kwargs):
            calls.append(command)
            result=synthetic_capture(command,pool,**faults)
            if binding_guard and result.returncode==0:
                target=Path(command[command.index('--output')+1])
                write_json(target/'gpu_binding_after_init.json',{
                    'status':'passed_simulator_initialization_only' if binding_pass else 'failed',
                    'binding_unchanged':binding_pass,'synthetic_CPU_fixture':True})
            return result
        with patch.object(runner,'environment',return_value={'synthetic_CPU':True}),\
                patch.object(runner,'assemble_runtime',return_value=runtime),\
                patch('level4.artifacts.runtime_commit',return_value='synthetic'),\
                patch.object(runner.subprocess,'run',side_effect=execute),\
                patch.dict('sys.modules',{'omegaconf':yaml}),patch('sys.stdout',io.StringIO()):
            code=runner.collect(request,root/'data')
        return code,calls

    def test_binding_guard_flag_reaches_both_fixed_children(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root,binding_guard=True)
            self.assertEqual(code,0);self.assertEqual(len(calls),2)
            for call in calls:
                self.assertIn('--gpu-builder-source-sha256',call)
                self.assertEqual(call[call.index('--gpu-builder-source-sha256')+1],
                                 sha256(root/'runtime/files/sim/generate/builder.py'))
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_failed_binding_receipt_stops_remaining_candidate_and_stays_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root,binding_guard=True,binding_pass=False)
            self.assertEqual(code,1);self.assertEqual(len(calls),1)
            ledger=read_json(root/'data/candidate_000_status_after_execution.json')
            self.assertEqual(ledger['candidate_001']['execution'],'not_executed')
            self.assertEqual(read_json(root/'data/stage_exit.json')['exit_code'],1)
            self.assertIn('binding guard',read_json(root/'data/probe_failure.json')['error'])
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_complete_pair_records_distinct_snapshots_and_verifiable_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root)
            self.assertEqual(code,0);self.assertEqual(len(calls),2)
            data=root/'data'
            before=read_json(data/'candidate_status_before.json')
            running=read_json(data/'candidate_000_status_before_execution.json')
            returned=read_json(data/'candidate_000_status_after_execution.json')
            final=read_json(data/'candidate_status_final.json')
            self.assertEqual(before['candidate_000']['execution'],'not_executed')
            self.assertEqual(running['candidate_000']['execution'],'running')
            self.assertEqual(returned['candidate_000']['execution'],'completed')
            self.assertEqual(returned['candidate_001']['execution'],'not_executed')
            for index in range(16):
                row=final['candidate_%03d'%index]
                self.assertEqual(row['execution'],'completed' if index<2 else 'not_executed')
                self.assertEqual(row['prediction'],'not_run')
            report=read_json(data/'diagnostic_report.json')
            self.assertTrue(report['clock_labels_match'])
            self.assertEqual(report['tool_transform_status'],'within_original_numeric_limits')
            self.assertFalse(report['candidate_pool_released'])
            verified=verify_zip(root/'diagnostic_evidence.zip')
            self.assertEqual(verified['status'],'passed')
            self.assertEqual(read_json(data/'stage_exit.json')['execution_subprocess_exits_recorded'],2)
            for call in calls:
                self.assertEqual(call[call.index('--topology-source-sha256')+1],'a'*64)

    def test_child_failure_retains_exit_and_partial_data_stops_second_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root,exit_code=7)
            self.assertEqual(code,1);self.assertEqual(len(calls),1)
            data=root/'data'
            self.assertEqual(read_json(data/'candidate_000_exit.json')['exit_code'],7)
            ledger=read_json(data/'candidate_000_status_after_execution.json')
            self.assertEqual(ledger['candidate_000']['execution'],'failed')
            self.assertEqual(ledger['candidate_001']['execution'],'not_executed')
            self.assertTrue((data/'execution/candidate_000/partial_capture.txt').is_file())
            self.assertEqual(verify_zip(root/'diagnostic_evidence.zip')['status'],'passed')

    def test_failed_reset_does_not_turn_child_exit_zero_into_interface_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root,reset_bad=True)
            self.assertEqual(code,1);self.assertEqual(len(calls),1)
            data=root/'data'
            self.assertEqual(read_json(data/'candidate_000_exit.json')['exit_code'],0)
            self.assertIn('not exactly equal',read_json(data/'probe_failure.json')['error'])
            self.assertFalse(read_json(data/'diagnostic_report.json')['candidate_pool_released'])

    def test_collection_zero_does_not_hide_time_or_tool_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);code,calls=self.exercise(root,extra_step=True,tool_bad=True)
            self.assertEqual(code,0);self.assertEqual(len(calls),2)
            report=read_json(root/'data/diagnostic_report.json')
            self.assertFalse(report['clock_labels_match'])
            self.assertEqual(report['tool_transform_status'],'violated')
            self.assertFalse(report['candidate_pool_released'])

    def test_second_invocation_rejects_existing_output_without_changing_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.exercise(root)
            original={p.relative_to(root/'data').as_posix():sha256(p) for p in (root/'data').rglob('*') if p.is_file()}
            with patch.object(runner.subprocess,'run') as child:
                with self.assertRaises(FileExistsError):runner.collect(root/'request.json',root/'data')
                child.assert_not_called()
            after={p.relative_to(root/'data').as_posix():sha256(p) for p in (root/'data').rglob('*') if p.is_file()}
            self.assertEqual(original,after)


if __name__=='__main__':unittest.main()
