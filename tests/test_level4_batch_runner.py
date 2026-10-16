"""L4-01 Phase B batch execution runner on synthetic captures (CPU only).

The simulator child, runtime assembly, environment check and OmegaConf are
mocked; real exclusive writers, reset/binding checks, clock replay, tool
diagnosis, contract v2 and the stop rule are exercised.
"""
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from level4.artifacts import read_json, sha256, write_json
from scripts import run_level4_batch as batch

SHIFT_SIM_PER_MM = 1 / 200.0


def capture(target, shared_state, shift_mm=(0.0, 0.0), exit_code=0, reset_bad=False, binding_ok=True):
    """Hand-built translated grippers and integer clock; never real simulation."""
    target.mkdir(parents=True)
    if exit_code:
        return NS(returncode=exit_code)
    shutil.copyfile(shared_state, target / 'reset_returned.npz')
    if reset_bad:
        with np.load(shared_state) as s:
            np.savez(target / 'reset_returned.npz', mpm_pos=s['mpm_pos'] + 0.01)
    write_json(target / 'gpu_binding_after_init.json', {
        'status': 'passed_simulator_initialization_only' if binding_ok else 'failed', 'binding_unchanged': binding_ok})
    sx, sy = shift_mm[0] * SHIFT_SIM_PER_MM, shift_mm[1] * SHIFT_SIM_PER_MM
    traces = target / 'actual_tools'; traces.mkdir()
    scenes = target / 'scenes/0000'; scenes.mkdir(parents=True)
    events, rows = [], []
    state = {'active': 0, 'counter': 0}
    def event(kind, **fields):
        events.append(dict(event=len(events) + 1, kind=kind, active_substeps=state['active'], **fields))
    def step():
        before = state['counter']; state['active'] += 10; state['counter'] += 10
        event('step', n_substeps=10, solver_counter_before=before, solver_counter_after=state['counter'])
    event('reset_origin', solver_counter_after=0)
    quat = np.array([1., 0., 0., 0.])
    for frame in range(63):
        for _ in range(1 if frame == 0 else 16):
            step()
        c = state['counter']
        packet = {'step': np.asarray(frame * 16), 'sample_global_substep': np.asarray(c),
                  'current_local_substep': np.asarray(c % 50), 'previous_local_substep': np.asarray((c - 1) % 50),
                  'substep_dt': np.asarray(.0002), 'active_branch_clock_known': np.asarray(True),
                  'active_branch_substeps': np.asarray(state['active']), 'clock_event_count': np.asarray(len(events))}
        left, right = np.array([sx, sy - .1, 0.]), np.array([sx, sy + .1, 0.])
        for i, pos in ((1, left), (2, right)):
            for prefix in ('', 'previous_', 'latest_'):
                packet[prefix + 'tool_%03d_pos' % i] = pos.copy()
                packet[prefix + 'tool_%03d_quat' % i] = quat.copy()
        np.savez(traces / ('frame_%03d.npz' % frame), **packet)
        rows.append({'step': frame * 16, 'frame_idx': frame, 't': frame * .032,
                     'ee': {'pos': np.array([sx, sy, 0.]), 'pos_left': left, 'pos_right': right, 'quat': quat, 'open': .2}})
    (target / 'clock_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    with (scenes / 'log.pkl').open('xb') as stream:
        pickle.dump(rows, stream, protocol=4)
    return NS(returncode=0)


def fixture(root, runs):
    pool = root / 'pool'; pool.mkdir()
    np.savez(pool / 'shared_source_state.npz', mpm_pos=np.zeros((4, 3), dtype=np.float32))
    np.savez(pool / 'static_material_info.npz', mat_idx=np.zeros(4, dtype=np.int32))
    (pool / 'source_scene.yaml').write_text('synthetic\n', encoding='utf-8')
    write_json(pool / 'pool.json', {'candidates': [], 'input_sha256': {
        n: sha256(pool / n) for n in ('shared_source_state.npz', 'static_material_info.npz', 'source_scene.yaml')}})
    reference = root / 'reference'
    capture(reference / 'execution/candidate_000', pool / 'shared_source_state.npz')
    write_json(reference / 'evidence_manifest.json', {'files_sha256': {
        p.relative_to(reference).as_posix(): sha256(p) for p in reference.rglob('*') if p.is_file()}})
    tools = root / 'parent/tools'; tools.mkdir(parents=True)
    for name in ('probe_level4_candidate_interface.py',):
        (tools / name).write_text('# synthetic\n', encoding='utf-8')
    for r in runs:
        (tools / r['scene']).write_text('synthetic scene %s\n' % r['run_id'], encoding='utf-8')
        r['scene_sha256'] = sha256(tools / r['scene'])
    write_json(tools / 'tool_manifest.json', {'files_sha256': {p.name: sha256(p) for p in tools.iterdir()
                                                               if p.name != 'tool_manifest.json'}})
    runtime = root / 'runtime/files'; (runtime / 'sim/generate').mkdir(parents=True)
    (runtime / 'sim/generate/builder.py').write_text('synthetic builder\n', encoding='utf-8')
    write_json(runtime.parent / 'manifest.json', {'kind': 'synthetic'})
    request = {'kind': batch.KIND, 'runs': runs, 'new_simulations_max': len(runs), 'new_Dyn_rollouts': 0,
               'candidate_pool_released': False, 'stop_rule': {'max_failure_fraction': 0.2, 'stop_on_binding_failure': True},
               'tool_interface_contract': 'tool_interface_contract_v2',
               'gpu_binding_policy': 'preserve_single_PBS_allocation_v1',
               'gpu_builder_source_sha256': sha256(runtime / 'sim/generate/builder.py'),
               'topology_source_sha256': 'a' * 64, 'expected_compute_environment': {'synthetic': True},
               'required_files': {}, 'source_pool': str(pool), 'source_pool_sha256': sha256(pool / 'pool.json'),
               'nominal_reference': {'evidence_dir': str(reference), 'job': 'synthetic'}}
    write_json(tools / 'request.json', request)
    return tools / 'request.json', runtime, pool


def run_row(rid, action=(0., 0., 0.), offset=(0., 0.)):
    return {'run_id': rid, 'candidate_id': rid, 'group': 'synthetic', 'action_mm_mm_deg': list(action),
            'tool_offset_mm': list(offset), 'scene': rid + '.yaml'}


class BatchRunnerTests(unittest.TestCase):
    def exercise(self, runs, faults=None):
        faults = faults or {}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            request, runtime, pool = fixture(root, runs)
            shifts = {r['run_id']: (r['action_mm_mm_deg'][0] + r['tool_offset_mm'][0],
                                    r['action_mm_mm_deg'][1] + r['tool_offset_mm'][1]) for r in runs}
            def child(command, **kwargs):
                target = Path(command[command.index('--output') + 1])
                return capture(target, pool / 'shared_source_state.npz', shifts[target.name], **faults.get(target.name, {}))
            fake_omegaconf = NS(OmegaConf=NS(load=lambda path: NS(actions=NS(grasp=NS(to_pos=[0., 0., 0.])))))
            with patch.object(batch, 'environment', return_value={'synthetic': True}), \
                    patch.object(batch, 'check_files'), \
                    patch.object(batch, 'assemble_runtime', return_value=runtime), \
                    patch('level4.artifacts.runtime_commit', return_value='synthetic'), \
                    patch.dict(sys.modules, {'omegaconf': fake_omegaconf}), \
                    patch('subprocess.run', side_effect=child) as calls:
                output = root / 'parent/data'
                status = batch.collect(request, output)
            report = read_json(output / 'batch_report.json')
            files = sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file())
            zipped = (root / 'parent/batch_evidence.zip').is_file()
            return status, report, calls.call_count, files, zipped

    def test_all_runs_pass_and_offsets_are_compared_as_executed_pose(self):
        runs = [run_row('candidate_002', (1.0, 0.0, 0.0)), run_row('S1_x_0p02', offset=(0.02, 0.0)),
                run_row('S2_x_0p1', offset=(0.1, 0.0))]
        status, report, calls, files, zipped = self.exercise(runs)
        self.assertEqual(status, 0)
        self.assertEqual(report['status'], 'batch_collected_requires_review')
        self.assertEqual((report['executed_runs'], report['failed_runs'], calls), (3, 0, 3))
        for rid in ('candidate_002', 'S1_x_0p02', 'S2_x_0p1'):
            r = report['results'][rid]
            self.assertEqual((r['binding']['status'], r['reset']['status']), ('passed', 'passed'))
            self.assertEqual(r['clock']['status'], 'verified_recorded_events_labels_match')
            self.assertEqual(r['tool_interface']['status'], 'passed_v2')
            self.assertLess(r['tool_interface']['worst_gated_position_mm'], 1e-5)
        self.assertIn('S1_x_0p02_tool_interface.json', files)
        self.assertTrue(zipped)
        self.assertFalse(report['candidate_pool_released'])

    def test_offset_not_declared_is_detected_by_contract(self):
        runs_exec = [run_row('S1_x_0p02', offset=(0.0, 0.0))]
        # executed with a 0.1 mm shift while the request declares none -> v2 failure recorded, not a stop
        def exercise_shifted():
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                request, runtime, pool = fixture(root, runs_exec)
                def child(command, **kwargs):
                    target = Path(command[command.index('--output') + 1])
                    return capture(target, pool / 'shared_source_state.npz', (0.1, 0.0))
                fake = NS(OmegaConf=NS(load=lambda path: NS(actions=NS(grasp=NS(to_pos=[0., 0., 0.])))))
                with patch.object(batch, 'environment', return_value={'synthetic': True}), \
                        patch.object(batch, 'check_files'), patch.object(batch, 'assemble_runtime', return_value=runtime), \
                        patch('level4.artifacts.runtime_commit', return_value='s'), \
                        patch.dict(sys.modules, {'omegaconf': fake}), patch('subprocess.run', side_effect=child):
                    batch.collect(request, root / 'parent/data')
                return read_json(root / 'parent/data/batch_report.json')
        report = exercise_shifted()
        self.assertEqual(report['status'], 'batch_collected_requires_review')
        self.assertEqual(report['results']['S1_x_0p02']['tool_interface']['status'], 'failed_v2')

    def test_failure_fraction_stops_remaining_runs(self):
        runs = [run_row('candidate_%03d' % i) for i in range(2, 7)]  # 5 runs; >1 failure stops
        faults = {'candidate_002': {'exit_code': 1}, 'candidate_003': {'reset_bad': True}}
        status, report, calls, files, zipped = self.exercise(runs, faults)
        self.assertEqual(status, 2)
        self.assertEqual(report['status'], 'stopped_by_rule')
        self.assertEqual(calls, 2)
        self.assertEqual(report['not_executed'], ['candidate_004', 'candidate_005', 'candidate_006'])
        self.assertEqual(report['results']['candidate_003']['reset']['status'], 'failed')
        self.assertTrue(zipped)

    def test_binding_failure_stops_immediately(self):
        runs = [run_row('candidate_%03d' % i) for i in range(2, 12)]
        status, report, calls, files, zipped = self.exercise(runs, {'candidate_002': {'binding_ok': False}})
        self.assertEqual(calls, 1)
        self.assertTrue(report['stop_reason'].startswith('gpu_binding_failure'))

    def test_request_scope_is_enforced(self):
        good = {'kind': batch.KIND, 'runs': [run_row('a')], 'new_simulations_max': 1, 'new_Dyn_rollouts': 0,
                'candidate_pool_released': False, 'stop_rule': {'max_failure_fraction': 0.2, 'stop_on_binding_failure': True},
                'tool_interface_contract': 'tool_interface_contract_v2', 'gpu_binding_policy': 'preserve_single_PBS_allocation_v1'}
        batch.validate_request(good)
        for key, value in (('new_Dyn_rollouts', 1), ('new_simulations_max', 2), ('candidate_pool_released', True),
                           ('tool_interface_contract', 'tool_interface_contract_v1'), ('gpu_binding_policy', None),
                           ('stop_rule', {'max_failure_fraction': 0.5, 'stop_on_binding_failure': True})):
            with self.assertRaises(ValueError):
                batch.validate_request(dict(good, **{key: value}))
        with self.assertRaises(ValueError):
            batch.validate_request(dict(good, runs=[run_row('a'), run_row('a')], new_simulations_max=2))

    def test_comparison_parameters_add_offset_to_action(self):
        self.assertEqual(batch.comparison_parameters(run_row('x', (1.0, -2.0, 5.0), (0.02, 0.1))), [1.02, -1.9, 5.0])


if __name__ == '__main__':
    unittest.main()
