"""L4-01 Phase B: login-node job preparation on a synthetic NSCC layout (no qsub, no GPU)."""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

from scripts import level4_phase_b as pb


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def layout(root):
    repo = root / 'repo'; (repo / 'scripts').mkdir(parents=True); (repo / 'level4').mkdir()
    for n in ('run_level4_batch.py', 'predict_level4_batch.py'):
        (repo / 'scripts' / n).write_text('# runner\n', encoding='utf-8')
    for n in pb.NEW_OVERLAYS:
        (repo / n).write_text('# overlay\n', encoding='utf-8')
    (repo / 'docs').mkdir(); (repo / 'docs/level4_l4_01_next_minimal_experiment.md').write_text('plan\n', encoding='utf-8')
    config = repo / 'configs/level4/l4_01'; (config / 'scenes').mkdir(parents=True)
    project = root / 'project'
    pool = project / 'records/pool'; pool.mkdir(parents=True)
    cands, hashes = [], {}
    for i in range(36):
        cid = 'candidate_%03d' % i
        group = 'frozen_seed0' if i < 16 else ('seed1_second_group' if i < 32 else 'd2_edge')
        place = pool if i < 16 else config / 'scenes'
        (place / (cid + '.yaml')).write_text('scene %s\n' % cid, encoding='utf-8')
        h = sha(place / (cid + '.yaml'))
        if i < 16: hashes[cid + '.yaml'] = h
        planv = 'existing_execution_reuse' if i < 2 else ('excluded_pre_screen_interpenetrating_at_frame0' if i == 24 else 'execute_and_predict')
        cands.append({'candidate_id': cid, 'group': group, 'parameters_mm_mm_deg': [float(i), 0, 0],
                      'scene': cid + '.yaml', 'scene_sha256': h, 'phase_b_plan': planv})
    (pool / 'pool.json').write_text(json.dumps({'input_sha256': hashes}), encoding='utf-8')
    (config / 'pool32_manifest.json').write_text(json.dumps({'candidates': cands}), encoding='utf-8')
    runs = []
    for rid, off in (('S0_repeat_nominal', [0, 0]), ('S1_x_0p02', [0.02, 0]), ('S1_y_0p02', [0, 0.02]), ('S2_x_0p1', [0.1, 0])):
        (config / 'scenes' / (rid + '.yaml')).write_text('s %s\n' % rid, encoding='utf-8')
        runs.append({'run_id': rid, 'source_candidate': 'candidate_000', 'action_mm_mm_deg': [0, 0, 0],
                     'tool_offset_mm': off, 'scene': rid + '.yaml', 'scene_sha256': sha(config / 'scenes' / (rid + '.yaml'))})
    (config / 'sensitivity_manifest.json').write_text(json.dumps({'runs': runs}), encoding='utf-8')
    frozen = project / pb.FROZEN22_DIRNAME; tools = frozen / 'tools'; tools.mkdir(parents=True)
    (frozen / 'data').mkdir(); (frozen / 'data/evidence_manifest.json').write_text('{}', encoding='utf-8')
    legacy = project / 'records/core/source_scene'; legacy.mkdir(parents=True)
    (legacy / 'log.pkl').write_bytes(b'x'); (legacy / 'config.yaml').write_text('c', encoding='utf-8')
    overlays = {'level4/__init__.py': 'l4___init__.py'}
    for n in list(overlays.values()) + list(pb.FROZEN_REUSED):
        (tools / n).write_text('# frozen %s\n' % n, encoding='utf-8')
    (tools / 'resume_level4_shared_input.py').write_text('def check_files(entries):\n    assert isinstance(entries, dict)\n', encoding='utf-8')
    req = {k: 'x' for k in pb.COPIED_FROM_REQUEST}
    req.update(runtime_overlays=overlays, required_files={}, source_pool=str(pool), legacy_source_log=str(legacy / 'log.pkl'))
    (tools / 'request.json').write_text(json.dumps(req), encoding='utf-8')
    (tools / 'tool_manifest.json').write_text(json.dumps({'files_sha256': {p.name: sha(p) for p in tools.iterdir()}}), encoding='utf-8')
    return repo, config, project, tools


class PhaseBPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.repo, self.config, self.project, tools = layout(root)
        self.patches = [patch.object(pb, 'REPO', self.repo), patch.object(pb, 'CONFIG', self.config),
                        patch.object(pb, 'FROZEN22_REQUEST_SHA256', sha(tools / 'request.json')),
                        patch.object(pb, 'FROZEN22_TOOL_MANIFEST_SHA256', sha(tools / 'tool_manifest.json'))]
        for p in self.patches: p.start()
        sys.modules.pop('resume_level4_shared_input', None)
    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def test_plan_partitions_37_runs(self):
        plan = pb.load_plan(self.config)
        sizes = {k: len(v) for k, v in plan['jobs'].items()}
        self.assertEqual(sizes, {'B-pilot': 4, 'B-exec-1': 9, 'B-exec-2': 8, 'B-exec-3': 8, 'B-exec-4': 8})
        flat = [r for v in plan['jobs'].values() for r in v]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertNotIn('candidate_024', flat); self.assertNotIn('candidate_000', flat)
        self.assertEqual(len(plan['predict_candidates']), 35)

    def test_prepare_pilot_and_dyn(self):
        parent = pb.prepare('B-pilot', self.project, submit=False)
        req = json.loads((parent / 'tools/request.json').read_text(encoding='utf-8'))
        self.assertEqual([r['run_id'] for r in req['runs']], list(pb.PILOT))
        self.assertEqual(req['new_simulations_max'], 4)
        self.assertIn('level4/tool_interface_contract.py', req['runtime_overlays'])
        man = json.loads((parent / 'tools/tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
        for name in ('run_level4_batch.py', 'run_job.pbs', 'request.json', 'candidate_002.yaml', 'S1_x_0p02.yaml',
                     'l4_tool_interface_contract.py', 'probe_level4_candidate_interface.py'):
            self.assertIn(name, man)
        self.assertTrue((parent / 'login_verification.json').is_file())
        dyn = pb.prepare('B-dyn', self.project, submit=False)
        dreq = json.loads((dyn / 'tools/request.json').read_text(encoding='utf-8'))
        self.assertEqual(dreq['new_Dyn_rollouts'], 70)
        self.assertEqual(dreq['new_simulations_max'], 0)

    def test_exec_refused_before_pilot_review(self):
        with self.assertRaisesRegex(ValueError, 'B-pilot first'):
            pb.prepare('B-exec-1', self.project, submit=False)
        pilot = pb.prepare('B-pilot', self.project, submit=False)
        (pilot / 'data').mkdir()
        good = {'exit_code': 0, 'binding': {'status': 'passed'}, 'reset': {'status': 'passed'},
                'clock': {'status': 'verified_recorded_events_labels_match'}, 'tool_interface': {'status': 'passed_v2'}}
        report = {'status': 'batch_collected_requires_review', 'executed_runs': 4,
                  'results': {r: dict(good) for r in pb.PILOT}}
        report['results']['S1_x_0p02']['tool_interface'] = {'status': 'failed_v2'}
        (pilot / 'data/batch_report.json').write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'S1_x_0p02'):
            pb.prepare('B-exec-1', self.project, submit=False)
        report['results']['S1_x_0p02']['tool_interface'] = {'status': 'passed_v2'}
        (pilot / 'data/batch_report.json').write_text(json.dumps(report), encoding='utf-8')
        parent = pb.prepare('B-exec-1', self.project, submit=False)
        self.assertEqual(len(json.loads((parent / 'tools/request.json').read_text(encoding='utf-8'))['runs']), 9)

    def test_tampered_frozen_scene_refused(self):
        (Path(self.project) / 'records/pool/candidate_002.yaml').write_text('changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'candidate_002'):
            pb.prepare('B-pilot', self.project, submit=False)


class CommittedPlanTests(unittest.TestCase):
    def test_committed_l4_01_configs_give_the_frozen_plan(self):
        plan = pb.load_plan()
        self.assertEqual(plan['jobs']['B-pilot'], list(pb.PILOT))
        self.assertEqual(sum(len(v) for v in plan['jobs'].values()), 37)
        self.assertEqual(len(plan['predict_candidates']), 35)
        for row in plan['runs'].values():
            if row['scene_origin'] == 'repo':
                self.assertEqual(sha(pb.CONFIG / 'scenes' / row['scene']), row['scene_sha256'])


if __name__ == '__main__':
    unittest.main()
