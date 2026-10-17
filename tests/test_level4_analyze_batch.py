"""L4-01 Phase C analysis end to end on a tiny synthetic NSCC layout (CPU, no scipy needed)."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from level4.artifacts import save_packet, sha256, write_json
from level4.geometry import particle_occupancy
from scripts import analyze_level4_batch as az

SCALE = 0.2
SPACING = 0.01  # 2 mm


def grid():
    axes = [np.arange(-0.12, 0.1201, SPACING), np.arange(-0.12, 0.1201, SPACING), np.arange(-0.03, 0.0701, SPACING)]
    query = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    return query, np.array([len(a) for a in axes]), np.full(3, SPACING)


def blob(sx, sy=1.0, split=False):
    xs = np.arange(-3, 4) * 0.01 * sx
    ys = np.arange(-3, 4) * 0.01 * sy
    pts = np.stack(np.meshgrid(xs, ys, [0.0, 0.01], indexing='ij'), -1).reshape(-1, 3)
    if split:
        pts = np.vstack([pts[pts[:, 0] < -0.001] - [0.02, 0, 0], pts[pts[:, 0] >= -0.001] + [0.02, 0, 0]])
    return pts


def frames(final_sx, split=False):
    rows = []
    for f in range(63):
        s = 1 - (1 - final_sx) * f / 62
        rows.append({'t': f * 0.032, 'step': f * 16, 'frame_idx': f, 'topology': {'components': 1},
                     'obj': {'particles': {'pos': blob(s, split=split and f >= 50).astype(np.float32)}}})
    return rows


def write_evidence(data, run_id, rows):
    scene = data / 'execution' / run_id / 'scenes/0000'
    scene.mkdir(parents=True)
    with (scene / 'log.pkl').open('xb') as stream:
        pickle.dump(rows, stream)


def manifest(data):
    write_json(data / 'evidence_manifest.json', {'files_sha256': {
        p.relative_to(data).as_posix(): sha256(p) for p in data.rglob('*') if p.is_file()
        and p.name != 'evidence_manifest.json'}})


def layout(root):
    project = root / 'project'
    query, counts, spacing = grid()
    pool = project / 'records/pool'; pool.mkdir(parents=True)
    np.savez(pool / 'grid.npz', query=query, counts=counts, spacing=spacing, unit_to_meter=np.asarray(SCALE))
    n = len(blob(1.0))
    np.savez(pool / 'static_material_info.npz', mass=np.ones(n, dtype=np.float32))
    write_json(pool / 'pool.json', {'initial_reference_mm': {'u': {'value': 16.0}, 'v': {'value': 16.0}}})
    nominal = project / az.NOMINAL_DIRNAME
    write_evidence(nominal, 'candidate_000', frames(0.8))
    write_evidence(nominal, 'candidate_001', frames(0.75))
    manifest(nominal)
    batch = project / az.BATCH_DIRNAME
    pilot = batch / 'B-pilot_20261017T000000Z'
    runs = {'candidate_002': frames(0.7), 'candidate_016': frames(0.9), 'candidate_032': frames(0.6, split=True),
            'S1_x_0p02': frames(0.8)}
    rows = []
    for rid, log in runs.items():
        write_evidence(pilot / 'data', rid, log)
        rows.append({'run_id': rid, 'candidate_id': 'candidate_000' if rid.startswith('S') else rid,
                     'group': 'sensitivity' if rid.startswith('S') else
                     ('d2_edge' if rid == 'candidate_032' else 'frozen_seed0' if rid == 'candidate_002' else 'seed1_second_group'),
                     'action_mm_mm_deg': [0.0, 0.0, 0.0], 'tool_offset_mm': [0.02, 0.0] if rid.startswith('S') else [0, 0]})
    manifest(pilot / 'data')
    write_json(pilot / 'tools/request.json', {'source_pool': str(pool), 'runs': rows})
    write_json(pilot / 'data/batch_report.json', {'status': 'batch_collected_requires_review',
                                                  'results': {rid: {'exit_code': 0} for rid in runs}})
    dyn = batch / 'B-dyn_20261017T000000Z/data'
    cands = ['candidate_000', 'candidate_001', 'candidate_002', 'candidate_016', 'candidate_032']
    hashes = {}
    for variant, bias in (('legacy_nominal', 0.0), ('v3_nominal', -0.05)):
        for cid, final in zip(cands, (0.8, 0.75, 0.7, 0.9, 0.6)):
            directory = dyn / 'predictions' / variant / cid
            directory.mkdir(parents=True)
            times = np.arange(13) * 0.16
            classes = [particle_occupancy(query, blob(1 - (1 - (final + bias)) * k / 12), 0.01) for k in range(13)]
            save_packet(directory / 'trajectory.npz', times, classes)
            hashes['%s/%s' % (variant, cid)] = {'trajectory.npz': sha256(directory / 'trajectory.npz')}
    write_json(dyn / 'prediction_manifest.json', {'variants': ['legacy_nominal', 'v3_nominal'], 'predictions_sha256': hashes})
    write_json(dyn / 'request.json', {'candidates': [{'candidate_id': c} for c in cands]})
    return project


class AnalyzeBatchTests(unittest.TestCase):
    def test_end_to_end_small_layout(self):
        with tempfile.TemporaryDirectory() as d:
            project = layout(Path(d))
            out = Path(d) / 'analysis'
            with unittest.mock.patch.object(az, 'TOP_K', 2), unittest.mock.patch.object(az, 'T1_GRID', (0.8,)), \
                    unittest.mock.patch.object(az, 'T2_GRID', (0.1,)), unittest.mock.patch.object(az, 'TOLERANCES', (0.2,)):
                self.assertEqual(az.run(out, project, workers=2), 0)
            report = json.loads((out / 'report/phase_c_report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['executed_runs']), 6)
            c002 = next(r for r in report['candidates'] if r['candidate_id'] == 'candidate_002')
            self.assertEqual(c002['execution'], 'completed')
            self.assertLess(c002['exec_common']['T1_r_u_end'], 1.0)
            self.assertLess(c002['pred_v3_nominal']['T1_r_u_end'], c002['pred_legacy_nominal']['T1_r_u_end'] + 1e-9)
            self.assertIsNotNone(c002['diag_voxel_iou_frame60_v3_nominal'])
            d2 = {r['run_id']: r for r in report['C3_D2']['runs']}
            self.assertEqual(d2['candidate_032']['break_frame60'], 'stable_break')
            self.assertEqual(d2['candidate_002']['break_frame60'], 'no_break')
            self.assertTrue(report['C3_D2']['any_break'])
            self.assertEqual(report['S4_selection']['chosen'][0], 'candidate_032')
            s1 = report['C4_S']['S1_x_0p02']['judgement']
            self.assertEqual(s1['status'], 'passed')  # identical synthetic trajectory -> zero change
            self.assertEqual(report['C4_S']['S0_repeat_nominal'], {'status': 'not_executed'})
            self.assertTrue(report['C5_matrices']['rows'])
            self.assertTrue((out / 'report/candidates.csv').is_file())
            self.assertIn('D2 any_break=True', (out / 'report/summary.txt').read_text(encoding='utf-8'))
            # truth is cached and a second report run refuses to overwrite outputs
            with self.assertRaises(FileExistsError):
                az.report(out, project)

    def test_tampered_prediction_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            project = layout(Path(d))
            dyn = next((project / az.BATCH_DIRNAME).glob('B-dyn_*')) / 'data'
            path = dyn / 'predictions/v3_nominal/candidate_002/trajectory.npz'
            data = dict(np.load(path)); data['classes'] = data['classes'] * 0
            path.unlink(); np.savez(path, **data)
            with unittest.mock.patch.object(az, 'TOP_K', 2):
                with self.assertRaisesRegex(ValueError, 'Frozen prediction changed'):
                    az.run(Path(d) / 'analysis', project, workers=1)


import unittest.mock  # noqa: E402

if __name__ == '__main__':
    unittest.main()
