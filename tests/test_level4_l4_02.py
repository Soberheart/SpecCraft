"""L4-02 CPU checks: D4 rules, A1/A2, A3, A4 helpers, B1 noise floor and the B-noise plan (numpy only)."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np

from level4 import d4_metrics as dm
from scripts import analyze_level4_batch as az
from scripts import l4_02_attributes as a12
from scripts import l4_02_component_check as a3
from scripts import l4_02_d4 as d4
from scripts import l4_02_training_distribution as a4
from scripts import level4_phase_b as pb
from tests import test_level4_analyze_batch as tab

ORIGINAL_FRAMES = tab.frames


class D4MetricTests(unittest.TestCase):
    def test_official_grid_matches_level3_diagnostic(self):
        query, steps = dm.official_query()
        self.assertEqual(steps.tolist(), [61, 105, 13])
        self.assertEqual(len(query), 83265)
        self.assertAlmostEqual(float(query[:, 2].min()) * 200, -1.0, places=4)

    def test_masking_attributes_and_subground(self):
        query, _ = dm.official_query()
        x, y, z = query[:, 0] * 200, query[:, 1] * 200, query[:, 2] * 200
        cls = ((abs(x) <= 20) & (abs(y) <= 10) & (z <= 9)).astype(np.int8)
        masked = dm.mask_like_evaluater(cls, query)
        self.assertEqual(int(masked[z < 0].sum()), 0)
        a = dm.attributes(masked, query, 0.0)
        self.assertAlmostEqual(a['span_u_mm'], 42.0, places=3)   # 21 cells x 2 mm
        self.assertAlmostEqual(a['span_v_mm'], 22.0, places=3)
        h = dm.subground_histogram(cls, query)
        self.assertGreater(h['below_zero'], 0)
        self.assertEqual(h['bins']['-1_to_0'], h['below_zero'])
        self.assertEqual(h['share_outside_guaranteed_domain'], 0.0)
        grid = np.stack(np.meshgrid([0.0], [0.0], np.arange(-0.1, 0.01, 0.005), indexing='ij'), -1).reshape(-1, 3)
        h = dm.subground_histogram(np.ones(len(grid), np.int8), grid)
        self.assertEqual(sum(h['bins'].values()), h['below_zero'])
        self.assertGreater(h['share_outside_guaranteed_domain'], 0.5)

    def test_original_rule_cannot_pass_the_l3_reference_itself(self):
        # the frozen P1 compares a frame-60 value with the L3 12-interval mean; L3's own frame 60 is 57.3
        p = dm.p_original(dm.L3_ROLLOUT_F60, [dm.L3_ROLLOUT_F60] * 30, [dm.L3_SINGLE_MEAN] * 360)
        self.assertEqual(p['P1'], 'fail')
        q = dm.p_corrected(dm.L3_ROLLOUT_MEAN, dm.L3_ROLLOUT_F60, [dm.L3_ROLLOUT_MEAN] * 30,
                           [dm.L3_ROLLOUT_F60] * 30, [dm.L3_SINGLE_MEAN] * 30)
        self.assertEqual((q['P1'], q['P2'], q['P3']), ('pass', 'pass', 'pass'))
        self.assertEqual(dm.p_corrected(80, 60, [80] * 5, [60] * 5, [90] * 5)['P2'], 'uncertain')

    def test_p4_d4_route_pc(self):
        ref = {'span_u_mm': 16.0, 'span_v_mm': 10.0, 'neck_mm': 1.0}
        ok = dm.p4({'span_u_mm': [10, 20], 'span_v_mm': [5, 9], 'neck_mm': [0.5, 2]}, ref)
        self.assertEqual(ok['P4'], 'pass')
        bad = dm.p4({'span_u_mm': [400], 'span_v_mm': [5], 'neck_mm': [1]}, ref)
        self.assertEqual(bad['P4'], 'fail')
        tiny = dm.p4({'span_u_mm': [0.1], 'span_v_mm': [5], 'neck_mm': [1]}, ref)
        self.assertEqual(tiny['rows']['span_u_mm']['status'], 'pass_below')
        self.assertEqual(dm.d4_verdict({'P1': 'pass', 'P2': 'pass', 'P3': 'pass'}, ok), 'pass')
        self.assertEqual(dm.d4_verdict({'P1': 'fail', 'P2': 'pass', 'P3': 'pass'}, ok), 'fail')
        self.assertEqual(dm.d4_verdict({'P1': 'pass', 'P2': 'uncertain', 'P3': 'pass'}, ok), 'uncertain')
        rec = [dm.L3_ROLLOUT_MEAN] * 12
        self.assertEqual(dm.positive_control(rec, [dm.L3_SINGLE_MEAN] * 12, 1e-5)['status'], 'normal')
        self.assertEqual(dm.positive_control([70] * 12, [dm.L3_SINGLE_MEAN] * 12, 1e-5)['status'], 'abnormal')
        self.assertEqual(dm.positive_control([dm.L3_ROLLOUT_MEAN - 2] * 12, [dm.L3_SINGLE_MEAN] * 12, 1e-5)['status'],
                         'uncertain')
        self.assertIn('pipeline_bug', dm.route('fail', 'abnormal'))
        self.assertIn('out_of_distribution', dm.route('fail', 'normal'))
        self.assertIn('D4_pass', dm.route('pass', 'normal'))

    def test_z_checks_and_sensitivity(self):
        z = dm.z_checks({'share_outside_guaranteed_domain': 0.9}, {'below_zero_share': 0.05}, 0.1, 0.3)
        self.assertEqual((z['Z1'], z['Z2'], z['Z3']), ('query_domain_extrapolation', 'decoder_property_without_dynamics',
                                                       'aligned'))
        z = dm.z_checks({'share_outside_guaranteed_domain': 0.1}, {'below_zero_share': 0.0}, 0.1, 3.0)
        self.assertEqual((z['Z1'], z['Z2'], z['Z3']), ('in_domain_subground_prediction', 'arises_in_dynamics', 'misaligned'))
        rng = np.random.default_rng(0)
        e = rng.normal(size=30)
        self.assertEqual(dm.sensitivity(e + 0.1 * rng.normal(size=30), e, 2000)['status'], 'tracks_action')
        self.assertEqual(dm.sensitivity(0.01 * rng.normal(size=30), e, 2000)['status'], 'insensitive')
        self.assertEqual(dm.sensitivity([1, 2], [1, 2])['status'], 'uncertain')


def fake_candidate(f60, mean, single, i=0):
    rows = [{'target': t, 'viou': (100.0 if t == 0 else (f60 if t == 60 else mean))} for t in d4.ROLLOUT_FRAMES]
    return {'rollout_frozen': {'v3_nominal': rows, 'legacy_nominal': rows},
            'single_step': [{'target': t, 'viou': single} for t in d4.ROLLOUT_FRAMES[1:]],
            'attributes': {'v3_nominal_f60': {'span_u_mm': 42.0, 'span_v_mm': 64.0, 'neck_mm': 3.0},
                           'truth_f60': {'span_u_mm': 58.0 + 0.1 * i, 'span_v_mm': 74.0, 'neck_mm': 4.0 + 0.1 * i},
                           'v3_nominal_dv_max': {'value': 2.0}, 'truth_dv_max': {'value': 12.0}}}


class D4VerdictTests(unittest.TestCase):
    def test_verdicts_from_raw_metrics(self):
        pc = {'intervals': [{'rollout': {'viou': dm.L3_ROLLOUT_MEAN}, 'single_step': {'viou': dm.L3_SINGLE_MEAN}}] * 12,
              'latent_vs_l3_stored_max_abs': 1e-6, 'z0_vs_l3_reference_max_abs': 0.0}
        cands = {'candidate_%03d' % i: fake_candidate(57.0 + 0.01 * i, 82.0, 92.0, i) for i in range(25)}
        stats = {'l4_c000_rendered_frame0': {'z_min_mm': 0.2}, 'l3_frame0': {'z_min_mm': 0.1}}
        r = d4.verdicts(pc, cands, None, None, {'below_zero_share': 0.0}, {'share_outside_guaranteed_domain': 0.95}, stats)
        self.assertEqual(r['positive_control']['status'], 'normal')
        self.assertEqual(r['D4_original'], 'fail')        # frame 60 against the 12-interval mean
        self.assertEqual(r['D4_corrected'], 'pass')
        self.assertEqual(r['P4']['P4'], 'pass')
        self.assertEqual(r['z_checks']['Z1'], 'query_domain_extrapolation')
        self.assertTrue(r['route_corrected'].startswith('D4_pass'))
        self.assertEqual(r['action_sensitivity']['neck_mm']['status'], 'insensitive')  # constant prediction


class ComponentCheckTests(unittest.TestCase):
    def test_simulator_split_still_touching(self):
        rng = np.random.default_rng(0)
        left = rng.uniform([-10, -5, 0], [0, 5, 5], size=(300, 3))
        right = rng.uniform([0, -5, 0], [10, 5, 5], size=(300, 3))
        p = np.vstack([left, right]) / 200.0
        rows = []
        for f in range(63):
            split = f >= 57
            rows.append({'step': 16 * f, 'topology': {'components': 2 if split else 1, 'genus': [0, 0] if split else [0]},
                         'obj': {'particles': {'pos': p, 'idx': np.r_[np.full(300, 2 if split else 0),
                                                                      np.full(300, 3 if split else 0)]}}})
        r = a3.analyse(rows, 30.0, frames=(0, 60))
        f60 = r['frames'][1]
        self.assertEqual(f60['simulator']['components'], 2)
        self.assertEqual(f60['l4_significant_0p5pct_by_multiplier']['1.0'], 1)
        self.assertLess(f60['cross_part']['min_distance_mm'], r['r0_mm'])
        self.assertGreater(f60['cross_part']['pairs_within_r0'], 0)


def with_labels(final_sx, split=False):
    rows = ORIGINAL_FRAMES(final_sx, split)
    for f, row in enumerate(rows):
        n = len(row['obj']['particles']['pos'])
        sim_split = f >= 57 and final_sx < 0.75
        row['topology'] = {'components': 2 if sim_split else 1, 'genus': [1, 1] if sim_split else [1]}
        row['obj']['particles']['idx'] = np.r_[np.full(n // 2, 2), np.full(n - n // 2, 3)] if sim_split else np.zeros(n, int)
    return rows


class AttributesTests(unittest.TestCase):
    def test_end_to_end_small_layout(self):
        with tempfile.TemporaryDirectory() as d:
            with unittest.mock.patch.object(tab, 'frames', with_labels):
                project = tab.layout(Path(d))
            analysis = project / az.BATCH_DIRNAME / 'C-analysis_20261017T000000Z'
            with unittest.mock.patch.object(az, 'TOP_K', 2), unittest.mock.patch.object(az, 'T1_GRID', (0.8,)), \
                    unittest.mock.patch.object(az, 'T2_GRID', (0.1,)), unittest.mock.patch.object(az, 'TOLERANCES', (0.2,)):
                az.run(analysis / 'data', project, workers=1)
            self.assertEqual(a12.main(['--project', str(project)]), 0)
            out = next(analysis.glob('l4_02_A1A2_*'))
            result = json.loads((out / 'a1a2.json').read_text(encoding='utf-8'))
            self.assertEqual(len(result['candidates']), 5)
            self.assertEqual(result['simulator_labels']['components60_ge2'], 2)   # candidate_002 (0.7) and _032 (0.6)
            necks = [r['neck60_mm'] for r in result['rows'] if r['group'] in a12.CANDIDATE_GROUPS and r['neck60_mm']]
            self.assertEqual(len(result['A2']['T1prime']), 3 if necks else 0)
            self.assertTrue((out / 'task_v2_draft.md').read_text(encoding='utf-8').startswith('# L4-02 A2'))
            self.assertIn('candidate_002', (out / 'attributes.csv').read_text(encoding='utf-8'))


class TrainingDistributionTests(unittest.TestCase):
    def test_helpers(self):
        left = np.c_[np.full(512, -0.05), np.linspace(-0.01, 0.01, 512), np.zeros(512)]
        right = np.c_[np.full(512, 0.05), np.linspace(-0.01, 0.01, 512), np.zeros(512)]
        pose = a4.finger_pose(np.vstack([left, right]))
        self.assertAlmostEqual(pose['yaw_deg'], 0.0)
        self.assertAlmostEqual(pose['gap_mm'], 20.0)
        rot = np.vstack([left, right]) @ np.array([[math.cos(0.5), math.sin(0.5), 0], [-math.sin(0.5), math.cos(0.5), 0],
                                                   [0, 0, 1]])
        self.assertAlmostEqual(a4.finger_pose(rot)['yaw_deg'], math.degrees(0.5), places=4)
        t = a4.topology(np.array([[-1, 1, -1, -1, -1]] * 5 + [[-1, -1, 1, 1, -1]] * 3))
        self.assertTrue(t['split'])
        self.assertEqual(t['first_change_frame'], 5)
        self.assertEqual(a4.shape_class([60, 60, 18], [1]), 'ring_genus1')
        self.assertEqual(a4.shape_class([60, 20, 20], [0]), 'elongated')
        scenes = [dict(group='train', row=i, extent_mm=[60 + i, 60, 18], genus0=[1], genus_end=[1, 1] if i % 2 else [1],
                       split=bool(i % 2), genus_changed=bool(i % 2), shape_class='ring_genus1', first_change_frame=None,
                       pinch={'final_gap_mm': 1.0, 'yaw_deg': 30.0, 'centre_offset_mm': [0, 0]}) for i in range(6)]
        placed = a4.place({'genus0': [1], 'extent_mm': [60, 60, 18]}, scenes)
        self.assertEqual(placed['n_similar_bbox_within_10mm'], 6)
        self.assertAlmostEqual(placed['split_rate_similar'], 0.5)
        self.assertEqual(a4.aggregate(scenes)['by_class']['ring_genus1']['n'], 6)


def attrs(neck_delta=0.0, comps=1, span=0.0):
    return {'raw_particle_span_mm': {'u': [60.0 + span] * 63, 'v': [60.0] * 63},
            'pinch_neck_4mm': [{'width_mm': 6.0 + neck_delta}] * 63,
            'components': {'60': {'by_multiplier': {'1.0': {'significant_0p5pct': comps}}}}}


class NoiseFloorTests(unittest.TestCase):
    def test_three_tiers(self):
        base = {'candidate_000': attrs(), 'S0_repeat_nominal': attrs(0.05), 'N_candidate_000_r1': attrs(-0.02),
                'N_candidate_000_r2': attrs(0.01), 'candidate_012': attrs(), 'N_candidate_012_r1': attrs(0.5),
                'N_candidate_012_r2': attrs(0.9)}
        noise = dict(base, S1_x_0p02=attrs(0.01), S1_y_0p02=attrs(0.06), S4_candidate_012_x_0p02=attrs(0.8))
        self.assertEqual(az.noise_floor(noise)['verdict'], 'tier1_simulation_noise')
        critical = dict(noise, S4_candidate_012_x_0p02=attrs(2.0))
        self.assertEqual(az.noise_floor(critical)['verdict'], 'tier2_critical_sensitivity')
        thick = dict(critical, S1_y_0p02=attrs(0.5))
        self.assertEqual(az.noise_floor(thick)['verdict'], 'tier3_0p02_fails_for_neck')
        topo = dict(noise, S4_candidate_012_x_0p02=attrs(0.1, comps=2))
        self.assertEqual(az.noise_floor(topo)['verdict'], 'tier2_critical_sensitivity')
        self.assertEqual(az.noise_floor({'candidate_000': attrs()})['verdict'], 'incomplete')

    def test_noise_plan_uses_frozen_scenes(self):
        plan = pb.load_plan()
        ids = pb.noise_runs(plan)
        self.assertEqual(ids, ['N_candidate_000_r1', 'N_candidate_000_r2', 'N_candidate_012_r1', 'N_candidate_012_r2'])
        for rid in ids:
            row = plan['runs'][rid]
            self.assertEqual(row['tool_offset_mm'], [0.0, 0.0])
            self.assertEqual(row['scene_origin'], 'frozen_pool')
            self.assertEqual(row['scene'], row['candidate_id'] + '.yaml')


if __name__ == '__main__':
    unittest.main()
