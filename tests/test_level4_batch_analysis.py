"""L4-01 Phase C statistics: hand-checked examples."""
import unittest

import numpy as np

from level4 import batch_analysis as ba


class BatchAnalysisTests(unittest.TestCase):
    def test_task_attributes(self):
        times = [0.0, 0.96, 1.92, 1.984]
        out = ba.task_attributes(times, [63, 60, 50.4, 40], [63, 66, 63, 90], 63.0, 63.0)
        self.assertAlmostEqual(out['T1_r_u_end'], 0.8)
        self.assertAlmostEqual(out['T2_max_abs_v_dev'], 3 / 63)  # 1.984 s is beyond the primary end
        self.assertIsNone(ba.task_attributes(times, [63, 60, None, 1], [63] * 4, 63., 63.)['T1_r_u_end'])
        self.assertIsNone(ba.task_attributes(times, [63] * 4, [63, None, 63, 63], 63., 63.)['T2_max_abs_v_dev'])
        with self.assertRaises(ValueError):
            ba.task_attributes([0, 1.0], [1, 1], [1, 1], 1., 1.)

    def test_ranks_and_spearman(self):
        np.testing.assert_array_equal(ba.ranks([10, 20, 20, 5]), [2, 3.5, 3.5, 1])
        perfect = ba.spearman([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], permutations=2000)
        self.assertAlmostEqual(perfect['rho'], 1.0)
        self.assertLess(perfect['p_two_sided'], 0.01)
        self.assertAlmostEqual(ba.spearman([1, 2, 3, 4], [4, 3, 2, 1], permutations=100)['rho'], -1.0)

    def test_top_k_tie_break_by_id(self):
        mask = ba.top_k_mask([0.5, 0.2, 0.2, 0.9], ['c', 'b', 'a', 'd'], 2)
        np.testing.assert_array_equal(mask, [False, True, True, False])
        mask = ba.top_k_mask([0.5, 0.2, 0.2, 0.9], ['c', 'b', 'a', 'd'], 1)
        np.testing.assert_array_equal(mask, [False, False, True, False])

    def test_permutation_detects_optimistic_selection(self):
        rng = np.random.default_rng(1)
        residual = rng.normal(0, 0.01, 31)
        mask = np.zeros(31, dtype=bool); mask[:8] = True
        residual[mask] -= 0.05
        out = ba.permutation_mean_difference(residual, mask, permutations=4000)
        self.assertLess(out['selected_minus_rest_mean'], -0.04)
        self.assertLess(out['p_one_sided_optimistic'], 0.01)
        self.assertEqual(ba.d1_verdict(out, None, None)['verdict'], 'support')
        null = ba.permutation_mean_difference(rng.normal(0, 0.01, 31), mask, permutations=2000)
        self.assertIn(ba.d1_verdict(null, None, None)['verdict'], ('not_support', 'uncertain'))
        flipped = dict(out, selected_minus_rest_mean=0.05)
        self.assertEqual(ba.d1_verdict(flipped, None, None)['verdict'], 'not_support')
        few = dict(out, selected_n=5)
        self.assertEqual(ba.d1_verdict(few, None, None)['verdict'], 'uncertain_insufficient_sample')

    def test_false_feasible_and_sweep(self):
        pred_ok = np.array([1, 1, 1, 0], bool); exec_ok = np.array([1, 0, 0, 0], bool)
        self.assertAlmostEqual(ba.false_feasible(pred_ok, exec_ok, np.array([1, 1, 0, 1], bool)), 0.5)
        self.assertIsNone(ba.false_feasible(pred_ok, exec_ok, np.array([0, 0, 0, 1], bool)))
        rows = ba.threshold_sweep([0.70, 0.75, 0.90, 0.95], [0.72, 0.85, 0.91, 0.96], list('abcd'),
                                  [0.8, 0.92], k=2, draws=500)
        self.assertEqual([r['threshold'] for r in rows], [0.8, 0.92])
        self.assertAlmostEqual(rows[0]['top_k_false_feasible'], 0.5)  # a ok, b predicted ok but 0.85>0.8
        self.assertAlmostEqual(rows[1]['top_k_false_feasible'], 0.0)

    def test_samples_needed(self):
        self.assertEqual(ba.samples_per_group(-0.01, 0.02), 50)
        self.assertIsNone(ba.samples_per_group(0.0, 0.02))

    def test_sensitivity_judgement(self):
        ok = ba.sensitivity_judgement({'span_u_mm': [0.01, -0.05], 'span_v_mm': [0.0], 'neck_mm': [0.09],
                                       'components_changed': False})
        self.assertEqual(ok['status'], 'passed')
        bad = ba.sensitivity_judgement({'span_u_mm': [0.12], 'span_v_mm': [0.0], 'neck_mm': [0.0],
                                        'components_changed': False})
        self.assertEqual(bad['status'], 'failed')
        split = ba.sensitivity_judgement({'span_u_mm': [0.0], 'span_v_mm': [0.0], 'neck_mm': [0.0],
                                          'components_changed': True})
        self.assertEqual(split['status'], 'failed')
        undefined = ba.sensitivity_judgement({'span_u_mm': [0.0], 'span_v_mm': [0.0], 'neck_mm': [],
                                              'components_changed': False})
        self.assertEqual((undefined['status'], undefined['not_applicable']), ('passed', ['neck_mm']))


if __name__ == '__main__':
    unittest.main()
