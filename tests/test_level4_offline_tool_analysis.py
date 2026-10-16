import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analyze_level4_tool_transform_offline import analyze
from scripts.plan_level3_pose_cem import transform_tools


def _packet(path, step, left, right):
    np.savez(path, step=np.asarray(step),
             tool_001_pos=np.asarray(left, dtype=np.float32),
             tool_002_pos=np.asarray(right, dtype=np.float32),
             latest_tool_001_pos=np.asarray(left, dtype=np.float32),
             latest_tool_002_pos=np.asarray(right, dtype=np.float32),
             previous_tool_001_pos=np.asarray(left, dtype=np.float32),
             previous_tool_002_pos=np.asarray(right, dtype=np.float32))


class OfflineToolAnalysisTests(unittest.TestCase):
    def test_frozen_transform_is_recovered_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence'
            nominal = evidence / 'execution' / 'candidate_000' / 'actual_tools'
            candidate = evidence / 'execution' / 'candidate_001' / 'actual_tools'
            nominal.mkdir(parents=True)
            candidate.mkdir(parents=True)
            parameters = [1.0, -2.0, 5.0]
            source_left = np.array([-0.21650635, -0.125, 0.0])
            source_right = np.array([0.21650635, 0.125, 0.0])
            for frame in range(63):
                _packet(nominal / ('frame_%03d.npz' % frame), frame * 16,
                        source_left, source_right)
                transformed = transform_tools(
                    np.asarray([[source_left, source_right]]), np.zeros(3),
                    [parameters], .2)[0, 0, :, :3]
                _packet(candidate / ('frame_%03d.npz' % frame), frame * 16,
                        transformed[0], transformed[1])
            (evidence / 'prospective_tool_diagnosis.json').write_text('{}', encoding='utf-8')
            (evidence / 'source_pool_manifest.json').write_text(json.dumps({
                'candidates': [{'candidate_id': 'candidate_001',
                                'parameters_mm_mm_deg': parameters}]}), encoding='utf-8')
            before = (candidate / 'frame_026.npz').read_bytes()
            result = analyze(evidence, nominal, ['candidate_001'])
            summary = result['analysis']['candidate_001']['hypothesis_summaries']
            self.assertLessEqual(
                summary['frozen_center0_mm_unit02_yaw_plus_translate_current']['max_error_mm'],
                1e-5)
            self.assertTrue(summary['center0_mm_unit02_yaw_plus_translate_after_rotation_current']
                            ['exceeds_original_limit'])
            self.assertEqual(before, (candidate / 'frame_026.npz').read_bytes())
            self.assertFalse(result['interpretation']['candidate_pool_release'])

    def test_missing_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'evidence'
            nominal = evidence / 'nominal'
            actual = evidence / 'execution' / 'candidate_001' / 'actual_tools'
            nominal.mkdir(parents=True)
            actual.mkdir(parents=True)
            for frame in range(62):
                _packet(nominal / ('frame_%03d.npz' % frame), frame * 16,
                        [0, 0, 0], [1, 0, 0])
                _packet(actual / ('frame_%03d.npz' % frame), frame * 16,
                        [0, 0, 0], [1, 0, 0])
            (evidence / 'prospective_tool_diagnosis.json').write_text('{}', encoding='utf-8')
            (evidence / 'source_pool_manifest.json').write_text(json.dumps({
                'candidates': [{'candidate_id': 'candidate_001',
                                'parameters_mm_mm_deg': [0, 0, 0]}]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Missing saved nominal'):
                analyze(evidence, nominal, ['candidate_001'])



class TransformHypothesisExpectationTests(unittest.TestCase):
    """L4-01 A3: hand-derived expectations, independent of transform_tools.

    Points are simulator units; translations are mm; unit_to_meter=0.2 so
    1 mm = 1/(1000*0.2) = 0.005 simulator units.  z is never changed.
    """
    POINT = np.array([0.3, -0.1, 0.05])
    CENTER = np.array([0.1, 0.2, 0.0])

    def check(self, parameters, order, expected, center=None):
        from scripts.analyze_level4_tool_transform_offline import _apply_hypothesis
        center = self.CENTER if center is None else center
        before = self.POINT.copy()
        actual = _apply_hypothesis(self.POINT, center, parameters, 0.2, order)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(self.POINT, before)

    def test_zero_transform_is_identity_for_both_orders(self):
        for order in ('rotate_then_translate', 'translate_then_rotate'):
            self.check([0.0, 0.0, 0.0], order, [0.3, -0.1, 0.05])
            self.check([0.0, 0.0, 0.0], order, [0.3, -0.1, 0.05], center=np.zeros(3))

    def test_pure_translation_converts_only_mm_offset(self):
        for order in ('rotate_then_translate', 'translate_then_rotate'):
            self.check([2.0, -4.0, 0.0], order, [0.31, -0.12, 0.05])

    def test_positive_and_negative_yaw_about_nonzero_center(self):
        # relative = (0.2, -0.3); +90 deg -> (0.3, 0.2); -90 deg -> (-0.3, -0.2)
        for order in ('rotate_then_translate', 'translate_then_rotate'):
            self.check([0.0, 0.0, 90.0], order, [0.4, 0.4, 0.05])
            self.check([0.0, 0.0, -90.0], order, [-0.2, 0.0, 0.05])

    def test_translation_order_differs_only_by_rotated_offset(self):
        # t = 2 mm in x = 0.01 sim; translate first: relative (0.21,-0.3) -> (0.3,0.21)
        self.check([2.0, 0.0, 90.0], 'translate_then_rotate', [0.4, 0.41, 0.05])
        self.check([2.0, 0.0, 90.0], 'rotate_then_translate', [0.41, 0.4, 0.05])
        self.check([2.0, 0.0, -90.0], 'translate_then_rotate', [-0.2, -0.01, 0.05])
        self.check([2.0, 0.0, -90.0], 'rotate_then_translate', [-0.19, 0.0, 0.05])


if __name__ == '__main__':
    unittest.main()
