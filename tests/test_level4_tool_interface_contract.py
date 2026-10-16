"""tool_interface_contract v1/v2 (decision D-1) on saved and synthetic metrics."""
import copy
from pathlib import Path
import tempfile
import unittest

from level4.tool_interface_contract import CONTRACTS, EXPECTED_METRICS, evaluate

# Saved maxima of job 25837718 prospective_tool_diagnosis.json (candidate_001).
JOB_25837718 = {
    'summary_pos_transform_error_mm': (2.9802322387695312e-05, 23),
    'summary_pos_left_transform_error_mm': (0.0034123659133911133, 32),
    'summary_pos_right_transform_error_mm': (0.0034540891647338867, 32),
    'summary_opening_error_mm': (0.008532404899597168, 32),
    'finger_1_physical_transform_error_mm': (0.004108250141143799, 26),
    'finger_2_physical_transform_error_mm': (0.004149973392486572, 26),
    'finger_1_latest_previous_error_mm': (0.0, 0), 'finger_2_latest_previous_error_mm': (0.0, 0),
    'finger_1_summary_latest_error_mm': (0.002148747444152832, 25),
    'finger_2_summary_latest_error_mm': (0.002148747444152832, 25),
    'finger_1_current_latest_error_mm': (0.004024803638458252, 26),
    'finger_2_current_latest_error_mm': (0.004024803638458252, 26),
    'finger_1_quaternion_error': (1.4159307948347077e-08, 0),
    'finger_2_quaternion_error': (1.4159307948347077e-08, 0),
}


def metrics(**override):
    rows = {k: {'max': v, 'frame': f} for k, (v, f) in JOB_25837718.items()}
    for key, value in override.items():
        rows[key]['max'] = value
    return rows


class ToolInterfaceContractTests(unittest.TestCase):
    def test_limits_are_the_pre_registered_values(self):
        self.assertEqual(CONTRACTS['tool_interface_contract_v1']['position_limit_mm'], 0.001)
        self.assertEqual(CONTRACTS['tool_interface_contract_v2']['position_limit_mm'], 0.02)
        for spec in CONTRACTS.values():
            self.assertEqual(spec['quaternion_limit'], 1e-6)

    def test_saved_pair_fails_v1_and_passes_v2(self):
        v1 = evaluate(metrics(), 'tool_interface_contract_v1')
        self.assertEqual(v1['status'], 'failed_v1')
        self.assertEqual(v1['failed_metrics'], sorted([
            'summary_pos_left_transform_error_mm', 'summary_pos_right_transform_error_mm',
            'summary_opening_error_mm', 'finger_1_physical_transform_error_mm',
            'finger_2_physical_transform_error_mm']))
        v2 = evaluate(metrics(), 'tool_interface_contract_v2')
        self.assertEqual(v2['status'], 'passed_v2')
        self.assertAlmostEqual(v2['worst_gated_position_mm'], 0.008532404899597168)
        self.assertIn('model_alignment', v2['not_certified'])

    def test_v2_gates_current_latest_and_summary_latest(self):
        bad = metrics(finger_2_current_latest_error_mm=0.021)
        self.assertEqual(evaluate(bad, 'tool_interface_contract_v1')['failed_metrics'],
                         evaluate(metrics(), 'tool_interface_contract_v1')['failed_metrics'])
        self.assertEqual(evaluate(bad, 'tool_interface_contract_v2')['failed_metrics'],
                         ['finger_2_current_latest_error_mm'])
        self.assertEqual(evaluate(metrics(finger_1_summary_latest_error_mm=0.0200001),
                                  'tool_interface_contract_v2')['status'], 'failed_v2')
        self.assertEqual(evaluate(metrics(summary_opening_error_mm=0.02),
                                  'tool_interface_contract_v2')['status'], 'passed_v2')

    def test_known_real_errors_are_still_caught_by_v2(self):
        # D-1 rationale: pre-fix legacy opening drift 0.047 mm; unit/yaw errors are mm-scale.
        self.assertEqual(evaluate(metrics(summary_opening_error_mm=0.047), 'tool_interface_contract_v2')
                         ['failed_metrics'], ['summary_opening_error_mm'])
        self.assertEqual(evaluate(metrics(finger_1_physical_transform_error_mm=9.66),
                                  'tool_interface_contract_v2')['status'], 'failed_v2')

    def test_quaternion_limit_unchanged(self):
        result = evaluate(metrics(finger_1_quaternion_error=2e-6), 'tool_interface_contract_v2')
        self.assertEqual(result['failed_metrics'], ['finger_1_quaternion_error'])

    def test_malformed_input_is_rejected(self):
        rows = metrics(); del rows['finger_1_quaternion_error']
        with self.assertRaisesRegex(ValueError, 'comparison set'):
            evaluate(rows, 'tool_interface_contract_v2')
        rows = metrics(); rows['extra_error_mm'] = {'max': 0.0, 'frame': 0}
        with self.assertRaisesRegex(ValueError, 'comparison set'):
            evaluate(rows, 'tool_interface_contract_v2')
        for value in (float('nan'), -1.0, True, '0.0'):
            with self.assertRaises(ValueError):
                evaluate(metrics(summary_opening_error_mm=value), 'tool_interface_contract_v2')
        with self.assertRaises(ValueError):
            evaluate(metrics(), 'tool_interface_contract_v3')

    def test_evaluate_does_not_mutate_input(self):
        rows = metrics(); before = copy.deepcopy(rows)
        evaluate(rows, 'tool_interface_contract_v2')
        self.assertEqual(rows, before)
        self.assertEqual(set(rows), set(EXPECTED_METRICS))

    def test_v2_audit_refuses_unfrozen_v1_script(self):
        from scripts.audit_level4_pair_contract_v2 import load_v1
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / 'audit_pair_results.py'
            fake.write_text('def audit(e):\n    return {}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'frozen'):
                load_v1(fake)


if __name__ == '__main__':
    unittest.main()
