"""CPU restoration tests, explicitly synthetic fixtures, never material evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from level4.artifacts import write_json, read_json, sha256
from scripts.audit_level4_reset_failure import initial_difference
from scripts.restore_level4_evidence import render_topview, restore_reset


class RestoreTests(unittest.TestCase):
    def test_float32_camera_json_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = np.array([[-2, -1, 0], [2, 1, 1]], dtype=np.float32)
            before = p.copy()
            camera = render_topview(p, Path(d) / 'frame.png', 'SYNTHETIC float32 renderer test')
            encoded = json.dumps(camera, allow_nan=False)
            self.assertIs(type(camera['pixels_per_mm']), float)
            self.assertEqual(json.loads(encoded)['center_mm'], [0., 0.])
            np.testing.assert_array_equal(p, before)
            self.assertEqual(p.dtype, np.float32)

    def fixture(self, root):
        evidence = root / 'evidence'; evidence.mkdir()
        ref = np.array([[-1, -1, 0], [1, 1, 1]], dtype=np.float32)
        actual = ref.copy(); actual[0, 2] += np.float32(.00002)
        np.savez_compressed(evidence / 'initial_states.npz', planner_initial_positions=ref,
                            execution_frame0_pos=actual)
        write_json(evidence / 'report.json', {
            'initial_difference': initial_difference(ref, actual, .2, .001),
            'pool': '/unavailable/nscc/path', 'kind': 'synthetic_fixture_not_real_run'})
        write_json(evidence / 'evidence_manifest.json', {'files_sha256': {
            p.name: sha256(p) for p in evidence.iterdir() if p.is_file()}})
        return evidence

    def test_restore_reset_float32_summary_image_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evidence = self.fixture(root)
            report = restore_reset(evidence, root / 'result')
            self.assertTrue(report['matches_original_summary'])
            self.assertEqual(report['scope'], 'synthetic_reset_fixture_only')
            self.assertFalse(report['full_pipeline_restored'])
            self.assertEqual(report['scientific_status'], 'reset_failed')
            self.assertTrue((root / 'result/restored_state.png').is_file())
            self.assertEqual(read_json(root / 'result/camera.json')['projection'], 'orthographic_world_xy')
            with self.assertRaises(FileExistsError): restore_reset(evidence, root / 'result')

    def test_tampered_input_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evidence = self.fixture(root)
            with (evidence / 'initial_states.npz').open('ab') as stream: stream.write(b'tamper')
            with self.assertRaises(ValueError): restore_reset(evidence, root / 'result')
            self.assertFalse((root / 'result').exists())


if __name__ == '__main__': unittest.main()
