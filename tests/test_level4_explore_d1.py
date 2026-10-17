"""Post-hoc exploratory D1 (E1/E2) on synthetic data; the pre-registered report stays untouched."""
import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np

from level4.artifacts import sha256
from scripts import analyze_level4_batch as az
from scripts import explore_level4_d1_lateral as ex
from tests.test_level4_analyze_batch import layout


def small_grid():
    axes = [np.arange(6) * 1.0, np.arange(6) * 1.0, np.arange(4) * 1.0]
    query = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    return {'query': query, 'counts': np.array([6, 6, 4]), 'spacing': np.ones(3), 'unit_to_meter': np.asarray(0.001)}


class ExploratoryD1Tests(unittest.TestCase):
    def test_z_face_contact_rules(self):
        grid = small_grid()
        k0 = np.zeros((6, 6, 4), np.uint8); k0[2:4, 2:4, 1] = 1
        k1 = k0.copy(); k1[1:5, 2:4, 0] = 1  # wider, and leaking below the k0 floor onto the z-min face
        classes = np.stack([k0.reshape(-1), k1.reshape(-1)])
        ok, out = np.ones(2, bool), np.zeros(2, bool)
        e1 = ex.spans(grid, classes, ok, out, ex.RULES[0])
        e2 = ex.spans(grid, classes, ok, out, ex.RULES[1])
        self.assertEqual(e1['k0_floor_index'], 1)
        self.assertEqual(e1['u'], [2.0, 4.0])   # sub-floor cells count laterally under E1
        self.assertEqual(e2['u'], [2.0, 2.0])   # and are removed under E2
        self.assertAlmostEqual(e1['below_floor_share'][1], 8 / 12)
        k2 = k0.copy(); k2[0, 2, 1] = 1        # x-face contact stays unknown under both rules
        classes = np.stack([k0.reshape(-1), k2.reshape(-1)])
        self.assertIsNone(ex.spans(grid, classes, ok, out, ex.RULES[0])['u'][1])
        self.assertIsNone(ex.spans(grid, classes, ok, out, ex.RULES[1])['u'][1])

    def test_end_to_end_writes_new_directory_only(self):
        with tempfile.TemporaryDirectory() as d:
            project = layout(Path(d))
            analysis = project / az.BATCH_DIRNAME / 'C-analysis_20261017T000000Z'
            with unittest.mock.patch.object(az, 'TOP_K', 2), unittest.mock.patch.object(az, 'T1_GRID', (0.8,)), \
                    unittest.mock.patch.object(az, 'T2_GRID', (0.1,)), unittest.mock.patch.object(az, 'TOLERANCES', (0.2,)):
                az.run(analysis / 'data', project, workers=1)
                before = sha256(analysis / 'data/report/phase_c_report.json')
                self.assertEqual(ex.main(['--project', str(project)]), 0)
            self.assertEqual(sha256(analysis / 'data/report/phase_c_report.json'), before)
            out = next(analysis.glob('exploratory_d1_*'))
            result = json.loads((out / 'exploratory_d1.json').read_text(encoding='utf-8'))
            self.assertEqual(result['label'], ex.LABEL)
            self.assertEqual(len(result['D1']), 16)
            self.assertTrue((out / 'summary.txt').read_text(encoding='utf-8').startswith(ex.LABEL))


if __name__ == '__main__':
    unittest.main()
