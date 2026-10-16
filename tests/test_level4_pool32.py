"""L4-01 A6 pool extension rules and the frame0 finger clearance screen (synthetic)."""
import unittest

import numpy as np

from level4.finger_clearance import finger_geometry, initial_clearance
from level4.pipeline import generate_actions
from scripts.build_level4_pool32 import BOUNDS, D2_EDGE

SCENE = {'actions': {'grasp': {'to_pos': [0.0, 0.0, 0.0], 'to_quat': [1.0, 0.0, 0.0, 0.0], 'init_d': 0.5}},
         'ee': {'entities': {name: {'geom': {'scale': [0.05, 0.05, 1.0], 'offset_pos': [sign * 0.025, 0.0, 0.125],
                                             'offset_euler': [0, 0, 0]}}
                             for name, sign in (('finger_left', -1), ('finger_right', 1))}}}


class Pool32Tests(unittest.TestCase):
    def test_seed0_prefix_rule_is_deterministic(self):
        frozen = generate_actions(0, 16)
        extended = generate_actions(0, 32)
        # rejection resampling depends on count, so rows 5 and 14 differ -> seed1 group rule
        self.assertEqual([i for i in range(16) if not np.array_equal(frozen[i], extended[i])], [5, 14])
        second = generate_actions(1, 17)[1:]
        self.assertEqual(second.shape, (16, 3))
        self.assertFalse(any((frozen == row).all(axis=1).any() for row in second))

    def test_d2_group_is_inside_contract_bounds(self):
        for action, reason in D2_EDGE.values():
            self.assertTrue((np.abs(action) <= BOUNDS).all(), reason)

    def test_geometry_from_scene(self):
        g = finger_geometry(SCENE)
        self.assertAlmostEqual(g['open_mm'], 100.0)
        self.assertAlmostEqual(g['yaw_deg'], 0.0)
        self.assertEqual(g['fingers'][0]['half_extent_mm'], [5.0, 5.0, 25.0])
        self.assertEqual(g['fingers'][1]['offset_mm'], [5.0, 0.0, 25.0])

    def test_clearance_hand_examples(self):
        g = finger_geometry(SCENE)
        # right finger box occupies x in [50, 60], y in [-5, 5], z in [0, 50]
        points = np.array([[40.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
        c = initial_clearance(points, g, [0, 0, 0])
        self.assertAlmostEqual(c['fingers']['finger_right']['min_clearance_mm'], 10.0)
        self.assertFalse(c['interpenetrating_at_frame0'])
        moved = initial_clearance(points, g, [-12.0, 0.0, 0.0])  # right finger now x in [38, 48]
        self.assertAlmostEqual(moved['fingers']['finger_right']['min_clearance_mm'], -2.0)
        self.assertTrue(moved['interpenetrating_at_frame0'])
        rotated = initial_clearance(np.array([[0.0, 55.0, 10.0]]), g, [0, 0, 90.0])  # right finger -> +y
        self.assertAlmostEqual(rotated['fingers']['finger_right']['min_clearance_mm'], -5.0)
        with self.assertRaises(ValueError):
            initial_clearance(np.zeros((2, 2)), g, [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
