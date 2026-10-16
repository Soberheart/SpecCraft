"""L4-01 A8: D2 measures on synthetic shapes with hand-derived expectations."""
import unittest

import numpy as np

from level4.material_split import (component_sensitivity, component_summary, connected_components,
                                   cross_section_profile, d2_measures, nearest_neighbour_spacing, neck,
                                   percolation_radius)


def box(x0, x1, y0, y1, z0=0, z1=4, step=1.0):
    """Lattice of points (mm) on [x0,x1]x[y0,y1]x[z0,z1] inclusive, spacing step."""
    xs, ys, zs = (np.arange(a, b + step / 2, step) for a, b in ((x0, x1), (y0, y1), (z0, z1)))
    return np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)


def unique(points):
    return np.unique(np.round(points, 9), axis=0)


class MaterialSplitTests(unittest.TestCase):
    def test_lattice_spacing(self):
        row = nearest_neighbour_spacing(box(0, 9, 0, 9))
        self.assertAlmostEqual(row['median_mm'], 1.0)
        self.assertAlmostEqual(row['max_mm'], 1.0)

    def test_two_separated_equal_blocks(self):
        points = np.vstack([box(0, 9, 0, 9), box(16, 25, 0, 9)])  # 7 mm gap
        result = component_sensitivity(points, 1.0)
        self.assertEqual(result['status'], 'stable')
        self.assertEqual(result['component_counts'], [2])
        row = result['rows'][2]
        self.assertEqual(row['particle_counts'], [500, 500])
        self.assertAlmostEqual(row['two_largest_balance'], 1.0)
        n = neck(points, slice_mm=1.0)
        self.assertEqual(n['status'], 'gap')
        self.assertEqual(n['neck_width_mm'], 0.0)
        self.assertTrue(10 < n['neck_position_mm'] < 16)

    def test_unequal_blocks_volume_share_and_mass_weights(self):
        points = np.vstack([box(0, 9, 0, 9), box(16, 20, 0, 9)])  # 500 and 250 points
        row = component_summary(points, 1.5)
        self.assertEqual(row['particle_counts'], [500, 250])
        self.assertAlmostEqual(row['volume_share'][0], 2 / 3)
        self.assertAlmostEqual(row['two_largest_balance'], 0.5)
        weights = np.r_[np.ones(500), np.full(250, 2.0)]  # equal mass when weighted
        weighted = component_summary(points, 1.5, weights)
        self.assertAlmostEqual(weighted['two_largest_balance'], 1.0)

    def test_dumbbell_waist_is_one_component_with_measured_neck(self):
        bridge = box(10, 19, 4, 6)  # v-extent of centres = 2 mm
        points = unique(np.vstack([box(0, 9, 0, 9), bridge, box(20, 29, 0, 9)]))
        self.assertEqual(component_sensitivity(points, 1.0)['component_counts'], [1])
        for h in (1.0, 2.0, 4.0):
            n = neck(points, slice_mm=h)
            self.assertEqual(n['status'], 'waist')
            self.assertAlmostEqual(n['neck_width_mm'], 2.0)
            self.assertTrue(9.5 < n['neck_position_mm'] < 19.5, n)
            self.assertAlmostEqual(n['max_width_mm'], 9.0)

    def test_single_block_has_no_waist(self):
        n = neck(box(0, 19, 0, 9), slice_mm=2.0)
        self.assertEqual(n['status'], 'no_waist')
        self.assertIsNone(n['neck_width_mm'])
        self.assertEqual(component_summary(box(0, 19, 0, 9), 1.5)['components'], 1)

    def test_threshold_sensitivity_is_reported_not_hidden(self):
        points = np.vstack([box(0, 9, 0, 9), box(11.5, 20.5, 0, 9)])  # 2.5 mm gap
        result = component_sensitivity(points, 1.0, multipliers=(1.5, 2.0, 3.0))
        self.assertEqual([r['components'] for r in result['rows']], [2, 2, 1])
        self.assertEqual(result['significant_status'], 'unstable')
        self.assertEqual(result['status'], 'unstable')

    def test_rigid_motion_invariance_and_label_order(self):
        rng = np.random.default_rng(0)
        points = np.vstack([box(0, 4, 0, 4), box(10, 19, 0, 9)])
        labels = connected_components(points, 1.5)
        self.assertEqual(labels[0], 1)  # smaller block ranked second
        self.assertEqual(labels[-1], 0)
        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        moved = points @ rot.T + [3.0, -7.0, 1.0]
        order = rng.permutation(len(points))
        a = component_summary(points, 1.5); b = component_summary(moved[order], 1.5)
        self.assertEqual(a['particle_counts'], b['particle_counts'])

    def test_profile_uses_fixed_axes_and_rejects_bad_input(self):
        points = box(0, 9, 0, 3)
        p = cross_section_profile(points, (0, 1, 0), (1, 0, 0), 1.0)
        self.assertEqual(set(p['widths_mm']), {9.0})
        with self.assertRaises(ValueError):
            cross_section_profile(points, (1, 0, 0), (1, 1, 0), 1.0)
        for bad in (np.zeros((1, 3)), np.zeros((4, 2)), np.full((4, 3), np.nan)):
            with self.assertRaises(ValueError):
                component_summary(bad, 1.0)
        with self.assertRaises(ValueError):
            component_summary(points, 0.0)
        with self.assertRaises(ValueError):
            component_sensitivity(points, -1.0)

    def test_combined_report_contains_every_setting(self):
        result = d2_measures(np.vstack([box(0, 9, 0, 9), box(16, 25, 0, 9)]), 1.0)
        self.assertEqual(len(result['components']['rows']), 4)
        self.assertEqual(set(result['neck']), {'3.0', '4.0', '6.0'})
        self.assertIn('not task thresholds', result['scope'])


    def test_percolation_radius_of_irregular_cloud(self):
        # lattice with one point 2.5 mm away from its nearest neighbour
        points = np.vstack([box(0, 9, 0, 9), [[11.5, 0.0, 0.0]]])
        r0 = percolation_radius(points)
        self.assertAlmostEqual(r0, 2.5, delta=0.006)
        self.assertGreaterEqual(r0, 2.5)
        self.assertEqual(component_summary(points, r0)['components'], 1)

    def test_stray_particle_is_debris_not_a_break(self):
        points = np.vstack([box(0, 9, 0, 9), [[30.0, 0.0, 0.0]]])  # 500 + 1 isolated particle
        row = component_summary(points, 1.5)
        self.assertEqual(row['components'], 2)
        # 1/501 = 0.2% of the mass: significant only at the 0.1% setting
        self.assertEqual(row['significant_components'], {'0.001': 2, '0.005': 1, '0.02': 1})
        self.assertEqual(row['debris_particles']['0.005'], 1)
        halves = np.vstack([box(0, 9, 0, 9), box(16, 25, 0, 9)])
        self.assertEqual(component_summary(halves, 1.5)['significant_components']['0.02'], 2)


if __name__ == '__main__':
    unittest.main()
