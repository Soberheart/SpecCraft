"""L4-01 A7: tool offset option; zero offset leaves the scene bit-identical."""
import copy
import json
import unittest

from level4.tool_offset import apply_tool_offset

SCENE = {'actions': {'grasp': {'type': 'grasp', 'to_pos': [0.005029208843735732, -0.0052841945316520755, 0.0],
                               'to_quat': [0.9499605297867098, 0.0, 0.0, 0.3123699598990818],
                               'from_offset': [0.0, 0.0, 0.0], 'init_d': 0.5, 'close_d': 0.005}},
         'log': {'base_dir': '/x'}, 'sim': {'step_dt': 0.002}}


class ToolOffsetTests(unittest.TestCase):
    def test_zero_offset_is_bitwise_identical_and_not_aliased(self):
        before = json.dumps(SCENE, sort_keys=True)
        out = apply_tool_offset(SCENE, (0.0, 0.0), 0.2)
        self.assertEqual(json.dumps(out, sort_keys=True), before)
        self.assertEqual(repr(out['actions']['grasp']['to_pos']), repr(SCENE['actions']['grasp']['to_pos']))
        out['actions']['grasp']['to_pos'][0] = 1.0
        self.assertEqual(json.dumps(SCENE, sort_keys=True), before)

    def test_offset_shifts_only_the_anchor_by_mm_over_200(self):
        out = apply_tool_offset(SCENE, (0.02, 0.0), 0.2)
        self.assertAlmostEqual(out['actions']['grasp']['to_pos'][0] - SCENE['actions']['grasp']['to_pos'][0], 1e-4, places=15)
        self.assertEqual(out['actions']['grasp']['to_pos'][1:], SCENE['actions']['grasp']['to_pos'][1:])
        rest_in, rest_out = copy.deepcopy(SCENE), copy.deepcopy(out)
        rest_in['actions']['grasp'].pop('to_pos'); rest_out['actions']['grasp'].pop('to_pos')
        self.assertEqual(rest_in, rest_out)
        y = apply_tool_offset(SCENE, (0.0, 0.1), 0.2)
        self.assertAlmostEqual(y['actions']['grasp']['to_pos'][1] - SCENE['actions']['grasp']['to_pos'][1], 5e-4, places=15)

    def test_invalid_offsets_rejected(self):
        for bad in ((0.0,), (float('nan'), 0.0), (2.0, 0.0), ('0', 0.0)):
            with self.assertRaises(ValueError):
                apply_tool_offset(SCENE, bad, 0.2)
        with self.assertRaises(ValueError):
            apply_tool_offset(SCENE, (0.0, 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()
