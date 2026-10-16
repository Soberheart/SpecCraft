"""L4-01 Phase B prediction entry: request scope, schedule, rollout order, tool sampler extraction."""
from pathlib import Path
import unittest

import numpy as np

from level4.tool_observation import FUNCTIONS, compare_clouds, extract_process_functions
from scripts import predict_level4_batch as predict

ROOT = Path(__file__).resolve().parents[1]


def request(**override):
    value = {'kind': predict.KIND, 'candidate_pool_released': False, 'new_simulations_max': 0,
             'variants': list(predict.VARIANTS), 'chunk_size': 4096,
             'candidates': [{'candidate_id': 'candidate_000', 'action_mm_mm_deg': [0, 0, 0]},
                            {'candidate_id': 'candidate_002', 'action_mm_mm_deg': [1, 2, 3]}],
             'new_Dyn_rollouts': 4}
    value.update(override)
    return value


class BatchPredictionTests(unittest.TestCase):
    def test_request_scope(self):
        predict.validate_request(request())
        for key, value in (('new_simulations_max', 1), ('candidate_pool_released', True),
                           ('variants', ['v3_nominal']), ('new_Dyn_rollouts', 3), ('chunk_size', 0)):
            with self.assertRaises(ValueError):
                predict.validate_request(request(**{key: value}))
        with self.assertRaises(ValueError):
            predict.validate_request(request(candidates=[{'candidate_id': 'a', 'action_mm_mm_deg': [0, 0, 0]}] * 2,
                                             new_Dyn_rollouts=4))

    def test_pool_schedule_maps_to_logged_frames(self):
        times = np.arange(13) * 0.16
        self.assertEqual(predict.schedule_frames(times), list(range(0, 61, 5)))
        with self.assertRaises(ValueError):
            predict.schedule_frames(times + 0.01)

    def test_rollout_is_autoregressive_and_keeps_privileged_initial_geometry(self):
        seen = []
        def step(i, latent):
            seen.append((i, latent))
            return latent + 1
        latents, classes = predict.rollout(step, lambda z: np.full(3, z), 10, np.zeros(3, dtype=np.uint8), np.zeros(13))
        self.assertEqual([i for i, _ in seen], list(range(12)))
        self.assertEqual(latents, list(range(10, 23)))
        self.assertEqual(classes[0].tolist(), [0, 0, 0])
        self.assertEqual(classes[-1].tolist(), [22, 22, 22])

    def test_training_sampler_functions_are_extracted_from_process_source(self):
        functions, hashes = extract_process_functions(ROOT / 'sim/process.py', object())
        self.assertEqual(set(functions), set(FUNCTIONS))
        self.assertTrue(all(len(h) == 64 for h in hashes.values()))

    def test_cloud_comparison_units(self):
        a = np.zeros((2, 4, 3)); b = a.copy(); b[1, :, 0] += 0.005  # 1 mm
        result = compare_clouds(a, b)
        self.assertAlmostEqual(result['max_bbox_mm'], 1.0)
        self.assertAlmostEqual(result['max_centroid_mm'], 1.0)
        with self.assertRaises(ValueError):
            compare_clouds(a, b[:1])


if __name__ == '__main__':
    unittest.main()
