import unittest
from level4.statistics import paired_scene_bootstrap


class StatisticsTests(unittest.TestCase):
    def rows(self):
        return [{"initial_state_group":scene,"task_family_id":"B","search_seed":seed,
                 "method":method,"strict_success":method=="constraint"}
                for scene in ("one","two") for seed in range(5) for method in ("constraint","latent")]

    def test_seeds_do_not_inflate_scene_count(self):
        result=paired_scene_bootstrap(self.rows(),"constraint","latent")
        self.assertEqual(result["independent_initial_states"],2)
        self.assertEqual(result["paired_mean_success_difference"],1)
        self.assertEqual(result["ci95"],[1,1])

    def test_unpaired_seed_rejected(self):
        with self.assertRaises(ValueError):paired_scene_bootstrap(self.rows()[:-1],"constraint","latent")

    def test_single_scene_has_no_confidence_interval(self):
        rows=[r for r in self.rows() if r["initial_state_group"]=="one"]
        self.assertIsNone(paired_scene_bootstrap(rows,"constraint","latent")["ci95"])
