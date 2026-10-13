import unittest
import numpy as np
from level4.selection import select,constraint_key,confusion
from level4.pipeline import generate_actions


def record(cid,status,cost=0,violation=0,scale=1,latent=0,legal=True):
    return {"candidate_id":cid,"legal":legal,"action_cost":cost,"latent_score":latent,
        "prediction":{"source":"prediction","status":status,"terminal":{"violation":violation,"normalization_scale":scale},
                      "process":{"samples":[]}},"execution":{"status":"satisfied"}}


class SelectionTests(unittest.TestCase):
    def test_soft_cannot_rescue_hard_and_ties_are_deterministic(self):
        rows=[record("bad","violated",0,.1,latent=1),record("good2","satisfied",5),record("good1","satisfied",5)]
        self.assertEqual(select(rows)["candidate_id"],"good1")
        self.assertEqual(select(rows,"latent")["candidate_id"],"bad")

    def test_unknown_illegal_and_no_feasible(self):
        rows=[record("unknown","unknown"),record("illegal","satisfied",legal=False),record("bad","violated")]
        self.assertEqual(select(rows)["status"],"no_feasible_candidate")
        self.assertIsNone(select(rows)["candidate_id"])

    def test_execution_oracle_cannot_enter_online_selection(self):
        r=record("x","satisfied");r["prediction"]["source"]="independent_execution"
        with self.assertRaises(ValueError):select([r])
        with self.assertRaises(ValueError):select([r],"latent")

    def test_duplicate_ids_and_nonboolean_legality_rejected(self):
        with self.assertRaises(ValueError):select([record("a","satisfied"),record("a","satisfied")])
        with self.assertRaises(ValueError):select([record("a","satisfied",legal="yes")])

    def test_mixed_unit_search_guidance_normalized(self):
        mm=record("mm","violated",violation=5,scale=10)
        ratio=record("ratio","violated",violation=.06,scale=.1)
        self.assertLess(constraint_key(mm),constraint_key(ratio))
        bad=record("bad","violated",scale=0)
        with self.assertRaises(ValueError):constraint_key(bad)

    def test_confusion_unknown_separate_and_denominators(self):
        rows=[record("a","satisfied"),record("b","satisfied"),record("c","violated"),record("d","unknown")]
        rows[1]["execution"]["status"]="violated"
        r=confusion(rows)
        self.assertEqual(r["predicted_feasible_actual_not_satisfied"],.5)
        self.assertAlmostEqual(r["predicted_infeasible_actual_satisfied"],1/3)
        self.assertEqual(r["matrix"]["unknown"]["satisfied"],1)

    def test_pool_reproducibility_nominal_and_legality(self):
        a=generate_actions(7,16);b=generate_actions(7,16)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[0],0)
        self.assertTrue((np.abs(a)<=[40,40,10]).all())
        self.assertEqual(len(np.unique(a,axis=0)),16)

    def test_false_acceptance_and_execution_unknown_are_distinct(self):
        rows=[record("a","satisfied"),record("b","satisfied"),record("c","satisfied")]
        rows[1]["execution"]["status"]="violated";rows[2]["execution"]["status"]="unknown"
        result=confusion(rows)
        self.assertAlmostEqual(result["observed_false_acceptance"],1/3)
        self.assertAlmostEqual(result["predicted_satisfied_execution_unknown"],1/3)
        self.assertIsNone(confusion([])["observed_false_acceptance"])


if __name__=="__main__":unittest.main()
