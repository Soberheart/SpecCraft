import copy
import unittest
from level4.spec import validate_spec
from level4.evaluate import evaluate
from test_level4_geometry import box


def synthetic_spec(process=True):
    return {"schema_version":"1.0.0","task_id":"synthetic_B" if process else "synthetic_A",
        "description":"Analytic unit test; thresholds do not describe material experiments",
        "object":"whole_material","semantics":"world_axis_extent",
        "reference_frame":{"kind":"initial_fixed_world","unit":"mm","origin_mm":[0,0,0],"u":[1,0,0],"v":[0,1,0]},
        "initial_reference":{"source":"privileged_full_material_initial_state","u_mm":100,"v_mm":60},
        "measurement":{"representation":"voxel_union","operator":"extrema","minimum_reference_span_mm":1,
                       "tolerance_ratio":0,"prediction_margin_ratio":0},
        "time":{"primary_s":1.92,"auxiliary_s":1.984,"required_s":[0,.96,1.92],"match_tolerance_s":1e-8},
        "terminal_hard":{"id":"narrow_u","rho":.2,"relation":"le","hard":True},
        "process_hard":[{"id":"protect_v","eta":.1,"relation":"le","hard":True}] if process else [],
        "soft_preferences":["minimum_normalized_action_l2"],
        "calibration":{"status":"synthetic_only","evidence_sha256":"analytic_fixture","split_id":"unit_tests"}}


class EvaluatorTests(unittest.TestCase):
    def test_positive_and_negative_terminal(self):
        s=synthetic_spec(False)
        for width,expected in ((75,"satisfied"),(85,"violated"),(80,"satisfied")):
            r=evaluate(s,[(0,box()),(.96,box()),(1.92,box((width,60,40)))])
            self.assertEqual(r["status"],expected)

    def test_process_boundary_and_first_violation(self):
        s=synthetic_spec()
        result=evaluate(s,[(0,box()),(.96,box((90,66,40))),(1.92,box((80,60,40)))])
        self.assertTrue(result["strict_success"])
        result=evaluate(s,[(0,box()),(.96,box((90,72,40))),(1.92,box((80,60,40)))])
        self.assertFalse(result["strict_success"])
        self.assertEqual(result["process"]["first_detected_violation_s"],.96)
        self.assertAlmostEqual(result["process"]["max_violation"],.1)
        self.assertEqual(result["terminal"]["status"],"satisfied")

    def test_missing_frame_unknown_and_known_violation_survives(self):
        s=synthetic_spec()
        r=evaluate(s,[(0,box()),(1.92,box((80,60,40)))])
        self.assertEqual(r["status"],"unknown")
        self.assertEqual(r["process"]["missing_or_unknown_times_s"],[.96])
        r=evaluate(s,[(0,box()),(1.92,box((95,60,40)))])
        self.assertEqual(r["status"],"violated")

    def test_auxiliary_cannot_rescue_primary(self):
        r=evaluate(synthetic_spec(False),[(0,box()),(1.92,box((90,60,40))),(1.984,box((60,60,40)))])
        self.assertEqual(r["status"],"violated")
        self.assertEqual(r["auxiliary"]["terminal"]["status"],"satisfied")

    def test_dense_actual_detects_peak_missed_by_common_schedule(self):
        s=synthetic_spec();s["time"]["execution_required_s"]=[0,.48,.96,1.44,1.92]
        traj=[(0,box()),(.48,box((90,72,40))),(.96,box()),(1.44,box()),(1.92,box((75,60,40)))]
        self.assertEqual(evaluate(s,traj,common_schedule=True)["status"],"satisfied")
        dense=evaluate(s,traj)
        self.assertEqual(dense["status"],"violated")
        self.assertEqual(dense["process"]["first_detected_violation_s"],.48)
        self.assertEqual(evaluate(s,[x for x in traj if x[0]!=.48])["status"],"unknown")

    def test_invalid_schedule_and_spec_structure_rejected(self):
        for times in ([0,1.92,.96],[0,.96,.96,1.92],[0,float("inf"),1.92]):
            s=synthetic_spec();s["time"]["execution_required_s"]=times
            with self.assertRaises(ValueError):validate_spec(s)
        with self.assertRaises(ValueError):validate_spec({"schema_version":"1.0.0","task_id":"missing"})

    def test_no_interpolation_of_physical_timestamps(self):
        r=evaluate(synthetic_spec(),[(0,box()),(.95,box()),(1.92,box((75,60,40)))])
        self.assertEqual(r["status"],"unknown")
        with self.assertRaises(ValueError):
            evaluate(synthetic_spec(),[(0,box()),(0,box())])

    def test_zero_reference_and_nonfinite_threshold_rejected(self):
        for value in (0,1):
            s=synthetic_spec();s["initial_reference"]["u_mm"]=value
            r=evaluate(s,[(0,box()),(1.92,box())])
            self.assertEqual(r["status"],"unknown")
        for value in (None,float("nan"),True):
            s=synthetic_spec();s["terminal_hard"]["rho"]=value
            with self.assertRaises(ValueError):validate_spec(s)

    def test_prediction_margin_tightens_only_prediction(self):
        s=synthetic_spec(False);s["measurement"]["prediction_margin_ratio"]=.02
        traj=[(0,box()),(1.92,box((79,60,40)))]
        self.assertEqual(evaluate(s,traj)["status"],"satisfied")
        self.assertEqual(evaluate(s,traj,prediction=True)["status"],"violated")

    def test_initial_mismatch_and_truncation_unknown(self):
        r=evaluate(synthetic_spec(False),[(0,box((90,60,40))),(1.92,box((70,60,40)))])
        self.assertEqual(r["status"],"unknown")
        g=box((70,60,40));g.truncated=True
        self.assertEqual(evaluate(synthetic_spec(False),[(0,box()),(1.92,g)])["status"],"unknown")

    def test_bad_frame_semantics_and_synthetic_experiment_blocked(self):
        for mutate in (lambda s:s.update(semantics="material_deformation"),
                       lambda s:s["reference_frame"].update(v=[1,0,0]),
                       lambda s:s["calibration"].update(status="uncalibrated")):
            s=synthetic_spec();mutate(s)
            with self.assertRaises(ValueError):validate_spec(s)
        with self.assertRaises(ValueError):validate_spec(synthetic_spec(),experiment=True)


if __name__=="__main__":unittest.main()
