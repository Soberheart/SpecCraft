"""File-boundary tests, with tiny analytic grids and no simulator or model."""
import copy
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from level4.artifacts import write_json,read_json,sha256,save_packet,snapshot_runtime,verify_runtime
from level4.geometry import grid_coordinates
from level4.pipeline import prediction_decisions,freeze_selection,analyze,generate_actions,validate_pool_config
from level4.pipeline import audit_reset,verify_reset_gate
from test_level4_evaluate import synthetic_spec


def fixture(root):
    pool=root/"pool";pool.mkdir()
    q,n=grid_coordinates([[-6,6],[-4,4],[-2,2]],[1,1,1])
    np.savez(pool/"grid.npz",query=q,counts=n,spacing=np.ones(3),unit_to_meter=.001)
    np.savez(pool/"planner_input.npz",z_goal=np.array([[1.,0.]]))
    candidates=[{"candidate_id":f"candidate_{i:03d}","legal":True,"action_cost":cost}
                for i,cost in enumerate((.2,.1))]
    write_json(pool/"pool.json",{"candidates":candidates,"execution_required_s":[0,.48,.96,1.44,1.92],
        "input_sha256":{name:sha256(pool/name) for name in ("grid.npz","planner_input.npz")}})
    def occupied(x,y=2):
        return ((np.abs(q[:,0])<=x)&(np.abs(q[:,1])<=y)&(q[:,2]==0)).astype(int)
    for i,c in enumerate(candidates):
        directory=pool/"predictions"/c["candidate_id"];directory.mkdir(parents=True)
        save_packet(directory/"trajectory.npz",[0,.96,1.92],[occupied(4),occupied(3),occupied(3 if i==0 else 4)])
        np.savez(directory/"latents.npz",times_s=[0,.96,1.92],z=np.array([[[1.,1.]],[[1.,1.]],[[float(i),float(1-i)]]]))
        write_json(directory/"complete.json",{"pool_sha256":sha256(pool/"pool.json"),
            "packet_sha256":sha256(directory/"trajectory.npz"),"latent_score":float(i),
            "latents_sha256":sha256(directory/"latents.npz")})
    spec=synthetic_spec();spec["initial_reference"].update(u_mm=9,v_mm=5)
    spec["time"]["execution_required_s"]=[0,.48,.96,1.44,1.92]
    spec_path=root/"spec.json";write_json(spec_path,spec)
    return pool,spec,spec_path


def raw_fixture(root,name="reset",error=0):
    pool=root/name;pool.mkdir()
    q,n=grid_coordinates([[-6,6],[-4,4],[-2,2]],[1,1,1])
    np.savez(pool/"grid.npz",query=q,counts=n,spacing=np.ones(3),unit_to_meter=.001)
    write_json(pool/"runtime_snapshot/manifest.json",{"files_sha256":{"sim.py":"same-source"}})
    write_json(pool/"pool.json",{"input_sha256":{},"config_sha256":"config","source_metadata_sha256":"source",
        "source_observations_sha256":"observations","source_model_config_sha256":"model-config",
        "checkpoint_sha256":"checkpoint","source_file_sha256":{"sim.py":"same-source"},
        "execution_required_s":[0,1.92],"config":{"initial_reset_max_error_mm":.001,"auxiliary_s":1.984,
            "axes":{"u":[1,0,0],"v":[0,1,0]}},
        "candidates":[{"candidate_id":"candidate_000","parameters_mm_mm_deg":[0,0,0],"legal":True}]})
    out=pool/"executions/candidate_000";out.mkdir(parents=True)
    (out/"scenes/0000").mkdir(parents=True)
    raw=[{"t":t,"step":i*16,"obj":{"particles":{"pos":np.array([[0.,0,0],[2.,0,0]])}}}
         for i,t in enumerate([0,1.92])]
    with (out/"scenes/0000/log.pkl").open("wb") as stream:pickle.dump(raw,stream)
    write_json(out/"complete.json",{"pool_sha256":sha256(pool/"pool.json"),
        "log_sha256":sha256(out/"scenes/0000/log.pkl"),"initial_reset_max_abs_error_units":error})
    write_json(out/"exit.json",{"exit_code":0})
    return pool


class PipelineTests(unittest.TestCase):
    def test_development_config_and_invalid_prior(self):
        cfg=read_json(Path(__file__).resolve().parents[1]/"configs/level4/development_pool.json")
        validate_pool_config(cfg)
        for bounds in ([0,40,10],[-1,40,10],[float("nan"),40,10]):
            with self.assertRaises(ValueError):generate_actions(0,16,bounds)
        cfg["primary_s"]=1.984
        with self.assertRaises(ValueError):validate_pool_config(cfg)

    def test_development_summary_retains_dense_peak(self):
        from scripts.summarize_level4_development import summarize
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,_,_=fixture(root)
            meta=read_json(pool/"pool.json")
            meta["config"]={"primary_s":1.92,"axes":{"u":[1,0,0],"v":[0,1,0]}}
            (pool/"pool.json").write_text(json.dumps(meta),encoding="utf-8")
            for p in (pool/"predictions").glob("*/complete.json"):
                content=read_json(p);content["pool_sha256"]=sha256(pool/"pool.json")
                p.write_text(json.dumps(content),encoding="utf-8")
            with np.load(pool/"grid.npz") as packet:q=packet["query"]
            def cells(x,y):return ((abs(q[:,0])<=x)&(abs(q[:,1])<=y)&(q[:,2]==0)).astype(int)
            out=pool/"executions/candidate_000";out.mkdir(parents=True)
            save_packet(out/"trajectory.npz",[0,.48,.96,1.44,1.92],[cells(4,2),cells(3,3),cells(3,2),cells(3,2),cells(3,2)])
            write_json(out/"complete.json",{"pool_sha256":sha256(pool/"pool.json"),"packet_sha256":sha256(out/"trajectory.npz")})
            summarize(pool,root/"summary")
            result=read_json(root/"summary/calibration_evidence.json")
            self.assertAlmostEqual(result["reachable_pairs"][0]["dense_minus_common_peak"],.4)
            self.assertEqual(result["reachable_pairs"][1]["status"],"unknown_or_not_executed")

    def test_packet_does_not_wrap_invalid_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in (256,-1,1.5,float("nan")):
                with self.assertRaises(ValueError):save_packet(Path(directory)/"invalid.npz",[0],[[value]])
            with self.assertRaises(ValueError):save_packet(Path(directory)/"invalid.npz",[0],[[1]],[1])
    def test_future_execution_removal_replacement_reorder_do_not_change_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,spec,_=fixture(root)
            baseline=prediction_decisions(pool,spec,experiment=False)[1]
            future=pool/"executions"/"candidate_000";future.mkdir(parents=True)
            for data in ([{"t":0,"truth":"pass"},{"t":1.92,"truth":"fail"}],
                         [{"t":1.92,"truth":"pass"},{"t":0,"truth":"corrupted"}],[]):
                (future/"future.json").write_text(json.dumps(data),encoding="utf-8")
                # Corrupt recognizable truth packets must never be read by online choice.
                (future/"trajectory.npz").write_bytes(json.dumps(data).encode())
                (future/"complete.json").write_text(json.dumps({"truth":data}),encoding="utf-8")
                self.assertEqual(prediction_decisions(pool,spec,experiment=False)[1],baseline)
            (future/"future.json").unlink()
            self.assertEqual(prediction_decisions(pool,spec,experiment=False)[1],baseline)
            self.assertEqual(baseline["Oracle-Constraint"]["candidate_id"],"candidate_000")

    def test_reset_probe_requires_distinct_matching_sources_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=raw_fixture(root,"a");b=raw_fixture(root,"b")
            audit_reset([a,b],root/"reset.json")
            self.assertEqual(read_json(root/"reset.json")["status"],"passed")
            with self.assertRaises(ValueError):audit_reset([a,a],root/"duplicate.json")
            (b/"runtime_snapshot/manifest.json").write_text(json.dumps({"files_sha256":{"sim.py":"changed"}}))
            with self.assertRaises(ValueError):audit_reset([a,b],root/"changed.json")
            c=raw_fixture(root,"c",error=.01)
            with self.assertRaises(ValueError):audit_reset([a,c],root/"failed.json")
            self.assertEqual(read_json(root/"failed.json")["status"],"failed")

    def test_raw_representation_audit_analytic_support_and_truncation(self):
        from scripts.audit_level4_representation import audit
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool=raw_fixture(root)
            audit(pool,root/"audit",spacings=(1,.5),radii=(.5,))
            with (root/"audit/representation.csv").open() as stream:rows=list(csv.DictReader(stream))
            u=[r for r in rows if r["axis"]=="u"]
            self.assertTrue(all(float(r["raw_particle_span_mm"])==2 for r in u))
            self.assertTrue(all(float(r["analytic_ball_union_span_mm"])==3 for r in u))
            self.assertTrue(all(r["status"]=="measured" for r in rows))
            with self.assertRaises(ValueError):audit(pool,root/"bad_id",candidate_ids=["absent"])
            audit(pool,root/"outside",spacings=(1,),radii=(8,))
            with (root/"outside/representation.csv").open() as stream:rows=list(csv.DictReader(stream))
            self.assertTrue(all(r["status"]=="unknown" for r in rows))

    def test_reset_gate_rejects_changed_actual_inputs_source_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/"runtime";root.mkdir()
            subprocess.run(["git","init","-q",str(root)],check=True)
            runtime_file=root/"sim.py";runtime_file.write_bytes(b"same runtime\n")
            snapshot=base/"snapshot";snapshot_runtime(root,snapshot)
            source=base/"source";source.mkdir()
            checkpoint=base/"checkpoint.bin";checkpoint.write_bytes(b"frozen checkpoint")
            config=base/"config.json";config.write_bytes(b"{}")
            observations=source/"observations.npz";observations.write_bytes(b"initial observations")
            model_config=source/"model_config.yaml";model_config.write_bytes(b"frozen model config")
            write_json(source/"metadata.json",{"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint)})
            pools=[raw_fixture(base,name) for name in ("a","b")]
            for pool in pools:
                (pool/"runtime_snapshot/manifest.json").write_bytes((snapshot/"manifest.json").read_bytes())
                manifest=read_json(pool/"pool.json")
                manifest.update(config_sha256=sha256(config),source_metadata_sha256=sha256(source/"metadata.json"),
                    source_observations_sha256=sha256(observations),source_model_config_sha256=sha256(model_config),
                    checkpoint_sha256=sha256(checkpoint))
                (pool/"pool.json").write_text(json.dumps(manifest),encoding="utf-8")
                complete=pool/"executions/candidate_000/complete.json";data=read_json(complete)
                data["pool_sha256"]=sha256(pool/"pool.json");complete.write_text(json.dumps(data),encoding="utf-8")
            report=base/"reset.json";audit_reset(pools,report)
            self.assertEqual(verify_reset_gate(report,config,source,root=root)["status"],"passed")
            for path in (config,observations,model_config,checkpoint,runtime_file,source/"metadata.json"):
                original=path.read_bytes()
                # Whitespace keeps JSON parseable while still changing its identity.
                path.write_bytes(original+b" ")
                with self.assertRaises(ValueError):verify_reset_gate(report,config,source,root=root)
                path.write_bytes(original)
            data=read_json(report);data["source_signature"]["runtime_environment"]={"python":"changed"}
            report.write_text(json.dumps(data),encoding="utf-8")
            with self.assertRaises(ValueError):verify_reset_gate(report,config,source,root=root)
            data["status"]="failed";report.write_text(json.dumps(data),encoding="utf-8")
            with self.assertRaises(ValueError):verify_reset_gate(report,config,source,root=root)

    def test_prospective_analysis_rejects_unassociated_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,spec,path=fixture(root)
            spec["calibration"].update(status="frozen_development",evidence_sha256="a"*64)
            path.write_text(json.dumps(spec),encoding="utf-8")
            frozen=root/"selection.json";freeze_selection(pool,path,frozen)
            out=pool/"executions/candidate_000";out.mkdir(parents=True)
            (out/"trajectory.npz").write_bytes((pool/"predictions/candidate_000/trajectory.npz").read_bytes())
            write_json(out/"complete.json",{"pool_sha256":sha256(pool/"pool.json"),"packet_sha256":sha256(out/"trajectory.npz")})
            write_json(out/"exit.json",{"exit_code":0,"selection_sha256":None})
            with self.assertRaises(ValueError):analyze(pool,path,root/"rejected.json",frozen)
            (out/"exit.json").write_text(json.dumps({"exit_code":0,"selection_sha256":sha256(frozen)}))
            analyze(pool,path,root/"associated.json",frozen)
            self.assertEqual(read_json(root/"associated.json")["selection_timing"],"prospective")

    def test_pointgoal_target_latent_change_only_changes_pointgoal_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            pool,spec,_=fixture(Path(directory))
            before=prediction_decisions(pool,spec,experiment=False)[1]
            # A different authorized representative target defines a new input version.
            np.savez(pool/"planner_input.npz",z_goal=np.array([[0.,1.]]))
            meta=read_json(pool/"pool.json");meta["input_sha256"]["planner_input.npz"]=sha256(pool/"planner_input.npz")
            (pool/"pool.json").write_text(json.dumps(meta),encoding="utf-8")
            for p in (pool/"predictions").glob("*/complete.json"):
                meta=read_json(p);meta["pool_sha256"]=sha256(pool/"pool.json")
                p.write_text(json.dumps(meta),encoding="utf-8")
            after=prediction_decisions(pool,spec,experiment=False)[1]
            self.assertNotEqual(before["PointGoal"],after["PointGoal"])
            for method in ("Oracle-Constraint","Oracle-TerminalOnly"):
                self.assertEqual(before[method],after[method])

    def test_prospective_manifest_is_immutable_and_late_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,_,spec=fixture(root);out=root/"selection.json"
            freeze_selection(pool,spec,out,experiment=False)
            self.assertTrue(read_json(out)["synthetic_only"])
            with self.assertRaises(FileExistsError):freeze_selection(pool,spec,out,experiment=False)
            (pool/"executions/candidate_000").mkdir(parents=True)
            with self.assertRaises(ValueError):freeze_selection(pool,spec,root/"late.json",experiment=False)

    def test_missing_actual_candidates_stay_in_denominator_and_cannot_change_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,spec,path=fixture(root)
            spec["calibration"].update(status="frozen_development",evidence_sha256="a"*64)
            path.write_text(json.dumps(spec),encoding="utf-8")
            analyze(pool,path,root/"analysis.json")
            report=read_json(root/"analysis.json")
            self.assertEqual(report["pool_size"],2)
            self.assertEqual(report["confusion"]["matrix"]["satisfied"]["unknown"],1)
            self.assertEqual(report["selections"]["Oracle-Constraint"]["candidate_id"],"candidate_000")
            self.assertEqual(report["selection_timing"],"exploratory_post_execution_not_formal")

    def test_cli_evaluate_and_uncalibrated_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,_,spec=fixture(root);out=root/"report.json"
            cli=Path(__file__).resolve().parents[1]/"scripts/level4.py"
            run=subprocess.run([sys.executable,str(cli),"evaluate","--spec",str(spec),"--pool",str(pool),
                "--trajectory",str(pool/"predictions/candidate_000/trajectory.npz"),"--output",str(out),
                "--prediction","--synthetic-only"],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(read_json(out)["status"],"satisfied")
            run=subprocess.run([sys.executable,str(cli),"validate-spec","--spec",str(spec),"--experiment"],
                               capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)

    def test_snapshot_includes_untracked_and_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"repo";root.mkdir()
            subprocess.run(["git","init","-q",str(root)],check=True)
            (root/"new.py").write_text("VALUE = 1\n",encoding="utf-8")
            snapshot_runtime(root,Path(directory)/"snapshot")
            self.assertEqual((Path(directory)/"snapshot/files/new.py").read_bytes(),(root/"new.py").read_bytes())
            verify_runtime(root,Path(directory)/"snapshot")
            (root/"new.py").write_text("VALUE = 2\n",encoding="utf-8")
            with self.assertRaises(ValueError):verify_runtime(root,Path(directory)/"snapshot")


if __name__=="__main__":unittest.main()
