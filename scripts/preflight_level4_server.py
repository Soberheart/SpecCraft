"""Read-only CPU input/resource audit; never submits a job or initializes CUDA."""
import argparse
import ast
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dyn-artifacts", type=Path, default=ROOT / (
        "records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        (ROOT / "records").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="level4_server_preflight_", dir=ROOT / "records"))
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    source = args.dyn_artifacts.resolve()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(ROOT), "output": str(output), "dyn_artifacts": str(source),
              "evidence_kind": "actual_cpu_preflight_not_gpu_experiment",
              "errors": [], "commands": {}, "checks": {}}
    # Preserve the exact collector separately, including when the audit fails.
    shutil.copyfile(__file__, output / "collector.py")
    report["collector_sha256"] = sha256(output / "collector.py")
    print("PREFLIGHT_DIR=" + str(output), flush=True)

    def section(name, operation):
        try:
            report["checks"][name] = operation()
            print(name + ": OK", flush=True)
        except Exception as exc:
            detail = {"section": name, "type": type(exc).__name__, "message": str(exc)}
            report["errors"].append(detail)
            (output / (name + "_error.txt")).write_text(traceback.format_exc(), encoding="utf-8")
            print(name + ": FAILED: " + str(exc), flush=True)

    def command(name, argv):
        try:
            run = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", timeout=30)
            content, code = run.stdout, run.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            content, code = str(exc) + "\n", None
        (output / (name + ".txt")).write_text(content, encoding="utf-8")
        report["commands"][name] = {"argv": argv, "exit_code": code,
                                     "output_sha256": sha256(output / (name + ".txt"))}
        if code != 0:
            report["errors"].append({"section": name, "message": "command failed", "exit_code": code})
        return content

    command("git_status", ["git", "status", "--short"])
    command("commit", ["git", "rev-parse", "HEAD"])
    command("environment", [sys.executable, "-m", "pip", "freeze"])
    queue = command("queue", ["qstat", "-Qf", "normal"])
    command("server", ["qstat", "-Bf"])
    jobs = command("user_jobs", ["qstat", "-u", getpass.getuser()])

    def packages():
        required = ("numpy", "torch", "taichi", "scipy", "omegaconf", "hydra-core", "h5py",
                    "Pillow", "scikit-image", "trimesh", "imageio")
        found = {}
        missing = []
        for name in required:
            try:
                found[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
        report["package_versions"] = found
        report["python"] = sys.version
        require(not missing, "Missing packages: " + ", ".join(missing))
        return found

    section("packages", packages)
    metadata = {}
    file_hashes = {}

    def inputs():
        metadata.update(json.loads((source / "metadata.json").read_text(encoding="utf-8")))
        require(metadata["kind"] == "recorded_action_interface_diagnostic", "Unexpected Dyn packet kind")
        scene = Path(metadata["scene_dir"])
        pairs = [(Path(metadata["checkpoint"]), metadata["checkpoint_sha256"]),
                 (scene / "config.yaml", metadata["scene_config_sha256"]),
                 (scene / "log.pkl", metadata["source_log_sha256"]),
                 (Path(metadata["processed"]), metadata["processed_sha256"])]
        pairs += [(source / name, None) for name in ("metadata.json", "observations.npz", "model_config.yaml")]
        require(bool(metadata.get("source_file_sha256")), "Missing historical runtime source hashes; review required")
        pairs += [(ROOT / name, digest) for name, digest in metadata["source_file_sha256"].items()]
        mismatches = []
        for path, expected in pairs:
            if not path.is_file():
                mismatches.append("Missing file: " + str(path))
                continue
            actual = sha256(path)
            file_hashes[str(path)] = {"actual": actual, "expected": expected,
                                     "matches": None if expected is None else actual == expected}
            if expected is not None and actual != expected:
                mismatches.append("Hash mismatch: " + str(path))
        report["files_sha256"] = file_hashes
        require(not mismatches, "; ".join(mismatches))
        return {"checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                "scene_dir": str(scene), "historical_source_files": len(metadata["source_file_sha256"])}

    section("inputs", inputs)

    def geometry_and_schedule():
        import pickle
        import numpy as np
        from omegaconf import OmegaConf
        scene = Path(metadata["scene_dir"])
        cfg = OmegaConf.load(scene / "config.yaml")
        OmegaConf.resolve(cfg)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        (output / "scene_config_resolved.json").write_text(
            json.dumps(resolved, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        model = OmegaConf.load(source / "model_config.yaml")
        OmegaConf.resolve(model)
        scale = float(metadata["unit_to_meter"])
        require(np.isclose(scale, .2, rtol=0, atol=1e-12), "Unexpected simulator scale")
        require(np.isclose(float(model.augmentation.render.unit_to_meter), scale, rtol=0, atol=1e-12),
                "Model/simulator unit mismatch")
        require(model.model.condition.ee_shape == "both" and model.dimensions.input_dim == 4,
                "Unexpected model tool interface")
        require(cfg.ee.type == "gripper" and list(cfg.actions) == ["grasp"] and
                cfg.actions.grasp.type == "grasp" and np.allclose(cfg.actions.grasp.from_offset, 0) and
                "init_pos" not in cfg.actions.grasp and "init_quat" not in cfg.actions.grasp,
                "Requires stationary single-grasp Level 3 interface")
        with (scene / "log.pkl").open("rb") as stream:
            logs = pickle.load(stream)
        times = np.asarray([float(row["t"]) for row in logs])
        require(len(logs) == metadata["num_source_frames"] and len(logs) > 62, "Raw/source frame count mismatch")
        require(np.isfinite(times).all() and times[0] == 0 and (np.diff(times) > 0).all(), "Invalid raw times")
        dt = float(cfg.sim.step_dt)
        n_iter = int(cfg.log.n_iter)
        require(n_iter > 0 and dt > 0 and np.isfinite(dt), "Invalid simulator time step")
        require(all(row["step"] == i * n_iter for i, row in enumerate(logs)), "Raw step phase mismatch")
        require(np.allclose(times, np.arange(len(logs)) * n_iter * dt, rtol=0, atol=1e-8), "Raw time mapping mismatch")
        require(np.isclose(times[60], 1.920, rtol=0, atol=1e-8) and
                np.isclose(times[62], 1.984, rtol=0, atol=1e-8), "Primary/auxiliary time mismatch")
        require(metadata["rollout_frames"][0] == 0 and metadata["rollout_frames"][-1] == 60,
                "Rollout endpoint mismatch")
        with np.load(source / "observations.npz", allow_pickle=False) as packet:
            frames = packet["frame_indices"].tolist()
            require(frames[0] == 0 and len(set(frames)) == len(frames), "Invalid observation frame mapping")
            indices = [frames.index(frame) for frame in metadata["rollout_frames"]]
            observed_times = packet["simulation_times"][indices]
            require(np.allclose(observed_times, times[metadata["rollout_frames"]], rtol=0, atol=1e-8),
                    "Observation/raw time mismatch")
            shapes = {}
            for key in ("observed", "ee_observed", "z_reference"):
                value = packet[key]
                require(value.shape[0] == len(frames) and np.isfinite(value).all(), "Invalid observations: " + key)
                shapes[key] = list(value.shape)
        positions = np.asarray(logs[0]["obj"]["particles"]["pos"], dtype=float)
        require(positions.ndim == 2 and positions.shape[1] == 3 and len(positions) and
                np.isfinite(positions).all(), "Invalid complete initial particles")
        raw_bytes = (scene / "log.pkl").stat().st_size
        disk = shutil.disk_usage(ROOT)
        report["storage"] = {"raw_log_bytes": raw_bytes, "raw_log_times_18_bytes": raw_bytes * 18,
                             "filesystem_free_bytes": disk.free,
                             "scope": "raw-log lower-bound estimate; project quota and other artifacts not included"}
        return {"unit_to_meter": scale, "step_dt": dt, "log_n_iter": n_iter,
                "raw_frames": len(logs), "raw_times_s": times.tolist(),
                "primary_s": float(times[60]), "auxiliary_s": float(times[62]),
                "warmup_horizon": int(cfg.warmup_horizon), "simulator_seed": int(cfg.sim.seed),
                "observation_seed": int(metadata["seed"]), "observation_shapes": shapes,
                "initial_particle_count": len(positions),
                "initial_particle_xyz_span_mm": (np.ptp(positions, axis=0) * 1000 * scale).tolist(),
                "mpm": resolved["sim"]["mpm"], "entities": resolved["entities"]}

    section("geometry_and_schedule", geometry_and_schedule)

    def material_and_assets():
        builder = ROOT / "sim/generate/builder.py"
        generator = ROOT / "sim/generate.py"
        names = [builder, generator, ROOT / "sim/generate/config/materials/dough.yaml",
                 ROOT / "sim/mpm/assets/meshes/processed/board-128.sdf",
                 ROOT / "sim/mpm/assets/meshes/processed/board-board.obj",
                 ROOT / "sim/mpm/assets/meshes/processed/board-board.provenance.json"]
        hashes = {}
        missing = []
        for path in names:
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = sha256(path)
            else:
                missing.append(str(path.relative_to(ROOT)))
        # Source expressions, rather than YAML alone, determine the effective yield parameters.
        text = builder.read_text(encoding="utf-8")
        func = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_material")
        expressions = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ElastoPlastic":
                expressions = {kw.arg: ast.get_source_segment(text, kw.value) for kw in node.keywords}
        require(bool(expressions), "Cannot identify the effective material constructor; inspect builder")
        excerpts = {path.name: [{"line": i, "source": line} for i, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1)
            if any(key in line for key in ("yield_", "warmup_horizon", "scene.step()", "check_and_log(step)"))]
                    for path in (builder, generator)}
        result = {"files_sha256": hashes, "missing_files": missing,
                  "elastoplastic_constructor_expressions": expressions, "source_excerpts": excerpts}
        from omegaconf import OmegaConf
        scene_cfg = OmegaConf.load(Path(metadata["scene_dir"]) / "config.yaml")
        OmegaConf.resolve(scene_cfg)
        materials = {}
        for name, entity in scene_cfg.entities.items():
            material = entity.material
            if material.type == "elastoplastic":
                row = {"declared": OmegaConf.to_container(material, resolve=True), "effective": {}}
                for key in ("yield_lower", "yield_higher"):
                    expression = expressions.get(key, "")
                    if expression in ("config.yield_low_high[0]", "config.yield_low_high[1]"):
                        row["effective"][key] = float(material.yield_low_high[int(expression[-2])])
                    else:
                        row["effective"][key] = {"needs_source_review": expression}
                materials[name] = row
        result["elastoplastic_entities"] = materials
        provenance = ROOT / "sim/mpm/assets/meshes/processed/board-board.provenance.json"
        if provenance.is_file():
            value = json.loads(provenance.read_text(encoding="utf-8"))
            result["board_provenance"] = value
            for path_key, hash_key in (("source_sdf", "source_sdf_sha256"), ("output_obj", "output_obj_sha256")):
                require(sha256(ROOT / value[path_key]) == value[hash_key], "Board provenance hash mismatch: " + path_key)
        report["material_and_assets_partial"] = result
        require(not missing, "Missing simulation assets/provenance: " + ", ".join(missing))
        return result

    section("material_and_assets", material_and_assets)

    def snapshot():
        from level4.artifacts import snapshot_runtime
        return {"manifest_sha256": snapshot_runtime(ROOT, output / "runtime_snapshot")}

    section("runtime_snapshot", snapshot)
    report["status"] = "ready_for_review" if not report["errors"] else "needs_review"
    report["not_verified"] = ["GPU execution", "allocation/project authorization", "nominal reset reproducibility",
                              "solver/log phase physical interpretation", "Level 4 task feasibility or model accuracy"]
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nPython: " + sys.version.split()[0])
    if "geometry_and_schedule" in report["checks"]:
        row = report["checks"]["geometry_and_schedule"]
        print("Frames=%s; unit_to_meter=%s; frame60=%ss; frame62=%ss" % (
            row["raw_frames"], row["unit_to_meter"], row["primary_s"], row["auxiliary_s"]))
    if "storage" in report:
        row = report["storage"]
        print("Raw-log x18 estimate: %.2f GiB; filesystem free: %.2f GiB" % (
            row["raw_log_times_18_bytes"] / 2**30, row["filesystem_free_bytes"] / 2**30))
    material = report.get("material_and_assets_partial", {})
    expressions = material.get("elastoplastic_constructor_expressions", {})
    print("Effective yield arguments: " + json.dumps({key: expressions.get(key) for key in ("yield_lower", "yield_higher")}))
    print("Effective material values: " + json.dumps(material.get("elastoplastic_entities", {})))
    print("\nQueue (full record: queue.txt):")
    print(queue.rstrip())
    print("\nCurrent user jobs:")
    print(jobs.rstrip())
    print("\nPREFLIGHT_STATUS=" + report["status"])
    print("REPORT=" + str(output / "report.json"))
    for error in report["errors"]:
        print("REVIEW: " + json.dumps(error))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
