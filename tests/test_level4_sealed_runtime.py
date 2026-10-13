"""Meaningful no-Git execution and immutable submission-snapshot checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from level4.artifacts import (read_json, runtime_files, runtime_commit, runtime_environment,
                              sha256, snapshot_runtime, verify_runtime, write_json)


def fixture(base):
    root = base / "runtime/files"
    root.mkdir(parents=True)
    (root / "sim.py").write_bytes(b"SOURCE = 1\n")
    write_json(root.parent / "manifest.json", {
        "kind": "sealed_submission_runtime", "code_commit": "a" * 40,
        "files_sha256": {"sim.py": sha256(root / "sim.py")}, "environment": runtime_environment()})
    env = {"LEVEL4_SUBMISSION_MANIFEST": str(root.parent / "manifest.json"),
           "LEVEL4_SUBMISSION_MANIFEST_SHA256": sha256(root.parent / "manifest.json")}
    return root, env


def separate_environments(root, env):
    manifest = read_json(root.parent / "manifest.json")
    submit = manifest["environment"]
    execution = {"python": "observed compute fixture Python", "packages": dict(submit["packages"])}
    evidence = root.parent / "compute_probe.json"
    write_json(evidence, {"kind": "environment_diagnostic_not_simulation", "job_id": "probe.fixture",
                         "simulation_run": False, "expected_environment": submit,
                         "actual_environment": execution})
    manifest.update(execution_environment=execution,
                    execution_environment_evidence={"path": str(evidence), "sha256": sha256(evidence)})
    (root.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    env["LEVEL4_SUBMISSION_MANIFEST_SHA256"] = sha256(root.parent / "manifest.json")
    return submit, execution, evidence


class SealedRuntimeTests(unittest.TestCase):
    def test_submit_and_compute_versions_are_separate_exact_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            submit, execution, _ = separate_environments(root, env)
            with patch.dict(os.environ, dict(env, PBS_JOBID="")):
                self.assertEqual(runtime_commit(root), "a" * 40)
                with patch("level4.artifacts.runtime_environment", return_value=execution):
                    with self.assertRaises(ValueError): runtime_commit(root)
            with patch.dict(os.environ, dict(env, PBS_JOBID="compute.fixture")):
                with patch("level4.artifacts.runtime_environment", return_value=execution):
                    self.assertEqual(runtime_commit(root), "a" * 40)
                with patch("level4.artifacts.runtime_environment", return_value=submit):
                    with self.assertRaises(ValueError): runtime_commit(root)
                unexpected = dict(execution, python="unreviewed third interpreter")
                with patch("level4.artifacts.runtime_environment", return_value=unexpected):
                    with self.assertRaises(ValueError): runtime_commit(root)

    def test_compute_package_drift_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            _, execution, _ = separate_environments(root, env)
            changed = dict(execution, packages=dict(execution["packages"], numpy="unreviewed"))
            with patch.dict(os.environ, dict(env, PBS_JOBID="compute.fixture")):
                with patch("level4.artifacts.runtime_environment", return_value=changed):
                    with self.assertRaises(ValueError): runtime_files(root)

    def test_compute_evidence_tampering_and_incomplete_contract_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            _, execution, evidence = separate_environments(root, env)
            raw = evidence.read_bytes()
            evidence.write_bytes(raw + b" ")
            with patch.dict(os.environ, dict(env, PBS_JOBID="compute.fixture")):
                with patch("level4.artifacts.runtime_environment", return_value=execution):
                    with self.assertRaises(ValueError): runtime_commit(root)
            evidence.write_bytes(raw)
            path = root.parent / "manifest.json"
            content = read_json(path)
            del content["execution_environment_evidence"]
            path.write_text(json.dumps(content), encoding="utf-8")
            env["LEVEL4_SUBMISSION_MANIFEST_SHA256"] = sha256(path)
            with patch.dict(os.environ, env):
                with self.assertRaises(ValueError): runtime_commit(root)
            content["execution_environment_evidence"] = {"path": str(evidence), "sha256": sha256(evidence)}
            content["execution_environment"] = dict(execution, python="not the observed interpreter")
            path.write_text(json.dumps(content), encoding="utf-8")
            env["LEVEL4_SUBMISSION_MANIFEST_SHA256"] = sha256(path)
            with patch.dict(os.environ, env):
                with self.assertRaises(ValueError): runtime_commit(root)

    def test_compute_snapshot_records_actual_environment_and_cannot_cross_reset_context(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, env = fixture(base)
            _, execution, _ = separate_environments(root, env)
            with patch.dict(os.environ, dict(env, PBS_JOBID="compute.fixture")):
                with patch("level4.artifacts.runtime_environment", return_value=execution):
                    snapshot_runtime(root, base / "output")
                    self.assertEqual(read_json(base / "output/manifest.json")["environment"], execution)
                    verify_runtime(root, base / "output")
            with patch.dict(os.environ, dict(env, PBS_JOBID="")):
                with self.assertRaises(ValueError): verify_runtime(root, base / "output")

    def test_no_git_snapshot_verification_and_cache_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, env = fixture(base)
            with patch.dict(os.environ, env), patch("level4.artifacts.subprocess.check_output", side_effect=FileNotFoundError("git absent")):
                self.assertEqual(runtime_commit(root), "a" * 40)
                self.assertEqual(runtime_files(root), [root / "sim.py"])
                cache = root / "__pycache__"
                cache.mkdir()
                (cache / "sim.cpython-39.pyc").write_bytes(b"cache")
                snapshot_runtime(root, base / "output")
                verify_runtime(root, base / "output")
                result = read_json(base / "output/manifest.json")
                self.assertEqual(result["submission_manifest_sha256"], env["LEVEL4_SUBMISSION_MANIFEST_SHA256"])
                self.assertEqual(result["inventory_origin"], "verified_isolated_submission_snapshot")
                with self.assertRaises(ValueError):
                    snapshot_runtime(root, root / "bad-output")
                self.assertFalse((root / "bad-output").exists())

    def test_changed_added_removed_files_and_wrong_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            with patch.dict(os.environ, env):
                original = (root / "sim.py").read_bytes()
                (root / "sim.py").write_bytes(b"changed source")
                with self.assertRaises(ValueError): runtime_files(root)
                (root / "sim.py").write_bytes(original)
                (root / "new.py").write_bytes(b"new source")
                with self.assertRaises(ValueError): runtime_commit(root)
                (root / "new.py").unlink()
                (root / "sim.py").unlink()
                with self.assertRaises(ValueError): runtime_files(root)
                (root / "sim.py").write_bytes(original)
                with self.assertRaises(ValueError): runtime_commit(root.parent)

    def test_manifest_tampering_and_incomplete_environment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            with patch.dict(os.environ, env):
                path = root.parent / "manifest.json"
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError): runtime_commit(root)
                path.write_bytes(original)
                del os.environ["LEVEL4_SUBMISSION_MANIFEST_SHA256"]
                with self.assertRaises(ValueError): runtime_files(root)

    def test_path_traversal_and_environment_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = fixture(Path(directory))
            path = root.parent / "manifest.json"
            content = read_json(path)
            content["files_sha256"] = {"../outside.py": "a" * 64}
            path.write_text(json.dumps(content), encoding="utf-8")
            env["LEVEL4_SUBMISSION_MANIFEST_SHA256"] = sha256(path)
            with patch.dict(os.environ, env):
                with self.assertRaises(ValueError): runtime_files(root)
            content["files_sha256"] = {"sim.py": sha256(root / "sim.py")}
            path.write_text(json.dumps(content), encoding="utf-8")
            env["LEVEL4_SUBMISSION_MANIFEST_SHA256"] = sha256(path)
            with patch.dict(os.environ, env):
                snapshot_runtime(root, Path(directory) / "output")
                with patch("level4.artifacts.runtime_environment", return_value={"python": "changed"}):
                    with self.assertRaises(ValueError): verify_runtime(root, Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
