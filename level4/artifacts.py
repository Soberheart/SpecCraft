"""Immutable, hash-addressed geometry packets and small JSON artifacts."""
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from .geometry import from_occupancy


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")


def load_grid(pool):
    with np.load(Path(pool)/"grid.npz",allow_pickle=False) as packet:
        return {k:packet[k] for k in packet.files}


def load_trajectory(packet_path, grid):
    with np.load(packet_path,allow_pickle=False) as packet:
        times, classes = packet["times_s"], packet["classes"]
        complete, outside = packet["complete"], packet["outside_support"]
    if times.ndim!=1 or classes.shape!=(len(times),len(grid["query"])) or complete.shape!=times.shape or outside.shape!=times.shape:
        raise ValueError("Invalid trajectory packet dimensions")
    return [(float(t), from_occupancy(grid["query"],c,grid["counts"],grid["spacing"],
                float(grid["unit_to_meter"]),complete=bool(ok),outside_support=bool(out)))
            for t,c,ok,out in zip(times,classes,complete,outside)]


def save_packet(path, times, classes, outside=None):
    times=np.asarray(times,dtype=float)
    classes=np.asarray(classes)
    if times.ndim!=1 or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError("Invalid packet timestamps")
    if classes.ndim!=2 or len(classes)!=len(times) or not np.isfinite(classes).all() or ((classes<0)|(classes>4)|(classes!=np.rint(classes))).any():
        raise ValueError("Invalid material classes; refusing lossy integer conversion")
    if outside is not None and (np.asarray(outside).shape!=times.shape or np.asarray(outside).dtype!=np.bool_):
        raise ValueError("outside_support must be one Boolean per timestamp")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream,times_s=times,classes=np.asarray(classes,dtype=np.uint8),
                            complete=np.ones(len(times),dtype=bool),
                            outside_support=np.zeros(len(times),dtype=bool) if outside is None else outside)


def checked_pool(root):
    root=Path(root)
    manifest=read_json(root/"pool.json")
    for relative,expected in manifest["input_sha256"].items():
        if sha256(root/relative)!=expected:
            raise ValueError(f"Pool artifact changed: {relative}")
    return manifest


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def runtime_environment():
    return {"python":sys.version, "packages":dict(sorted(
        (d.metadata["Name"].lower(),d.version) for d in importlib.metadata.distributions()
        if d.metadata.get("Name")))}


def _submission_environment(manifest):
    """Distinct exact submit/compute environments, backed by a frozen probe."""
    submit = manifest.get("environment")
    execution = manifest.get("execution_environment")
    evidence = manifest.get("execution_environment_evidence")
    if execution is None and evidence is None:
        return submit
    if not isinstance(execution, dict) or not isinstance(evidence, dict):
        raise ValueError("Execution environment requires its frozen probe evidence")
    digest = evidence.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or not evidence.get("path"):
        raise ValueError("Invalid execution environment evidence entry")
    path = Path(evidence["path"])
    if not path.is_file() or path.is_symlink() or sha256(path) != digest:
        raise ValueError("Execution environment probe evidence changed")
    probe = read_json(path)
    if (probe.get("kind") != "environment_diagnostic_not_simulation"
            or probe.get("simulation_run") is not False or not probe.get("job_id")
            or probe.get("expected_environment") != submit
            or probe.get("actual_environment") != execution):
        raise ValueError("Execution environment differs from its frozen probe evidence")
    return execution if os.environ.get("PBS_JOBID") else submit


def _sealed_runtime(root):
    """Verify a complete isolated source tree frozen on the submission host."""
    manifest_path = os.environ.get("LEVEL4_SUBMISSION_MANIFEST")
    expected = os.environ.get("LEVEL4_SUBMISSION_MANIFEST_SHA256")
    if not manifest_path and not expected:
        return None
    if not manifest_path or not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Submission manifest and SHA256 must be supplied together")
    path = Path(manifest_path).resolve()
    root = Path(root).resolve()
    if root != path.parent / "files" or sha256(path) != expected:
        raise ValueError("Wrong sealed runtime root or changed submission manifest")
    manifest = read_json(path)
    if manifest.get("kind") != "sealed_submission_runtime" or not re.fullmatch(
            r"[0-9a-f]{40}", manifest.get("code_commit", "")):
        raise ValueError("Invalid sealed runtime provenance")
    if _submission_environment(manifest) != runtime_environment():
        raise ValueError("Sealed runtime Python or dependency versions changed")
    entries = manifest.get("files_sha256", {})
    if not entries:
        raise ValueError("Empty sealed runtime")
    for name, digest in entries.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid sealed runtime file entry")
        target = root / name
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise ValueError("Sealed runtime source/asset changed: " + name)
    actual = set()
    for target in root.rglob("*"):
        if "__pycache__" in target.relative_to(root).parts:
            continue
        if target.is_symlink():
            raise ValueError("Symlink added to sealed runtime")
        if target.is_file():
            actual.add(target.relative_to(root).as_posix())
    if actual != set(entries):
        raise ValueError("Sealed runtime file set changed")
    return manifest


def runtime_commit(root):
    """Actual submit-host Git identity, or live Git when using a normal checkout."""
    manifest = _sealed_runtime(root)
    if manifest is not None:
        return manifest["code_commit"]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def runtime_files(root):
    """Actual runtime sources plus ignored simulation assets, never experiment data."""
    root=Path(root)
    sealed = _sealed_runtime(root)
    if sealed is not None:
        return [root / name for name in sorted(sealed["files_sha256"])]
    names=set(subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard"],
                                     cwd=root,text=True,encoding="utf-8").split("\0"))-{""}
    files={p for name in names if (p:=root/name).is_file()}
    assets=root/"sim/mpm/assets"
    if assets.exists():
        files.update(p for p in assets.rglob("*") if p.is_file())
    # These ignored in-repository dependencies are imported by simulation/model.
    # Preserve their actual sources without Git credentials, builds or caches.
    source_suffixes={".py",".cpp",".cu",".c",".h",".hpp",".cuh",".cmake",".toml",".yaml",".json",".txt",".md"}
    for relative in ("sim/sdftoolbox","net/nvdiffrast"):
        dependency=root/relative
        if dependency.exists():
            files.update(p for p in dependency.rglob("*") if p.is_file()
                         and not {".git","__pycache__","build","dist"}.intersection(p.relative_to(dependency).parts)
                         and (p.suffix in source_suffixes or p.name in ("LICENSE","CMakeLists.txt")))
    # Absolute source maps remain portable through relative names.
    return sorted(files,key=lambda p:p.relative_to(root).as_posix())


def snapshot_runtime(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    # Resolve and validate before creating output; a sealed source tree has no
    # generated artifacts except Python's __pycache__ directories.
    source_files = runtime_files(root)
    if os.environ.get("LEVEL4_SUBMISSION_MANIFEST") and (output == root or root in output.parents):
        raise ValueError("Write sealed runtime outputs outside its source tree")
    output.mkdir(parents=True,exist_ok=False)
    entries={}
    for path in source_files:
        name=path.relative_to(root).as_posix()
        dest=output/"files"/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        entries[name]=sha256(dest)
        if sha256(path)!=entries[name]:
            raise ValueError(f"Source changed during snapshot: {name}")
    manifest = {"created_utc":utc_now(),"files_sha256":entries,
        "environment":runtime_environment(),
        "external_dependencies":"Installed packages recorded by version; external repos/weights require their own stable archive and hash"}
    if os.environ.get("LEVEL4_SUBMISSION_MANIFEST"):
        manifest.update(code_commit=runtime_commit(root),
                        inventory_origin="verified_isolated_submission_snapshot",
                        submission_manifest_sha256=os.environ["LEVEL4_SUBMISSION_MANIFEST_SHA256"])
    write_json(output/"manifest.json",manifest)
    return sha256(output/"manifest.json")


def verify_runtime(root,snapshot):
    root,snapshot=Path(root),Path(snapshot)
    manifest=read_json(snapshot/"manifest.json")
    for name,expected in manifest["files_sha256"].items():
        if sha256(snapshot/"files"/name)!=expected or sha256(root/name)!=expected:
            raise ValueError(f"Runtime source/asset differs from snapshot: {name}")
    current={p.relative_to(root).as_posix() for p in runtime_files(root)}
    if current!=set(manifest["files_sha256"]):
        raise ValueError("Runtime source file set changed")
    if runtime_environment()!=manifest["environment"]:
        raise ValueError("Python or dependency versions changed; create a new attempt")
