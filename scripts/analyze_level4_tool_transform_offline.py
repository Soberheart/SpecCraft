"""Read-only CPU analysis of the saved Level 4 tool trajectories.

This command is deliberately outside the prediction/selection path.  It reads
the fixed-pair evidence and an already saved nominal ``actual_tools`` directory,
then compares current, previous and latest solver poses under a small, explicit
set of transform hypotheses.  It never imports the simulator, Torch, Taichi,
or model code and never changes the 0.001 mm diagnostic limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.plan_level3_pose_cem import transform_tools


FRAME_COUNT = 63
UNIT_TO_METER = 0.2
POSITION_LIMIT_MM = 0.001


def _finite_vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Invalid %s" % name)
    return array


def _load_packet(path: Path):
    with np.load(path, allow_pickle=False) as packet:
        return {name: packet[name].copy() for name in packet.files}


def _max_error_mm(actual, expected):
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64)
                               - np.asarray(expected, dtype=np.float64))) * 200.0)


def _hypotheses(parameters):
    """Return documented transform hypotheses, without choosing an action."""
    parameters = _finite_vector(parameters, (3,), "candidate parameters")
    dx, dy, yaw = (float(value) for value in parameters)
    # The first entry is the frozen Level 4 convention.  The remaining entries
    # are diagnostics for unit, yaw-sign, and translation-order mistakes only.
    return {
        "frozen_center0_mm_unit02_yaw_plus_translate":
            (np.zeros(3), np.array([dx, dy, yaw]), 0.2, "rotate_then_translate"),
        "center0_mm_unit01_yaw_plus_translate":
            (np.zeros(3), np.array([dx, dy, yaw]), 1.0, "rotate_then_translate"),
        "center0_mm_unit02_yaw_minus_translate":
            (np.zeros(3), np.array([dx, dy, -yaw]), 0.2, "rotate_then_translate"),
        "center0_mm_unit02_yaw_plus_translate_after_rotation":
            (np.zeros(3), np.array([dx, dy, yaw]), 0.2, "translate_then_rotate"),
    }


def _apply_hypothesis(point, center, parameters, unit_to_meter, order):
    point = _finite_vector(point, (3,), "tool point")
    center = _finite_vector(center, (3,), "transform center")
    parameters = _finite_vector(parameters, (3,), "transform parameters")
    if order == "rotate_then_translate":
        return transform_tools(point[None, None, :], center,
                               [parameters], unit_to_meter)[0, 0, 0, :3]
    if order != "translate_then_rotate":
        raise ValueError("Unknown transform order")
    # Diagnostic alternate only: translation first, then rotation about the
    # frozen center.  It is never used by the planner.  ``point`` is already in
    # simulator units; only the mm translation is converted (L4-01 A3 fix: the
    # previous version divided the point itself by 1000*unit_to_meter).
    translated = point.copy()
    translated[:2] += parameters[:2] / (1000.0 * float(unit_to_meter))
    angle = np.deg2rad(float(parameters[2]))
    c, s = np.cos(angle), np.sin(angle)
    relative = translated - center
    rotated = np.array([c * relative[0] - s * relative[1],
                        s * relative[0] + c * relative[1], relative[2]])
    return (rotated + center).astype(np.float32)


def _candidate_parameters(evidence: Path):
    pool = json.loads((evidence / "source_pool_manifest.json").read_text(encoding="utf-8"))
    rows = {row["candidate_id"]: row for row in pool["candidates"]}
    return rows


def _frame_paths(root: Path):
    paths = [root / ("frame_%03d.npz" % frame) for frame in range(FRAME_COUNT)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing saved nominal tool frames: " + missing[0])
    return paths


def analyze_candidate(evidence: Path, nominal_tools: Path, candidate_id: str):
    rows = _candidate_parameters(evidence)
    if candidate_id not in rows:
        raise ValueError("Candidate is not present in frozen pool: " + candidate_id)
    parameters = _finite_vector(rows[candidate_id]["parameters_mm_mm_deg"], (3,),
                                 "candidate parameters")
    actual_root = evidence / "execution" / candidate_id / "actual_tools"
    actual_paths = _frame_paths(actual_root)
    nominal_paths = _frame_paths(nominal_tools)
    hypotheses = _hypotheses(parameters)
    records = []
    for frame, (actual_path, nominal_path) in enumerate(zip(actual_paths, nominal_paths)):
        actual = _load_packet(actual_path)
        nominal = _load_packet(nominal_path)
        if int(actual["step"]) != int(nominal["step"]) or int(actual["step"]) != frame * 16:
            raise ValueError("Saved solver step mismatch at frame %d" % frame)
        record = {"frame": frame, "step": int(actual["step"]),
                  "nominal_label_s": frame * 0.032}
        for finger in (1, 2):
            key = "tool_%03d_pos" % finger
            actual_point = _finite_vector(actual[key], (3,), key)
            nominal_point = _finite_vector(nominal[key], (3,), "nominal " + key)
            latest = _finite_vector(actual["latest_tool_%03d_pos" % finger], (3,),
                                    "latest " + key)
            previous = _finite_vector(actual["previous_tool_%03d_pos" % finger], (3,),
                                      "previous " + key)
            for hypothesis_name, (center, hyp_parameters, unit, order) in hypotheses.items():
                expected = _apply_hypothesis(nominal_point, center, hyp_parameters, unit, order)
                prefix = "finger_%d_%s" % (finger, hypothesis_name)
                record[prefix + "_current_error_mm"] = _max_error_mm(actual_point, expected)
                record[prefix + "_latest_error_mm"] = _max_error_mm(latest, expected)
                record[prefix + "_previous_error_mm"] = _max_error_mm(previous, expected)
            record["finger_%d_current_latest_error_mm" % finger] = _max_error_mm(actual_point, latest)
            record["finger_%d_current_previous_error_mm" % finger] = _max_error_mm(actual_point, previous)
        records.append(record)

    summaries = {}
    for hypothesis_name in hypotheses:
        for sample in ("current", "latest", "previous"):
            keys = ["finger_%d_%s_%s_error_mm" % (finger, hypothesis_name, sample)
                    for finger in (1, 2)]
            worst = max(records, key=lambda row: max(row[key] for key in keys))
            value = max(worst[key] for key in keys)
            summaries[hypothesis_name + "_" + sample] = {
                "max_error_mm": float(value), "frame": worst["frame"],
                "comparison_limit_mm": POSITION_LIMIT_MM,
                "exceeds_original_limit": bool(value > POSITION_LIMIT_MM),
            }
    current_latest = {
        "finger_1_max_mm": max(row["finger_1_current_latest_error_mm"] for row in records),
        "finger_2_max_mm": max(row["finger_2_current_latest_error_mm"] for row in records),
        "finger_1_frame": max(records, key=lambda row: row["finger_1_current_latest_error_mm"])["frame"],
        "finger_2_frame": max(records, key=lambda row: row["finger_2_current_latest_error_mm"])["frame"],
    }
    frozen = summaries["frozen_center0_mm_unit02_yaw_plus_translate_current"]
    return {
        "candidate_id": candidate_id,
        "parameters_mm_mm_deg": [float(value) for value in parameters],
        "frame_count": len(records),
        "nominal_tools_directory": str(nominal_tools),
        "frozen_position_limit_mm_unchanged": POSITION_LIMIT_MM,
        "hypothesis_summaries": summaries,
        "current_latest_separation": current_latest,
        "frozen_hypothesis_status": "violated" if frozen["exceeds_original_limit"] else "within_limit",
        "records": records,
        "scope": "read_only_cpu_diagnosis; no_model; no_simulator; no_selection; no_task_evaluation",
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_receipt(evidence: Path, nominal_tools: Path):
    """Hash every input actually read, and state how the nominal relates to the run."""
    files = {}
    for name in ("source_pool_manifest.json", "prospective_tool_diagnosis.json"):
        files[name] = _sha256(evidence / name)
    tree = hashlib.sha256()
    for path in _frame_paths(nominal_tools):
        tree.update(path.name.encode("utf-8") + b"\0" + _sha256(path).encode("ascii"))
    same_run = nominal_tools == (evidence / "execution" / "candidate_000" / "actual_tools").resolve()
    return {
        "files_sha256": files,
        "nominal_tools_frames_sha256_of_hashes": tree.hexdigest(),
        "nominal_reference": ("same_evidence_candidate_000_actual_tools" if same_run
                              else "external_directory_not_bound_to_this_run"),
    }


def analyze(evidence, nominal_tools, candidates):
    evidence, nominal_tools = Path(evidence).resolve(), Path(nominal_tools).resolve()
    if not (evidence / "prospective_tool_diagnosis.json").is_file():
        raise ValueError("Fixed-pair evidence does not contain prospective_tool_diagnosis.json")
    candidate_rows = _candidate_parameters(evidence)
    if candidates == ["all"]:
        candidates = [name for name in ("candidate_000", "candidate_001") if name in candidate_rows]
    if not candidates:
        raise ValueError("No requested candidate is present")
    return {
        "schema": "level4_cpu_tool_transform_offline_v2",
        "schema_change": "v2 fixes the translate_then_rotate unit conversion and adds input_receipt; "
                         "the frozen rotate_then_translate branch is unchanged",
        "evidence_directory": str(evidence),
        "nominal_tools_directory": str(nominal_tools),
        "input_receipt": _input_receipt(evidence, nominal_tools),
        "position_limit_mm_unchanged": POSITION_LIMIT_MM,
        "analysis": {candidate: analyze_candidate(evidence, nominal_tools, candidate)
                     for candidate in candidates},
        "interpretation": {
            "frozen_hypothesis": "the Level 4 transform_tools convention only",
            "alternate_hypotheses": "diagnostic explanations, never planner inputs",
            "candidate_pool_release": False,
            "task_evaluation": "not_evaluated",
            "model_alignment": "not_certified",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--nominal-tools", type=Path, required=True,
                        help="saved nominal actual_tools directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=["all"],
                        choices=("all", "candidate_000", "candidate_001"))
    args = parser.parse_args()
    candidates = [name for name in args.candidate if name != "all"] or ["all"]
    result = analyze(args.evidence, args.nominal_tools, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing report: " + str(args.output))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("CPU_OFFLINE_ANALYSIS_PASSED")
    print("REPORT=" + str(args.output.resolve()))


if __name__ == "__main__":
    main()
