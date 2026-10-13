"""Versioned minimal task contract. Null calibration prevents task experiments."""
import math
from .geometry import axis_vector
import numpy as np


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number (uncalibrated null is blocked)")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def validate_spec(spec, *, experiment=False):
    try:
        return _validate_spec(spec, experiment=experiment)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed TaskSpec: {exc}") from exc


def _validate_spec(spec, *, experiment=False):
    if spec.get("schema_version") != "1.0.0" or not isinstance(spec.get("task_id"),str) or not spec["task_id"]:
        raise ValueError("Unsupported task schema or missing task_id")
    if spec.get("object") != "whole_material" or spec.get("semantics") != "world_axis_extent":
        raise ValueError("Only whole-material world-axis extent is implemented; deformation needs a pose guard")
    frame = spec["reference_frame"]
    if frame.get("kind") != "initial_fixed_world" or frame.get("unit") != "mm":
        raise ValueError("Reference frame must be fixed world coordinates in mm")
    if np.asarray(frame["origin_mm"]).shape != (3,) or not np.isfinite(frame["origin_mm"]).all():
        raise ValueError("Invalid origin")
    u, v = axis_vector(frame["u"]), axis_vector(frame["v"])
    if abs(u @ v) > 1e-10:
        raise ValueError("u and v must be orthogonal")
    reference = spec["initial_reference"]
    if reference.get("source") != "privileged_full_material_initial_state":
        raise ValueError("Initial extent needs full material evidence; partial depth is insufficient")
    for name in ("u_mm", "v_mm"):
        number(reference[name], name, 0)
    resolution = number(spec["measurement"]["minimum_reference_span_mm"], "minimum_reference_span_mm", 0)
    if resolution <= 0:
        raise ValueError("Measurable resolution must be positive")
    if spec["measurement"].get("representation") != "voxel_union" or spec["measurement"].get("operator") != "extrema":
        raise ValueError("Primary evaluation requires voxel-union extrema")
    number(spec["measurement"]["tolerance_ratio"], "tolerance_ratio", 0)
    number(spec["measurement"]["prediction_margin_ratio"], "prediction_margin_ratio", 0)
    t = spec["time"]
    primary = number(t["primary_s"], "primary_s", 0)
    number(t["auxiliary_s"], "auxiliary_s", primary)
    if experiment and (primary!=1.920 or t["auxiliary_s"]!=1.984):
        raise ValueError("This protocol freezes primary=1.920s and auxiliary=1.984s")
    eps = number(t["match_tolerance_s"], "match_tolerance_s", 0)
    times = t["required_s"]
    if not isinstance(times, list) or len(times) < 2:
        raise ValueError("Need initial and terminal required times")
    for item in times:
        number(item, "required_s", 0)
    if times[0] != 0 or times[-1] != primary or np.min(np.diff(times)) <= 2*eps:
        raise ValueError("Required times must be distinct, increasing, initial=0, terminal=primary")
    terminal = spec["terminal_hard"]
    if terminal.get("id") != "narrow_u" or terminal.get("relation") != "le" or terminal.get("hard") is not True:
        raise ValueError("Unsupported terminal hard condition")
    rho = number(terminal["rho"], "rho", 0, 1)
    if rho <= 0 or rho >= 1:
        raise ValueError("Require 0 < rho < 1")
    process = spec["process_hard"]
    if not isinstance(process,list) or len(process) > 1:
        raise ValueError("Only one protect-v condition supported")
    if process:
        condition = process[0]
        if condition.get("id") != "protect_v" or condition.get("relation") != "le" or condition.get("hard") is not True:
            raise ValueError("Unsupported process hard condition")
        number(condition["eta"], "eta", 0, 1)
    dense = t.get("execution_required_s")
    if dense is not None:
        if not isinstance(dense, list) or len(dense) < 2:
            raise ValueError("Execution schedule must list frozen raw log times")
        for item in dense:
            number(item, "execution_required_s", 0)
        if dense[0] != 0 or dense[-1] != primary or np.min(np.diff(dense)) <= 2*eps:
            raise ValueError("Execution schedule must increase from zero to primary")
        if any(sum(abs(x-y) <= eps for y in dense) != 1 for x in times):
            raise ValueError("Common schedule must be a subset of dense execution times")
    elif experiment and process:
        raise ValueError("B experiments require a frozen dense execution schedule")
    if spec.get("soft_preferences") != ["minimum_normalized_action_l2"]:
        raise ValueError("Unsupported soft preference")
    calibration = spec["calibration"]
    if calibration.get("status") not in ("frozen_development", "synthetic_only"):
        raise ValueError("Threshold calibration not frozen")
    if experiment and calibration["status"] != "frozen_development":
        raise ValueError("Synthetic thresholds cannot start material experiments")
    if any(not isinstance(calibration.get(k),str) or not calibration[k] for k in ("evidence_sha256","split_id")):
        raise ValueError("Calibration evidence and grouped split are mandatory")
    if experiment:
        digest=calibration["evidence_sha256"]
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Calibration must reference a SHA-256 digest")
    return spec
