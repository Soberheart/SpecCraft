"""Three-valued hard conditions over physical timestamps, never interpolated."""
import numpy as np
from .geometry import measure
from .spec import validate_spec


def aggregate(states):
    # A known violation proves failure even when other measurements are missing.
    return "violated" if "violated" in states else "unknown" if "unknown" in states else "satisfied"


def evaluate(spec, trajectory, *, prediction=False, experiment=False, common_schedule=False):
    validate_spec(spec, experiment=experiment)
    times = np.asarray([t for t, _ in trajectory], dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (len(times)>1 and (np.diff(times)<=0).any()):
        raise ValueError("Trajectory timestamps must be finite, unique and ordered")
    tolerance = spec["measurement"]["tolerance_ratio"]
    margin = spec["measurement"]["prediction_margin_ratio"] if prediction else 0.
    # Positive margins tighten only predicted feasibility, never execution truth.
    eps = spec["time"]["match_tolerance_s"]
    references = spec["initial_reference"]
    axes = spec["reference_frame"]
    minimum = spec["measurement"]["minimum_reference_span_mm"]

    def at(t, axis, kind, threshold):
        ref = references[f"{axis}_mm"]
        row = {"time_s": t, "axis": axis, "value_mm": None, "reference_mm": ref,
               "value": None, "threshold": threshold, "unit": "ratio", "relation": "le",
               "tolerance": tolerance, "prediction_margin": margin, "violation": None,
               "raw_residual": None, "status": "unknown", "reason": "missing_required_frame"}
        indices = np.flatnonzero(np.abs(times-t) <= eps)
        if len(indices) != 1:
            if len(indices)>1:
                row["reason"] = "ambiguous_timestamp_match"
            return row
        geometry = trajectory[int(indices[0])][1]
        if geometry.representation != spec["measurement"]["representation"]:
            row["reason"] = "representation_mismatch"
            return row
        measurement = measure(geometry, axes[axis], np.asarray(axes["origin_mm"])/(1000*geometry.unit_to_meter))
        if measurement["status"] != "measured":
            row["reason"] = measurement["reason"]
            return row
        value_mm = measurement["value"]
        row["value_mm"] = value_mm
        if ref <= minimum:
            row["reason"] = "unmeasurable_initial_span"
            return row
        value = value_mm/ref if kind == "terminal" else abs(value_mm/ref-1)
        residual = value-threshold
        adjusted = residual-tolerance+margin
        # Numerical roundoff only; this is not a physical measurement tolerance.
        roundoff = 32*np.finfo(float).eps*max(1.,abs(value),abs(threshold))
        satisfied=adjusted <= roundoff
        row.update(value=float(value), raw_residual=float(residual), violation=0. if satisfied else float(max(0.,adjusted)),
                   status="satisfied" if satisfied else "violated", reason="")
        return row

    # Verify frozen initial references against the actual initial geometry.
    reference_rows = []
    for axis in ("u", "v"):
        row = at(0., axis, "terminal", 1.)
        if row["value_mm"] is not None and row["reference_mm"]>minimum:
            error = abs(row["value_mm"]/row["reference_mm"]-1)
            row.update(value=error, threshold=tolerance, prediction_margin=0.,
                       raw_residual=error-tolerance, violation=max(0.,error-tolerance),
                       status="satisfied" if error <= tolerance+32*np.finfo(float).eps else "unknown",
                       reason="" if error <= tolerance+32*np.finfo(float).eps else "initial_reference_mismatch")
        reference_rows.append(row)
    initial_ok = all(row["status"] == "satisfied" for row in reference_rows)
    terminal = at(spec["time"]["primary_s"], "u", "terminal", 1-spec["terminal_hard"]["rho"])
    process_rows = []
    required = spec["time"]["required_s"]
    scope = "model_common_schedule"
    if not prediction and not common_schedule and "execution_required_s" in spec["time"]:
        required = spec["time"]["execution_required_s"]
        scope = "frozen_dense_execution_schedule"
    if spec["process_hard"]:
        process_rows = [at(t,"v","process",spec["process_hard"][0]["eta"]) for t in required]
    auxiliary = {"time_s": spec["time"]["auxiliary_s"],
                 "terminal": at(spec["time"]["auxiliary_s"], "u", "terminal", 1-spec["terminal_hard"]["rho"]),
                 "used_for_selection": False}
    hard_rows = [terminal] + process_rows
    if not initial_ok:
        for row in hard_rows:
            row.update(status="unknown", reason="initial_reference_unverified")
        auxiliary["terminal"].update(status="unknown", reason="initial_reference_unverified")
    status = aggregate([row["status"] for row in hard_rows])
    violations = [row for row in process_rows if row["status"] == "violated"]
    return {"schema_version": "1.1.0", "task_id": spec["task_id"],
            "source": "prediction" if prediction else "independent_execution",
            "status": status, "strict_success": status == "satisfied", "initial_check": reference_rows,
            "terminal": terminal, "process": {"status": aggregate([r["status"] for r in process_rows]),
                "samples": process_rows, "sampling_scope": scope,
                "first_detected_violation_s": violations[0]["time_s"] if violations else None,
                "max_violation": max((r["violation"] for r in process_rows if r["violation"] is not None), default=None),
                "missing_or_unknown_times_s": [r["time_s"] for r in process_rows if r["status"] == "unknown"]},
            "auxiliary": auxiliary, "auxiliary_used_for_selection": False, "continuous_time_guarantee": False}
