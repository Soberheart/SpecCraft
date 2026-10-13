"""Feasibility-first selection. Execution truth is not accepted by online selection."""
import numpy as np


def point_goal_score(latent,goal):
    latent,goal=np.asarray(latent,dtype=float),np.asarray(goal,dtype=float)
    if latent.shape!=goal.shape or latent.ndim!=2 or not np.isfinite(latent).all() or not np.isfinite(goal).all():
        raise ValueError("PointGoal requires matching finite token matrices")
    norms=np.linalg.norm(latent,axis=-1)*np.linalg.norm(goal,axis=-1)
    if (norms<=0).any():
        raise ValueError("Undefined PointGoal cosine")
    return float(np.mean(np.sum(latent*goal,axis=-1)/norms))


def constraint_key(record):
    report = record["prediction"]
    if report.get("source") != "prediction":
        raise ValueError("Online selection cannot use execution truth")
    constraints = [report["terminal"]] + report["process"]["samples"]
    unknown = report["status"] == "unknown"
    # Each residual is normalized by its declared scale, independently of unit.
    if any(not np.isfinite(row.get("normalization_scale",1.)) or row.get("normalization_scale",1.) <= 0
           for row in constraints):
        raise ValueError("Violation normalization scales must be finite and positive")
    if any(row["violation"] is not None and (not np.isfinite(row["violation"]) or row["violation"]<0) for row in constraints):
        raise ValueError("Violations must be finite and nonnegative")
    residual = sum(row["violation"]/row.get("normalization_scale",1.)
                   for row in constraints if row["violation"] is not None)
    return (not record["legal"], unknown, report["status"] != "satisfied",
            residual, record["action_cost"], record["candidate_id"])


def select(records, method="constraint"):
    if method not in ("constraint","latent"):
        raise ValueError("Unknown selection method")
    for record in records:
        if record["prediction"].get("source")!="prediction":
            raise ValueError("Online selection requires prediction reports")
        if type(record["legal"]) is not bool or not isinstance(record["candidate_id"],str) or not record["candidate_id"]:
            raise ValueError("Invalid candidate identity or legality")
        if not np.isfinite(record["action_cost"]) or record["action_cost"] < 0:
            raise ValueError("Invalid action cost")
    if len({r["candidate_id"] for r in records})!=len(records):
        raise ValueError("Duplicate candidate IDs")
    legal = [r for r in records if r["legal"]]
    if method == "latent":
        finite = [r for r in legal if np.isfinite(r["latent_score"])]
        chosen = min(finite, key=lambda r:(-r["latent_score"],r["action_cost"],r["candidate_id"]), default=None)
    else:
        ordered = sorted(legal, key=constraint_key)
        chosen = next((r for r in ordered if r["prediction"]["status"] == "satisfied"), None)
    return {"status": "selected" if chosen else "no_feasible_candidate",
            "candidate_id": chosen["candidate_id"] if chosen else None}


def confusion(records):
    matrix = {p:{a:0 for a in ("satisfied","violated","unknown")} for p in ("satisfied","violated","unknown")}
    for record in records:
        matrix[record["prediction"]["status"]][record["execution"]["status"]] += 1
    predicted = sum(matrix["satisfied"].values())
    actual = sum(row["satisfied"] for row in matrix.values())
    return {"schema_version":"1.1.0", "matrix":matrix,
            "observed_false_acceptance":matrix["satisfied"]["violated"]/predicted if predicted else None,
            "predicted_satisfied_execution_unknown":matrix["satisfied"]["unknown"]/predicted if predicted else None,
            "false_rejection":matrix["violated"]["satisfied"]/actual if actual else None,
            "predicted_unknown_missed_feasible":matrix["unknown"]["satisfied"]/actual if actual else None,
            "legacy_not_satisfied_meaning":"not certified by execution; includes violated and unknown",
            "predicted_feasible_actual_not_satisfied":
                (matrix["satisfied"]["violated"]+matrix["satisfied"]["unknown"])/predicted if predicted else None,
            "predicted_infeasible_actual_satisfied":matrix["violated"]["satisfied"]/actual if actual else None,
            "predicted_unknown_actual_satisfied":matrix["unknown"]["satisfied"]/actual if actual else None,
            "predicted_feasible_denominator":predicted,"actual_feasible_denominator":actual}
