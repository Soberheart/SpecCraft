"""Development reachability and errors; never chooses task thresholds."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from level4.artifacts import checked_pool,load_grid,load_trajectory,read_json,write_json,sha256
from level4.geometry import measure


def summarize(pool,output):
    pool,output=Path(pool),Path(output)
    manifest=checked_pool(pool);grid=load_grid(pool)
    rows=[];dense_rows=[];reachable=[]
    primary=manifest["config"]["primary_s"]
    for candidate in manifest["candidates"]:
        cid=candidate["candidate_id"];data={}
        for stage in ("predictions","executions"):
            directory=pool/stage/cid
            if not (directory/"complete.json").exists():
                data[stage]=[];continue
            meta=read_json(directory/"complete.json")
            if meta["pool_sha256"]!=sha256(pool/"pool.json") or meta["packet_sha256"]!=sha256(directory/"trajectory.npz"):
                raise ValueError("Changed development artifacts")
            data[stage]=[(t,{a:measure(g,manifest["config"]["axes"][a]) for a in ("u","v")})
                         for t,g in load_trajectory(directory/"trajectory.npz",grid)]
        common_times=[t for t,_ in data["predictions"] if t<=primary+1e-8]
        for t,predicted in data["predictions"]:
            matches=[actual for at,actual in data["executions"] if abs(at-t)<=1e-8]
            if len(matches)>1:raise ValueError("Ambiguous paired development time")
            for axis in ("u","v"):
                actual=matches[0][axis] if matches else {"value":None,"status":"not_executed"}
                p,a=predicted[axis]["value"],actual["value"]
                rows.append({"candidate_id":cid,"time_s":t,"axis":axis,"prediction_mm":p,"execution_mm":a,
                    "error_mm":p-a if p is not None and a is not None else None,
                    "prediction_status":predicted[axis]["status"],"execution_status":actual["status"]})
        for t,attributes in data["executions"]:
            for axis in ("u","v"):
                dense_rows.append({"candidate_id":cid,"time_s":t,"axis":axis,
                    "execution_mm":attributes[axis]["value"],"status":attributes[axis]["status"],
                    "reason":attributes[axis]["reason"],"primary_interval":t<=primary+1e-8})
        def value_at(t,axis):
            matches=[a[axis]["value"] for at,a in data["executions"] if abs(at-t)<=1e-8]
            return matches[0] if len(matches)==1 else None
        u0,v0,uT=value_at(0,"u"),value_at(0,"v"),value_at(primary,"u")
        dense=[a["v"]["value"] for t,a in data["executions"] if t<=primary+1e-8]
        common=[value_at(t,"v") for t in common_times]
        known=bool(dense and common and u0 and v0 and uT is not None and all(x is not None for x in dense+common))
        if known:
            dense_peak=max(abs(x/v0-1) for x in dense);common_peak=max(abs(x/v0-1) for x in common)
            reachable.append({"candidate_id":cid,"status":"measured","rho_observed":1-uT/u0,
                "eta_required_dense":dense_peak,"eta_required_common":common_peak,
                "dense_minus_common_peak":dense_peak-common_peak,
                "execution_dense_samples":len(dense),"common_samples":len(common)})
        else:reachable.append({"candidate_id":cid,"status":"unknown_or_not_executed"})
    output.mkdir(parents=True,exist_ok=False)
    for filename,table in (("paired_attributes.csv",rows),("dense_execution_attributes.csv",dense_rows)):
        with (output/filename).open("x",encoding="utf-8",newline="") as stream:
            if table:
                writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    errors=[r["error_mm"] for r in rows if r["error_mm"] is not None]
    write_json(output/"calibration_evidence.json",{"status":"development_summary_not_threshold_freeze",
        "pool_sha256":sha256(pool/"pool.json"),"paired_attributes_sha256":sha256(output/"paired_attributes.csv"),
        "dense_attributes_sha256":sha256(output/"dense_execution_attributes.csv"),"reachable_pairs":reachable,
        "unknown_attribute_rows":sum(r["error_mm"] is None for r in rows),
        "mae_mm":float(np.mean(np.abs(errors))) if errors else None,
        "max_absolute_error_mm":float(np.max(np.abs(errors))) if errors else None,
        "remaining":["nominal reset/step0 audit","grid 2/1/0.5mm and particle radius sensitivity",
                     "AE vs Dyn attribution","choose rho/eta/epsilon/m with separate evidence",
                     "freeze split, meaningful effect, cost cap and model reliability gate"],"thresholds_frozen":False})
    return {"status":"development_summary_saved","output":str(output)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(summarize(args.pool,args.output))
