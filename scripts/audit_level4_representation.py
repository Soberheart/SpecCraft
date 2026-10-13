"""Reprocess saved raw material logs on CPU; no new physical simulation."""
import argparse
import csv
from pathlib import Path
import pickle
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from level4.artifacts import checked_pool,load_grid,read_json,sha256,write_json
from level4.geometry import grid_coordinates,particle_occupancy,from_occupancy,measure


def audit(pool,output,spacings=(2.,1.,.5),radii=(2.,),candidate_ids=None,max_cells=32_000_000):
    pool,output=Path(pool),Path(output)
    manifest=checked_pool(pool);grid=load_grid(pool);scale=float(grid["unit_to_meter"])
    if not spacings or not radii or any(not np.isfinite(x) or x<=0 for x in list(spacings)+list(radii)):
        raise ValueError("Spacing and support radius must be positive mm")
    if candidate_ids and set(candidate_ids)-{c["candidate_id"] for c in manifest["candidates"]}:
        raise ValueError("Unknown representation-audit candidate IDs")
    bounds=np.column_stack([grid["query"].min(axis=0),grid["query"].max(axis=0)])
    rows=[];logs={};started=time.monotonic()
    for candidate in manifest["candidates"]:
        cid=candidate["candidate_id"]
        if candidate_ids and cid not in candidate_ids:continue
        directory=pool/"executions"/cid;path=directory/"scenes/0000/log.pkl"
        meta=read_json(directory/"complete.json")
        if meta["pool_sha256"]!=sha256(pool/"pool.json") or meta["log_sha256"]!=sha256(path):
            raise ValueError("Changed raw material logs")
        logs[cid]=sha256(path)
        with path.open("rb") as stream:raw=pickle.load(stream)
        for spacing in spacings:
            step=np.full(3,spacing/(1000*scale))
            # Align down so every reprocessed grid stays inside the original range.
            aligned=bounds.copy();aligned[:,1]=bounds[:,0]+np.floor((bounds[:,1]-bounds[:,0])/step)*step
            q,n=grid_coordinates(aligned,step,max_cells)
            low,high=q.min(axis=0),q.max(axis=0)
            for radius_mm in radii:
                radius=radius_mm/(1000*scale)
                for frame in raw:
                    t=float(frame["t"])
                    if t>manifest["config"]["auxiliary_s"]+1e-8:continue
                    p=np.asarray(frame["obj"]["particles"]["pos"],dtype=float)
                    c=particle_occupancy(q,p,radius)
                    outside=bool(((p.min(axis=0)-radius<low)|(p.max(axis=0)+radius>high)).any())
                    geometry=from_occupancy(q,c,n,step,scale,outside_support=outside)
                    for axis in ("u","v"):
                        raw_span=float(np.ptp(p@np.asarray(manifest["config"]["axes"][axis]))*1000*scale)
                        result=measure(geometry,manifest["config"]["axes"][axis])
                        rows.append({"candidate_id":cid,"time_s":t,"axis":axis,"spacing_mm":spacing,
                            "radius_mm":radius_mm,"raw_particle_span_mm":raw_span,
                            "analytic_ball_union_span_mm":raw_span+2*radius_mm,
                            "voxel_span_mm":result["value"],"status":result["status"],"reason":result["reason"],
                            "voxel_minus_ball_mm":result["value"]-raw_span-2*radius_mm if result["value"] is not None else None,
                            "query_cells":len(q),"outside_support":outside})
    if not rows:raise ValueError("No completed raw logs selected")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"representation.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/"manifest.json",{"kind":"development_representation_sensitivity_not_threshold_freeze",
        "pool_sha256":sha256(pool/"pool.json"),"raw_log_sha256":logs,"spacings_mm":list(spacings),"radii_mm":list(radii),
        "csv_sha256":sha256(output/"representation.csv"),"elapsed_cpu_wall_s":time.monotonic()-started,
        "grid_source":"same predeclared initial grid range; no future truth used to expand bounds",
        "model_fine_grid_decode":"not performed; required separately if prediction classification is resolution-sensitive"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--spacing-mm",type=float,nargs="+",default=[2,1,.5])
    parser.add_argument("--radius-mm",type=float,nargs="+",default=[2])
    parser.add_argument("--candidate-id",action="append");parser.add_argument("--max-cells",type=int,default=32_000_000)
    args=parser.parse_args();audit(args.pool,args.output,args.spacing_mm,args.radius_mm,args.candidate_id,args.max_cells)
