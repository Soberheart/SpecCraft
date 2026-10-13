"""Paired scenario-level statistics. Search seeds never inflate scene count."""
import numpy as np


def paired_scene_bootstrap(rows,method,baseline,seed=0,repetitions=2000):
    groups={}
    for row in rows:
        if row["method"] not in (method,baseline):
            continue
        key=(row["initial_state_group"],row["task_family_id"])
        pair=groups.setdefault(key,{method:{},baseline:{}})
        search_seed=row["search_seed"]
        if search_seed in pair[row["method"]]:
            raise ValueError("Duplicate method/group/search-seed observation")
        if type(row["strict_success"]) is not bool:
            raise ValueError("Unknown, rejection and failures must be encoded as False, not dropped")
        pair[row["method"]][search_seed]=int(row["strict_success"])
    if not groups:
        raise ValueError("No paired scenario groups")
    # Cluster by initial state, including all task families from that state.
    scene_differences={}
    for (scene,task),pair in groups.items():
        if not pair[method] or set(pair[method])!=set(pair[baseline]):
            raise ValueError("Methods require paired search seeds in every group")
        delta=np.mean([pair[method][s]-pair[baseline][s] for s in pair[method]])
        scene_differences.setdefault(scene,[]).append(delta)
    values=np.array([np.mean(v) for v in scene_differences.values()])
    if repetitions<1:
        raise ValueError("Bootstrap repetitions must be positive")
    result={"paired_mean_success_difference":float(values.mean()),"independent_initial_states":len(values),
            "bootstrap_seed":seed,"repetitions":repetitions,"ci95":None,
            "scope":"equal weight initial states; within-state average over task groups and paired seeds"}
    if len(values)>=2:
        rng=np.random.default_rng(seed)
        means=values[rng.integers(len(values),size=(repetitions,len(values)))].mean(axis=1)
        result["ci95"]=np.quantile(means,[.025,.975]).tolist()
    else:
        result["reason"]="one independent initial state cannot support a scene-level confidence interval"
    return result
