"""Fixed-pool development pipeline; GPU imports occur only inside their stages.

    prepare -> predict -> independent simulator execute -> offline analyze

No execution packets or candidate future geometry enter prediction. Each
simulation uses a new subprocess and a copied scene/seed, never another action's
deformed final state. This implementation has not been run on CUDA yet.
"""
import copy
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
from .artifacts import (sha256,write_json,read_json,load_grid,save_packet,load_trajectory,
                        checked_pool,snapshot_runtime,verify_runtime,utc_now,runtime_files,runtime_environment,runtime_commit)
from .geometry import grid_coordinates,particle_occupancy,measure,from_occupancy,axis_vector
from .evaluate import evaluate
from .selection import select,confusion,point_goal_score
from .spec import validate_spec

ROOT=Path(__file__).resolve().parent.parent


def generate_actions(seed,count,bounds=(40.,40.,10.),std=(8.,8.,10.)):
    from scripts.plan_level3_pose_cem import bounded_gaussian
    if type(count) is not int or count<2 or count>32:
        raise ValueError("Development pool must contain 2..32 actions, nominal included")
    for values in (bounds,std):
        if np.asarray(values).shape!=(3,) or not np.isfinite(values).all() or (np.asarray(values)<=0).any():
            raise ValueError("Action bounds and sampling std must be three positive finite values")
    actions=np.vstack([np.zeros((1,3)),bounded_gaussian(np.random.default_rng(seed),count-1,
                                                     np.zeros(3),std,bounds)])
    if len(np.unique(actions,axis=0))!=count:
        raise ValueError("Duplicate candidate actions")
    return actions


def validate_pool_config(cfg):
    if cfg.get("schema_version")!="1.0.0" or cfg.get("mode")!="development_geometry_collection":
        raise ValueError("Only uncalibrated development geometry collection is supported")
    generate_actions(cfg["search_seed"],cfg["candidate_count"],cfg["bounds_mm_mm_deg"],cfg["initial_std_mm_mm_deg"])
    if (cfg["primary_frame"],cfg["primary_s"],cfg["auxiliary_frame"],cfg["auxiliary_s"])!=(60,1.920,62,1.984):
        raise ValueError("Development primary/auxiliary endpoints are frozen")
    if abs(axis_vector(cfg["axes"]["u"])@axis_vector(cfg["axes"]["v"]))>1e-10:
        raise ValueError("Development axes must be orthogonal")
    grid=cfg["grid"]
    for key in ("spacing_mm","particle_support_radius_mm"):
        if not np.isfinite(grid[key]) or grid[key]<=0:raise ValueError("Invalid representation parameter")
    padding=np.asarray(grid["initial_padding_mm"])
    if padding.shape!=(3,) or not np.isfinite(padding).all() or (padding<=0).any():
        raise ValueError("Invalid initial padding")
    if type(grid["max_cells"]) is not int or grid["max_cells"]<=0:raise ValueError("Invalid query budget")
    reset=cfg["initial_reset_max_error_mm"]
    if isinstance(reset,bool) or not np.isfinite(reset) or reset<=0:
        raise ValueError("Reset diagnostic tolerance must be positive finite mm")
    if cfg["material_thresholds"]!={"rho":None,"eta":None,"measurement_tolerance_ratio":None}:
        raise ValueError("Development collection does not freeze task thresholds")
    return cfg


def prepare(dyn_artifacts,output,config_path):
    """Development geometry collection only; does not assign material thresholds."""
    from omegaconf import OmegaConf
    from scripts.plan_level3_pose_cem import compose_yaw
    cfg=validate_pool_config(read_json(config_path))
    if cfg["mode"]!="development_geometry_collection":
        raise ValueError("Use a frozen task spec for task experiments; prepare collects development geometry only")
    source=Path(dyn_artifacts).resolve()
    metadata=read_json(source/"metadata.json")
    shared=metadata["kind"]=="shared_initial_observation_input_v1"
    if metadata["kind"]!="recorded_action_interface_diagnostic" and not shared:
        raise ValueError("Expected existing Level 3 Dyn artifacts")
    scene=Path(metadata["scene_dir"])
    checks=[(Path(metadata["checkpoint"]),metadata["checkpoint_sha256"]),
            (scene/"config.yaml",metadata["scene_config_sha256"])]
    if not shared: checks.append((scene/"log.pkl",metadata["source_log_sha256"]))
    for path,expected in checks:
        if sha256(path)!=expected:
            raise ValueError(f"Source hash mismatch: {path}")
    if shared:
        from .shared_input import checked_initial_input
        values=checked_initial_input(source,metadata)
        initial,z_initial,z_goal=(values[k] for k in ("observed_initial","z_initial","z_goal"))
        tools,times,particles=(values[k] for k in ("tools","times_s","initial_particles"))
        schedule=metadata["rollout_frames"]
    else:
        with np.load(source/"observations.npz",allow_pickle=False) as packet:
            frames=packet["frame_indices"].tolist()
            schedule=metadata["rollout_frames"]
            index=[frames.index(f) for f in schedule]
            times=packet["simulation_times"][index]
            # Explicit whitelist; never export intermediate object observations/latents.
            initial=packet["observed"][0]
            z_initial=packet["z_reference"][0]
            z_goal=packet["z_reference"][frames.index(schedule[-1])]
            tools=packet["ee_observed"][index]
    if schedule[-1]!=cfg["primary_frame"] or not np.isclose(times[-1],cfg["primary_s"],rtol=0,atol=1e-8):
        raise ValueError("Source planning endpoint differs from the declared primary physical time")
    if times[0]!=0 or (np.diff(times)<=0).any():
        raise ValueError("Source sequence must start at time zero and have ordered rollout times")
    scene_cfg=OmegaConf.load(scene/"config.yaml")
    OmegaConf.resolve(scene_cfg)
    if (scene_cfg.ee.type!="gripper" or list(scene_cfg.actions)!=["grasp"]
        or scene_cfg.actions.grasp.type!="grasp" or not np.allclose(scene_cfg.actions.grasp.from_offset,0)
        or "init_pos" in scene_cfg.actions.grasp or "init_quat" in scene_cfg.actions.grasp):
        raise ValueError("Requires the Level 3 stationary single-grasp tool/schedule")
    if not np.isclose(float(scene_cfg.log.n_iter)*float(scene_cfg.sim.step_dt)*cfg["auxiliary_frame"],cfg["auxiliary_s"],atol=1e-8):
        raise ValueError("Auxiliary physical time differs from scene logging")
    if not shared:
        with (scene/"log.pkl").open("rb") as stream:
            logs=pickle.load(stream)
    # Reject process.py's trailing-window/padded sequences. Frame 0 must be initial.
    if not shared and (len(logs)!=metadata["num_source_frames"] or logs[0]["step"]!=0):
        raise ValueError("Source has trailing-window/padded frames; cannot infer initial state")
    if not shared: particles=np.asarray(logs[0]["obj"]["particles"]["pos"],dtype=float)
    if particles.ndim!=2 or particles.shape[1]!=3 or not len(particles) or not np.isfinite(particles).all():
        raise ValueError("Invalid initial full material particles")
    scale=float(metadata["unit_to_meter"])
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("Invalid source coordinate scale")
    padding=np.asarray(cfg["grid"]["initial_padding_mm"])/(1000*scale)
    spacing=np.full(3,cfg["grid"]["spacing_mm"]/(1000*scale))
    bounds=np.column_stack([particles.min(axis=0)-padding,particles.max(axis=0)+padding])
    query,counts=grid_coordinates(bounds,spacing,cfg["grid"]["max_cells"])
    radius=cfg["grid"]["particle_support_radius_mm"]/(1000*scale)
    occupancy=particle_occupancy(query,particles,radius)
    initial_geo=from_occupancy(query,occupancy,counts,spacing,scale)
    reference={axis:measure(initial_geo,cfg["axes"][axis]) for axis in ("u","v")}
    if any(r["status"]!="measured" for r in reference.values()):
        raise ValueError("Initial geometry is unmeasurable/truncated")
    actions=generate_actions(cfg["search_seed"],cfg["candidate_count"],cfg["bounds_mm_mm_deg"],cfg["initial_std_mm_mm_deg"])
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    snapshot_hash=snapshot_runtime(ROOT,output/"runtime_snapshot")
    OmegaConf.save(scene_cfg,output/"source_scene.yaml")
    model_config=source/"model_config.yaml"
    (output/"model_config.yaml").write_bytes(model_config.read_bytes())
    with (output/"planner_input.npz").open("xb") as stream:
        np.savez_compressed(stream,observed_initial=initial,z_initial=z_initial,z_goal=z_goal,
                            tools=tools,times_s=times,initial_particles=particles,initial_classes=occupancy)
    with (output/"grid.npz").open("xb") as stream:
        np.savez_compressed(stream,query=query,counts=counts,spacing=spacing,unit_to_meter=scale)
    candidates=[]
    center=np.asarray(scene_cfg.actions.grasp.to_pos,dtype=float)
    for i,action in enumerate(actions):
        cid=f"candidate_{i:03d}"
        candidate=OmegaConf.create(OmegaConf.to_container(scene_cfg,resolve=True))
        position=center.copy()
        position[:2]+=action[:2]/(1000*scale)
        candidate.actions.grasp.to_pos=position.tolist()
        candidate.actions.grasp.to_quat=compose_yaw(scene_cfg.actions.grasp.to_quat,action[2])
        candidate.render=False
        candidate.cuda_id=0
        candidate.scene_id="0000"
        candidate.log.base_dir=str(output/"executions"/cid/"scenes")
        path=output/f"{cid}.yaml"
        OmegaConf.save(candidate,path)
        candidates.append({"candidate_id":cid,"parameters_mm_mm_deg":action.tolist(),
            "legal":True,"legality_scope":"finite_in_pose_bounds_only; physical collision checked by simulator",
            "action_cost":float(np.linalg.norm(action/np.asarray(cfg["bounds_mm_mm_deg"]))),
            "scene":path.name})
    (output/"pool_config.json").write_bytes(Path(config_path).read_bytes())
    (output/"source_metadata.json").write_bytes((source/"metadata.json").read_bytes())
    execution_times=(metadata["execution_required_s"] if shared else
                     [float(row["t"]) for row in logs if float(row["t"])<=cfg["primary_s"]+1e-8])
    if not execution_times or execution_times[0]!=0 or not np.isclose(execution_times[-1],cfg["primary_s"],rtol=0,atol=1e-8):
        raise ValueError("Source raw execution schedule does not cover primary interval")
    execution_times[-1]=cfg["primary_s"]
    files=["planner_input.npz","grid.npz","source_scene.yaml","model_config.yaml","pool_config.json",
           "source_metadata.json","runtime_snapshot/manifest.json"]+[r["scene"] for r in candidates]
    if shared:
        import shutil
        for name in ("shared_source_state.npz","static_material_info.npz","logged_frame0.npz","initialization_evidence.json"):
            shutil.copyfile(source/name,output/name)
            files.append(name)
    write_json(output/"pool.json",{"schema_version":"1.0.0","kind":"fixed_development_candidate_pool",
        "code_commit":runtime_commit(ROOT),
        "source_file_sha256":{str(p.relative_to(ROOT)):sha256(p) for folder in (ROOT/"level4",ROOT/"scripts")
                              for p in folder.glob("*.py")},
        "config":cfg,"config_sha256":sha256(config_path),"checkpoint":metadata["checkpoint"],
        "runtime_snapshot_sha256":snapshot_hash,"created_utc":utc_now(),"execution_required_s":execution_times,
        "checkpoint_sha256":metadata["checkpoint_sha256"],"source_metadata_sha256":sha256(source/"metadata.json"),
        "source_observations_sha256":sha256(source/"observations.npz"),
        "source_model_config_sha256":sha256(source/"model_config.yaml"),
        "observation_seed":metadata["seed"],"simulator_seed":int(scene_cfg.sim.seed),
        "bounds_mm_mm_deg":cfg["bounds_mm_mm_deg"],"initial_reference_mm":reference,
        "query_provenance":"initial privileged full material bbox plus fixed diagnostic padding; coverage unproven; no future material",
        "query_cell_count":len(query),"prediction_mode":"autoregressive_initial_only",
        "initialization_mode":"shared_saved_state" if shared else "independent_warmup",
        "initialization_scope":"new input prepared; candidate-specific shared reset not yet certified" if shared else "legacy logged frame0",
        "point_goal":"recorded frame 60 representative; not yet certified semantically feasible for task A/B",
        "candidates":candidates,"input_sha256":{f:sha256(output/f) for f in files}})
    return {"status":"prepared_not_predicted_or_executed","output":str(output),"candidate_count":len(actions)}


def predict(pool,chunk_size=4096):
    from scripts.check_level3_dyn import canonical_model_state
    from scripts.plan_level3_pose_cem import transform_tools
    from omegaconf import OmegaConf
    import torch
    from net.model.model import Predictor
    manifest=checked_pool(pool)
    pool=Path(pool)
    verify_runtime(ROOT,pool/"runtime_snapshot")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; prediction not run")
    checkpoint=Path(manifest["checkpoint"])
    if sha256(checkpoint)!=manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    cfg=OmegaConf.load(pool/"model_config.yaml")
    OmegaConf.resolve(cfg)
    if cfg.model.condition.ee_shape!="both" or cfg.dimensions.input_dim!=4:
        raise ValueError("Unexpected model tool interface")
    torch.manual_seed(manifest["observation_seed"])
    torch.cuda.manual_seed_all(manifest["observation_seed"])
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device("cuda:0")
    model=Predictor(cfg).to(device).eval()
    model.load_state_dict(canonical_model_state(torch.load(checkpoint,map_location="cpu")["model_state_dict"]),strict=True)
    grid=load_grid(pool)
    if not np.isclose(float(cfg.augmentation.render.unit_to_meter),float(grid["unit_to_meter"])):
        raise ValueError("Model and grid units differ")
    with np.load(pool/"planner_input.npz",allow_pickle=False) as packet:
        initial=packet["z_initial"]
        observed_initial=packet["observed_initial"]
        goal=packet["z_goal"]
        tools,times=packet["tools"],packet["times_s"]
        initial_classes=packet["initial_classes"]
    scene=OmegaConf.load(pool/"source_scene.yaml")
    if chunk_size<1:
        raise ValueError("chunk_size must be positive")
    with torch.no_grad():
        _,_,encoded=model.reconstruct(torch.as_tensor(observed_initial,device=device)[None],None,decode=False)
        if not np.allclose(encoded.cpu().numpy()[0],initial,rtol=1e-4,atol=1e-5):
            raise ValueError("Initial latent cache does not reproduce from initial observation")
        for record in manifest["candidates"]:
            output=pool/"predictions"/record["candidate_id"]
            if (output/"complete.json").exists():
                done=read_json(output/"complete.json")
                if (done["pool_sha256"]!=sha256(pool/"pool.json") or done["packet_sha256"]!=sha256(output/"trajectory.npz")
                    or done["latents_sha256"]!=sha256(output/"latents.npz") or done["chunk_size"]!=chunk_size):
                    raise ValueError("Prediction resume provenance mismatch")
                continue
            output.mkdir(parents=True,exist_ok=False)
            torch.cuda.reset_peak_memory_stats(device)
            started=time.monotonic()
            transformed=transform_tools(tools,np.asarray(scene.actions.grasp.to_pos),
                                        [record["parameters_mm_mm_deg"]],float(grid["unit_to_meter"]))
            action=torch.as_tensor(transformed,device=device)
            latent=torch.as_tensor(initial,device=device)[None]
            goal_tensor=torch.as_tensor(goal,device=device)[None]
            classes=[initial_classes]
            latents=[initial]
            for i in range(len(times)-1):
                _,_,latent=model.predict(torch.stack([action[:,i],action[:,i+1]],dim=0),None,latent,decode=False)
                if not torch.isfinite(latent).all():
                    raise ValueError("Nonfinite autoregressive latent")
                decoded=[]
                for start in range(0,len(grid["query"]),chunk_size):
                    q=torch.as_tensor(grid["query"][start:start+chunk_size],dtype=latent.dtype,device=device)[None]
                    logits,_=model.decode(q,latent)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite occupancy logits")
                    decoded.append(logits.argmax(-1).cpu().numpy()[0])
                classes.append(np.concatenate(decoded))
                latents.append(latent.cpu().numpy()[0])
            if (torch.linalg.vector_norm(latent,dim=-1)==0).any() or (torch.linalg.vector_norm(goal_tensor,dim=-1)==0).any():
                raise ValueError("Undefined latent cosine")
            score=float(torch.nn.functional.cosine_similarity(latent,goal_tensor,dim=-1).mean())
            save_packet(output/"trajectory.npz",times,classes)
            with (output/"latents.npz").open("xb") as stream:
                np.savez_compressed(stream,z=np.stack(latents),times_s=times)
            write_json(output/"complete.json",{"candidate_id":record["candidate_id"],"status":"predicted",
                "pool_sha256":sha256(pool/"pool.json"),"packet_sha256":sha256(output/"trajectory.npz"),
                "latents_sha256":sha256(output/"latents.npz"),"chunk_size":chunk_size,"completed_utc":utc_now(),
                "latent_score":score,"elapsed_s":time.monotonic()-started,
                "decoded_states":len(times)-1,"query_chunks_per_state":int(np.ceil(len(grid["query"])/chunk_size)),
                "dyn_calls":len(times)-1,"decode_calls":(len(times)-1)*int(np.ceil(len(grid["query"])/chunk_size)),
                "decoded_query_points":(len(times)-1)*len(grid["query"]),"cache_hit":False,
                "gpu_peak_allocated_bytes_including_model":torch.cuda.max_memory_allocated(device),
                "first_geometry_source":"privileged initial reference; subsequent states model decode",
                "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)})
    return {"status":"predictions_complete_execution_pending"}


def execute(pool,selection_path=None,candidate_ids=None,all_candidates=False):
    """Independent simulator subprocesses; retain failures and record exit codes."""
    manifest=checked_pool(pool)
    if manifest.get("initialization_mode")=="shared_saved_state":
        raise ValueError("Shared initial input is prepared, but candidate tool/material restore integration is not certified; do not replay legacy warmup")
    pool=Path(pool).resolve()
    verify_runtime(ROOT,pool/"runtime_snapshot")
    records=manifest["candidates"]
    if all_candidates and (selection_path is None or candidate_ids is not None):
        raise ValueError("All-candidate prospective diagnosis requires --selection and no candidate ID override")
    if selection_path is not None:
        selection=checked_selection(pool,selection_path)
        if not all_candidates:
            candidate_ids=[c["candidate_id"] for c in selection["selections"].values() if c["candidate_id"] is not None]
    if candidate_ids is not None:
        if set(candidate_ids)-{r["candidate_id"] for r in records}:
            raise ValueError("Unknown candidate IDs")
        records=[r for r in records if r["candidate_id"] in candidate_ids]
    grid=load_grid(pool)
    with np.load(pool/"planner_input.npz",allow_pickle=False) as packet:
        initial=packet["initial_particles"]
    config=manifest["config"]
    radius=config["grid"]["particle_support_radius_mm"]/(1000*float(grid["unit_to_meter"]))
    failed=[]
    for record in records:
        output=pool/"executions"/record["candidate_id"]
        if (output/"complete.json").exists():
            done=read_json(output/"complete.json")
            if (done["pool_sha256"]!=sha256(pool/"pool.json") or done["packet_sha256"]!=sha256(output/"trajectory.npz")
                or done["log_sha256"]!=sha256(output/"scenes/0000/log.pkl")):
                raise ValueError("Execution resume provenance mismatch")
            if selection_path and read_json(output/"exit.json").get("selection_sha256")!=sha256(selection_path):
                raise ValueError("Cannot retrospectively associate execution with a selection manifest")
            continue
        output.mkdir(parents=True,exist_ok=False)
        (output/"scenes").mkdir()
        command=[sys.executable,"-u",str(ROOT/"sim/generate.py"),"--config-path",str(pool),"--config-name",Path(record["scene"]).stem]
        started=time.monotonic()
        with (output/"generate.log").open("x",encoding="utf-8") as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        simulator_wall=time.monotonic()-started
        write_json(output/"exit.json",{"command":command,"exit_code":result.returncode,
                   "elapsed_s":simulator_wall,"pool_sha256":sha256(pool/"pool.json"),
                   "selection_sha256":sha256(selection_path) if selection_path else None,"completed_utc":utc_now()})
        if result.returncode:
            failed.append(record["candidate_id"])
            continue
        log_path=output/"scenes/0000/log.pkl"
        try:
            extraction_started=time.monotonic()
            with log_path.open("rb") as stream:
                logs=pickle.load(stream)
            positions=[np.asarray(row["obj"]["particles"]["pos"]) for row in logs]
            # step 0 is logged after one solver step; audit rather than assume a common reset.
            initial_error=float(np.max(np.abs(positions[0]-initial))) if positions[0].shape==initial.shape else None
            if initial_error is None or initial_error>config["initial_reset_max_error_mm"]/(1000*float(grid["unit_to_meter"])):
                raise ValueError("independent_initial_reset_mismatch")
            times=np.asarray([row["t"] for row in logs],dtype=float)
            if times[0]!=0 or (np.diff(times)<=0).any() or times[-1]<config["auxiliary_s"]-1e-8:
                raise ValueError("independent_execution_missing_or_irregular_times")
            primary_times=times[times<=config["primary_s"]+1e-8]
            if (len(primary_times)!=len(manifest["execution_required_s"])
                or not np.allclose(primary_times,manifest["execution_required_s"],rtol=0,atol=1e-8)):
                raise ValueError("independent_execution_changed_raw_sampling_schedule")
            classes=[particle_occupancy(grid["query"],p,radius) for p in positions]
            low,high=grid["query"].min(axis=0),grid["query"].max(axis=0)
            outside=[bool(((p.min(axis=0)-radius < low)|(p.max(axis=0)+radius > high)).any()) for p in positions]
            save_packet(output/"trajectory.npz",times,classes,outside)
            # Independent raw-particle span audit, not a replacement for voxel truth.
            spans={axis:[float(np.ptp(p@np.asarray(config["axes"][axis]))*1000*float(grid["unit_to_meter"])) for p in positions]
                   for axis in ("u","v")}
            write_json(output/"complete.json",{"candidate_id":record["candidate_id"],"status":"independently_simulated",
                "pool_sha256":sha256(pool/"pool.json"),"packet_sha256":sha256(output/"trajectory.npz"),
                "log_sha256":sha256(log_path),"initial_reset_max_abs_error_units":initial_error,
                "completed_utc":utc_now(),"simulator_wall_s":simulator_wall,
                "geometry_postprocess_wall_s":time.monotonic()-extraction_started,
                "raw_log_bytes":log_path.stat().st_size,"trajectory_bytes":(output/"trajectory.npz").stat().st_size,
                "truth_source":"all raw material particles, no tool/ground/contact mask, equal-radius-ball occupancy",
                "times_s":times.tolist(),"raw_particle_spans_mm":spans,
                "particle_ball_span_offset_mm":2*config["grid"]["particle_support_radius_mm"]})
        except Exception as exc:
            write_json(output/"failure.json",{"stage":"raw_truth_extraction","error":str(exc)})
            failed.append(record["candidate_id"])
    if failed:
        raise RuntimeError(f"Candidates failed and retained (use a new pool for retries): {failed}")
    return {"status":"independent_executions_complete"}


def prediction_decisions(pool,spec,*,experiment=True):
    """File-level online boundary: never opens executions or source future logs."""
    validate_spec(spec,experiment=experiment)
    manifest=checked_pool(pool)
    pool=Path(pool)
    if spec["process_hard"] and experiment:
        if spec["time"]["execution_required_s"]!=manifest["execution_required_s"]:
            raise ValueError("Task dense schedule differs from frozen pool schedule")
    grid=load_grid(pool)
    terminal_spec=copy.deepcopy(spec)
    terminal_spec["process_hard"]=[]
    records=[]
    hashes={}
    with np.load(pool/"planner_input.npz",allow_pickle=False) as packet:
        goal=packet["z_goal"]
    for candidate in manifest["candidates"]:
        directory=pool/"predictions"/candidate["candidate_id"]
        meta=read_json(directory/"complete.json")
        if meta["pool_sha256"]!=sha256(pool/"pool.json") or meta["packet_sha256"]!=sha256(directory/"trajectory.npz"):
            raise ValueError("Prediction provenance mismatch")
        predicted=load_trajectory(directory/"trajectory.npz",grid)
        with np.load(directory/"latents.npz",allow_pickle=False) as packet:
            indices=np.flatnonzero(np.abs(packet["times_s"]-spec["time"]["primary_s"])<=spec["time"]["match_tolerance_s"])
            if len(indices)!=1 or sha256(directory/"latents.npz")!=meta["latents_sha256"]:
                raise ValueError("Invalid primary latent packet")
            score=point_goal_score(packet["z"][int(indices[0])],goal)
        records.append({**candidate,"latent_score":score,
            "prediction":evaluate(spec,predicted,prediction=True,experiment=experiment),
            "terminal_prediction":evaluate(terminal_spec,predicted,prediction=True,experiment=experiment)})
        hashes[candidate["candidate_id"]]={name:sha256(directory/name) for name in ("complete.json","trajectory.npz","latents.npz")}
    selections={"PointGoal":select(records,"latent"),"Oracle-Constraint":select(records),
                "Oracle-TerminalOnly":select([{**r,"prediction":r["terminal_prediction"]} for r in records])}
    return records,selections,hashes


def freeze_selection(pool,spec_path,output,*,experiment=True):
    pool=Path(pool)
    if (pool/"executions").exists() and any((pool/"executions").iterdir()):
        raise ValueError("Execution already started; use explicitly exploratory analyze, not prospective freeze")
    spec=read_json(spec_path)
    records,selections,hashes=prediction_decisions(pool,spec,experiment=experiment)
    write_json(output,{"schema_version":"1.1.0","kind":"prediction_only_selection",
        "created_utc":utc_now(),"pool_sha256":sha256(pool/"pool.json"),"spec_sha256":sha256(spec_path),
        "spec":spec,"prediction_sha256":hashes,"selections":selections,"records":records,
        "prospective":True,"synthetic_only":not experiment,
        "point_goal_validity":"representative target not yet certified; research comparison blocked until reviewed"})
    return {"status":"selection_frozen","output":str(output),"sha256":sha256(output)}


def checked_selection(pool,path):
    pool=Path(pool)
    manifest=read_json(path)
    if manifest.get("kind")!="prediction_only_selection" or not manifest.get("prospective") or manifest.get("synthetic_only"):
        raise ValueError("Requires a prospective non-synthetic selection manifest")
    if manifest["pool_sha256"]!=sha256(pool/"pool.json"):
        raise ValueError("Selection pool changed")
    _,choices,hashes=prediction_decisions(pool,manifest["spec"])
    if choices!=manifest["selections"] or hashes!=manifest["prediction_sha256"]:
        raise ValueError("Frozen predictions or selections changed")
    return manifest


def audit_reset(pools,output):
    """Two completed nominal independent processes; no threshold adjustment."""
    if len(pools)<2:
        raise ValueError("Need at least two independent nominal resets")
    if len({Path(p).resolve() for p in pools})!=len(pools):
        raise ValueError("Repeated pool paths are not independent resets")
    rows=[];initials=[];reference=None
    for path in pools:
        pool=Path(path);manifest=checked_pool(pool)
        signature={k:manifest[k] for k in ("config_sha256","source_metadata_sha256","source_observations_sha256",
                                         "source_model_config_sha256","checkpoint_sha256","source_file_sha256","execution_required_s")}
        signature["runtime_files"]=read_json(pool/"runtime_snapshot/manifest.json")["files_sha256"]
        signature["runtime_environment"]=read_json(pool/"runtime_snapshot/manifest.json").get("environment")
        nominal=next(r for r in manifest["candidates"] if r["candidate_id"]=="candidate_000")
        if nominal["parameters_mm_mm_deg"]!=[0,0,0] or nominal["legal"] is not True:
            raise ValueError("Reset audit requires the nominal action")
        if reference is None:reference=signature
        if signature!=reference:raise ValueError("Reset probe inputs differ")
        directory=pool/"executions/candidate_000"
        meta=read_json(directory/"complete.json")
        if meta["pool_sha256"]!=sha256(pool/"pool.json") or meta["log_sha256"]!=sha256(directory/"scenes/0000/log.pkl"):
            raise ValueError("Reset probe provenance mismatch")
        if read_json(directory/"exit.json")["exit_code"]!=0:
            raise ValueError("Reset probe simulator did not exit successfully")
        with (directory/"scenes/0000/log.pkl").open("rb") as stream:
            log=pickle.load(stream)
        initial=np.asarray(log[0]["obj"]["particles"]["pos"])
        if log[0]["step"]!=0 or log[0]["t"]!=0 or not np.isfinite(initial).all():
            raise ValueError("Invalid nominal logged initial state")
        scale=float(load_grid(pool)["unit_to_meter"])
        initials.append(initial*1000*scale)
        rows.append({"pool":str(pool),"pool_sha256":sha256(pool/"pool.json"),"log_sha256":meta["log_sha256"],
            "initial_max_error_mm":meta["initial_reset_max_abs_error_units"]*1000*scale,
            "tolerance_mm":manifest["config"]["initial_reset_max_error_mm"]})
    error=max(float(np.max(np.abs(p-q))) if p.shape==q.shape else float("inf")
              for i,p in enumerate(initials) for q in initials[:i])
    passed=all(r["initial_max_error_mm"]<=r["tolerance_mm"] for r in rows) and error<=min(r["tolerance_mm"] for r in rows)
    write_json(output,{"status":"passed" if passed else "failed","kind":"nominal_initial_reset_probe",
        "created_utc":utc_now(),"source_signature":reference,"rows":rows,"pairwise_initial_max_error_mm":error if np.isfinite(error) else None,
        "scope":"logged step0 is after one solver step; candidate-dependent initial contact still needs whole-pool audit"})
    if not passed:raise ValueError("Nominal reset probe failed; retain logs and diagnose")
    return {"status":"nominal_reset_probe_passed","output":str(output)}


def verify_reset_gate(report_path,config_path,dyn_artifacts,*,root=ROOT):
    """Reuse a nominal reset only for unchanged actual inputs/source/environment."""
    report=read_json(report_path)
    rows=report.get("rows",[])
    if (report.get("status")!="passed" or report.get("kind")!="nominal_initial_reset_probe"
        or len(rows)<2 or len({Path(r["pool"]).resolve() for r in rows})!=len(rows)):
        raise ValueError("Requires a passed independent nominal reset audit")
    source=Path(dyn_artifacts);root=Path(root).resolve()
    metadata=read_json(source/"metadata.json")
    signature=report.get("source_signature",{})
    actual={"config_sha256":sha256(config_path),"source_metadata_sha256":sha256(source/"metadata.json"),
        "source_observations_sha256":sha256(source/"observations.npz"),
        "source_model_config_sha256":sha256(source/"model_config.yaml"),
        "checkpoint_sha256":sha256(metadata["checkpoint"]),
        "runtime_files":{p.relative_to(root).as_posix():sha256(p) for p in runtime_files(root)},
        "runtime_environment":runtime_environment()}
    for key,value in actual.items():
        if signature.get(key)!=value:
            raise ValueError("Reset evidence differs from current "+key)
    if actual["checkpoint_sha256"]!=metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from source metadata")
    return {"status":"passed","reset_audit_sha256":sha256(report_path),"current_signature":actual}


def analyze(pool,spec_path,output,selection_path=None):
    """Only this offline stage joins predicted and executed candidates."""
    spec=validate_spec(read_json(spec_path),experiment=True)
    manifest=checked_pool(pool)
    pool=Path(pool)
    grid=load_grid(pool)
    predictions,selections,_=prediction_decisions(pool,spec)
    frozen=checked_selection(pool,selection_path) if selection_path else None
    if frozen and frozen["spec_sha256"]!=sha256(spec_path):
        raise ValueError("Analysis spec differs from pre-execution spec")
    if frozen:
        selections=copy.deepcopy(frozen["selections"])
    records=[]
    for record in predictions:
        cid=record["candidate_id"]
        pred_dir=pool/"predictions"/cid
        actual_dir=pool/"executions"/cid
        pmeta=read_json(pred_dir/"complete.json")
        completed=(actual_dir/"complete.json").exists()
        ameta=read_json(actual_dir/"complete.json") if completed else None
        exit_record=read_json(actual_dir/"exit.json") if (actual_dir/"exit.json").exists() else None
        for directory,meta in [(pred_dir,pmeta)]+([(actual_dir,ameta)] if completed else []):
            if meta["pool_sha256"]!=sha256(pool/"pool.json") or meta["packet_sha256"]!=sha256(directory/"trajectory.npz"):
                raise ValueError("Analysis provenance mismatch")
        if completed and "log_sha256" in ameta and sha256(actual_dir/"scenes/0000/log.pkl")!=ameta["log_sha256"]:
            raise ValueError("Analysis raw material log changed")
        if frozen and completed and (not exit_record or exit_record.get("selection_sha256")!=sha256(selection_path)):
            raise ValueError("Execution was not associated with this prospective selection")
        actual=load_trajectory(actual_dir/"trajectory.npz",grid) if completed else []
        availability=("completed" if completed else "simulation_process_failed" if exit_record and exit_record["exit_code"]!=0
                      else "truth_extraction_failed" if (actual_dir/"failure.json").exists() else "not_executed")
        records.append({**record,"execution_availability":availability,
            "infrastructure_failure":availability=="simulation_process_failed",
            "failure_attribution":"unresolved_process_exit" if availability=="simulation_process_failed" else availability,
            "execution":evaluate(spec,actual,experiment=True),
            "common_schedule_execution":evaluate(spec,actual,experiment=True,common_schedule=True)})
    # Online selections use all predictions, including candidates with failed or
    # absent execution. Filtering by execution availability would leak truth.
    valid=records
    feasible=[r for r in valid if r["execution"]["strict_success"]]
    selections["Execution-Oracle pool reference"]={"candidate_id":min(feasible,key=lambda r:(r["action_cost"],r["candidate_id"]))["candidate_id"] if feasible else None,
        "offline_only":True}
    for method,choice in selections.items():
        selected=next((r for r in valid if r["candidate_id"]==choice["candidate_id"]),None)
        choice["actual_strict_success"]=bool(selected and selected["execution"]["strict_success"])
        choice["actual_status"]=selected["execution"]["status"] if selected else "not_selected"
        choice["common_schedule_actual_strict_success"]=bool(selected and selected["common_schedule_execution"]["strict_success"])
    common_records=[{**r,"execution":r["common_schedule_execution"]} for r in valid]
    write_json(output,{"kind":"fixed_pool_offline_diagnostic","pool_sha256":sha256(pool/"pool.json"),
        "spec_sha256":sha256(spec_path),"records":records,"confusion":confusion(valid),
        "common_schedule_confusion":confusion(common_records),
        "selection_timing":"prospective" if frozen else "exploratory_post_execution_not_formal",
        "selection_manifest_sha256":sha256(selection_path) if frozen else None,
        "selections":selections,"pool_size":len(records),"infrastructure_failure_count":sum(r["infrastructure_failure"] for r in records),
        "failure_count_by_availability":{state:sum(r["execution_availability"]==state for r in records)
            for state in ("completed","simulation_process_failed","truth_extraction_failed","not_executed")},
        "actual_feasible_count":len(feasible),"coverage_statement":"feasibility is assessed within this pool only",
        "independent_scenario_count":1,"generalization_claim":False})
    return {"status":"offline_analysis_complete","output":str(output)}
