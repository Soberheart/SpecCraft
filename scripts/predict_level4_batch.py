"""L4-01 Phase B prediction-only Dyn rollouts for the 32+4 pool (job B-dyn).

Never reads any execution of a scored candidate.  For every listed candidate
and every tool-input variant it rolls the frozen Dyn checkpoint forward from the
shared initial latent over the pool schedule (13 times, 0..1.92 s every 0.16 s)
and decodes occupancy on the frozen query grid, exactly as level4.pipeline.predict:

* ``legacy_nominal`` - the pool's planner_input ``tools`` (Level 3 nominal tool
  point clouds, old execution semantics);
* ``v3_nominal`` - the same representation rebuilt from the v3 nominal execution
  (job 25837718 candidate_000 log ``ee`` summary) with the training sampler
  (level4.tool_observation); its construction is first validated by rebuilding
  the legacy sequence from the legacy log and comparing with the stored one.

Candidate tool input = transform_tools(variant nominal, grasp centre, action);
candidates' own executed trajectories are never used.  Outputs are hashed into
``prediction_manifest.json`` (the prediction-only freeze) before any analysis.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

import numpy as np

try:
    from prepare_level4_shared_input import assemble_runtime, digest, write
    from resume_level4_shared_input import environment, check_files
except ModuleNotFoundError:
    from scripts.prepare_level4_shared_input import assemble_runtime, digest, write
    from scripts.resume_level4_shared_input import environment, check_files

KIND = 'level4_batch_prediction_v1'
VARIANTS = ('legacy_nominal', 'v3_nominal')
FRAME_INTERVAL_S = 0.032


def validate_request(request):
    if request.get('kind') != KIND or request.get('candidate_pool_released') is not False:
        raise ValueError('Not a prediction-only Level 4 request')
    if request.get('new_simulations_max') != 0:
        raise ValueError('Prediction jobs never simulate')
    if tuple(request.get('variants', ())) != VARIANTS:
        raise ValueError('Both tool-input variants are required, in the frozen order')
    ids = [c['candidate_id'] for c in request['candidates']]
    if not ids or len(set(ids)) != len(ids) or any(len(c['action_mm_mm_deg']) != 3 for c in request['candidates']):
        raise ValueError('Invalid candidate list')
    if request['new_Dyn_rollouts'] != len(ids) * len(VARIANTS):
        raise ValueError('Declared Dyn rollout count differs from candidates x variants')
    if int(request['chunk_size']) < 1:
        raise ValueError('chunk_size must be positive')


def schedule_frames(times):
    frames = np.rint(np.asarray(times) / FRAME_INTERVAL_S).astype(int)
    if not np.allclose(frames * FRAME_INTERVAL_S, times, rtol=0, atol=1e-8) or frames[0] != 0:
        raise ValueError('Pool schedule is not on the logged 0.032 s frame grid')
    return frames.tolist()


def rollout(step, decode, initial_latent, initial_classes, tools):
    """Autoregressive latent rollout; step(pair, latent) and decode(latent) are injected."""
    latent, latents, classes = initial_latent, [initial_latent], [np.asarray(initial_classes)]
    for i in range(len(tools) - 1):
        latent = step(i, latent)
        latents.append(latent)
        classes.append(np.asarray(decode(latent)))
    return latents, classes


def build_tool_inputs(request, pool_inputs, output, o3d=None):
    """Return {variant: (T, N, 3) sim-unit points} and write their provenance."""
    from omegaconf import OmegaConf
    from sim.util import resolve_config
    from level4.tool_observation import compare_clouds, observed_tool_sequence
    times, legacy = pool_inputs['times_s'], pool_inputs['tools']
    frames = schedule_frames(times)
    process = Path('sim/process.py').resolve()
    check = request['legacy_validation']
    with Path(check['source_log']).open('rb') as stream:
        legacy_rows = pickle.load(stream)
    legacy_cfg = resolve_config(OmegaConf.load(check['scene_config']))
    rebuilt, provenance = observed_tool_sequence(legacy_rows, frames, legacy_cfg, process, request['sampling_seed'], o3d)
    agreement = compare_clouds(rebuilt, legacy)
    agreement.update(limits_mm={'bbox': check['max_bbox_mm'], 'centroid': check['max_centroid_mm']},
                     provenance=provenance)
    agreement['status'] = ('passed' if agreement['max_bbox_mm'] <= check['max_bbox_mm']
                           and agreement['max_centroid_mm'] <= check['max_centroid_mm'] else 'failed')
    write(output / 'tool_observation_validation.json', agreement)
    if agreement['status'] != 'passed':
        raise ValueError('Rebuilt legacy tool input disagrees with the stored model input; v3 input not trusted')
    reference = Path(request['v3_nominal']['evidence_dir'])
    manifest = json.loads((reference / 'evidence_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name in ('execution/candidate_000/scenes/0000/log.pkl', 'execution/candidate_000/requested_config.yaml'):
        if digest(reference / name) != manifest[name]:
            raise ValueError('v3 nominal evidence changed: ' + name)
    with (reference / 'execution/candidate_000/scenes/0000/log.pkl').open('rb') as stream:
        v3_rows = pickle.load(stream)
    v3_cfg = resolve_config(OmegaConf.load(reference / 'execution/candidate_000/requested_config.yaml'))
    v3, v3_provenance = observed_tool_sequence(v3_rows, frames, v3_cfg, process, request['sampling_seed'], o3d)
    inputs = {'legacy_nominal': np.asarray(legacy, dtype=np.float32), 'v3_nominal': v3}
    directory = output / 'tool_inputs'
    directory.mkdir()
    for name, value in inputs.items():
        with (directory / (name + '.npz')).open('xb') as stream:
            np.savez_compressed(stream, tools=value, times_s=times, frames=np.asarray(frames))
    write(directory / 'provenance.json', {'v3_nominal': v3_provenance, 'legacy_nominal': 'pool planner_input tools (unchanged)',
          'difference_v3_minus_legacy_mm': compare_clouds(v3, legacy)})
    return inputs


def run(request, tools, output):
    validate_request(request)
    write(output / 'environment.json', environment())
    if environment() != request['expected_compute_environment']:
        raise ValueError('Compute environment changed')
    check_files(request['required_files'])
    toolkit = json.loads((tools / 'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name, h in toolkit.items():
        if Path(name).name != name or digest(tools / name) != h:
            raise ValueError('Delivered toolkit file changed: ' + name)
    import os
    import torch
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not os.environ.get('PBS_JOBID') or not mask or ',' in mask or not torch.cuda.is_available() \
            or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires exactly one PBS-allocated GPU; no CPU fallback')
    runtime = assemble_runtime(request, tools, output.parent)
    from level4.artifacts import checked_pool, load_grid, save_packet, sha256, utc_now, write_json
    from scripts.check_level3_dyn import canonical_model_state
    from scripts.plan_level3_pose_cem import transform_tools
    from omegaconf import OmegaConf
    from net.model.model import Predictor
    pool = Path(request['source_pool'])
    manifest = checked_pool(pool)
    if sha256(pool / 'pool.json') != request['source_pool_sha256']:
        raise ValueError('Frozen pool differs')
    checkpoint = Path(manifest['checkpoint'])
    if sha256(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    with np.load(pool / 'planner_input.npz', allow_pickle=False) as packet:
        pool_inputs = {k: packet[k] for k in ('observed_initial', 'z_initial', 'tools', 'times_s', 'initial_classes')}
    inputs = build_tool_inputs(request, pool_inputs, output)
    cfg = OmegaConf.load(pool / 'model_config.yaml')
    OmegaConf.resolve(cfg)
    if cfg.model.condition.ee_shape != 'both' or cfg.dimensions.input_dim != 4:
        raise ValueError('Unexpected model tool interface')
    torch.manual_seed(manifest['observation_seed'])
    torch.cuda.manual_seed_all(manifest['observation_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    model = Predictor(cfg).to(device).eval()
    model.load_state_dict(canonical_model_state(torch.load(checkpoint, map_location='cpu')['model_state_dict']), strict=True)
    grid = load_grid(pool)
    if not np.isclose(float(cfg.augmentation.render.unit_to_meter), float(grid['unit_to_meter'])):
        raise ValueError('Model and grid units differ')
    centre = np.asarray(OmegaConf.load(pool / 'source_scene.yaml').actions.grasp.to_pos, dtype=float)
    chunk = int(request['chunk_size'])
    times = pool_inputs['times_s']
    hashes = {}
    with torch.no_grad():
        _, _, encoded = model.reconstruct(torch.as_tensor(pool_inputs['observed_initial'], device=device)[None], None, decode=False)
        if not np.allclose(encoded.cpu().numpy()[0], pool_inputs['z_initial'], rtol=1e-4, atol=1e-5):
            raise ValueError('Initial latent cache does not reproduce from the initial observation')
        def decode(latent):
            out = []
            for start in range(0, len(grid['query']), chunk):
                q = torch.as_tensor(grid['query'][start:start + chunk], dtype=latent.dtype, device=device)[None]
                logits, _ = model.decode(q, latent)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite occupancy logits')
                out.append(logits.argmax(-1).cpu().numpy()[0])
            return np.concatenate(out)
        for variant in VARIANTS:
            for row in request['candidates']:
                cid = row['candidate_id']
                target = output / 'predictions' / variant / cid
                target.mkdir(parents=True, exist_ok=False)
                started = time.monotonic()
                transformed = transform_tools(inputs[variant], centre, [row['action_mm_mm_deg']], float(grid['unit_to_meter']))
                action = torch.as_tensor(transformed, device=device)
                def step(i, latent):
                    _, _, nxt = model.predict(torch.stack([action[:, i], action[:, i + 1]], dim=0), None, latent, decode=False)
                    if not torch.isfinite(nxt).all():
                        raise ValueError('Nonfinite autoregressive latent')
                    return nxt
                latents, classes = rollout(step, decode, torch.as_tensor(pool_inputs['z_initial'], device=device)[None],
                                           pool_inputs['initial_classes'], times)
                latents = [np.asarray(l.cpu().numpy()[0]) if hasattr(l, 'cpu') else np.asarray(l)[0] for l in latents]
                save_packet(target / 'trajectory.npz', times, classes)
                with (target / 'latents.npz').open('xb') as stream:
                    np.savez_compressed(stream, z=np.stack(latents), times_s=times)
                with (target / 'tool_input.npz').open('xb') as stream:
                    np.savez_compressed(stream, tools=transformed[0, :, :, :3])
                write_json(target / 'complete.json', {'candidate_id': cid, 'variant': variant, 'status': 'predicted',
                           'action_mm_mm_deg': row['action_mm_mm_deg'], 'elapsed_s': time.monotonic() - started,
                           'dyn_calls': len(times) - 1, 'decoded_states': len(times) - 1, 'chunk_size': chunk,
                           'first_geometry_source': 'privileged initial reference; later states model decode',
                           'completed_utc': utc_now(), 'gpu': torch.cuda.get_device_name(0),
                           'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device)})
                hashes['%s/%s' % (variant, cid)] = {n: sha256(target / n) for n in
                                                     ('trajectory.npz', 'latents.npz', 'tool_input.npz', 'complete.json')}
    write_json(output / 'prediction_manifest.json', {
        'kind': 'prediction_only_freeze', 'created_utc': utc_now(), 'pool_sha256': request['source_pool_sha256'],
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'variants': list(VARIANTS),
        'tool_inputs_sha256': {v: sha256(output / 'tool_inputs' / (v + '.npz')) for v in VARIANTS},
        'predictions_sha256': hashes, 'execution_truth_read': False})
    return {'status': 'predictions_frozen_execution_truth_not_read', 'rollouts': len(hashes),
            'new_simulations': 0, 'candidate_pool_released': False}


def collect(request_path, output):
    output.mkdir(exist_ok=False)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    status, started = 0, time.monotonic()
    try:
        report = run(request, request_path.parent, output)
    except Exception as exc:
        status = 1
        report = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
        write(output / 'prediction_failure.json', report)
    write(output / 'prediction_report.json', report)
    write(output / 'request.json', request)
    write(output / 'stage_exit.json', {'exit_code': status, 'elapsed_s': time.monotonic() - started})
    write(output / 'evidence_manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): digest(p)
          for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
    print(json.dumps({k: report.get(k) for k in ('status', 'rollouts', 'error')}))
    return status


def preflight(request_path, output):
    """Login-node CPU check before queueing: rebuild both tool inputs, no model, no GPU.

    The compute-node Python differs slightly from the login node, so the frozen
    environment equality is checked only in the real job; files and the tool
    input construction (legacy agreement gate included) are checked here.
    """
    output.mkdir(parents=True, exist_ok=False)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    validate_request(request)
    check_files(request['required_files'])
    tools = request_path.parent
    toolkit = json.loads((tools / 'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name, h in toolkit.items():
        if digest(tools / name) != h:
            raise ValueError('Delivered toolkit file changed: ' + name)
    assemble_runtime(request, tools, output.parent)
    from level4.artifacts import checked_pool
    pool = Path(request['source_pool'])
    checked_pool(pool)
    with np.load(pool / 'planner_input.npz', allow_pickle=False) as packet:
        pool_inputs = {k: packet[k] for k in ('tools', 'times_s')}
    inputs = build_tool_inputs(request, pool_inputs, output)
    write(output / 'preflight_report.json', {'status': 'passed', 'tool_input_shapes': {k: list(v.shape) for k, v in inputs.items()},
          'tool_inputs_sha256': {k: digest(output / 'tool_inputs' / (k + '.npz')) for k in inputs}})
    print('PREFLIGHT_PASSED')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true', help='login-node CPU tool-input check only')
    args = parser.parse_args()
    if args.preflight:
        sys.exit(preflight(args.request.resolve(), args.output.resolve()))
    sys.exit(collect(args.request.resolve(), args.output.resolve()))
