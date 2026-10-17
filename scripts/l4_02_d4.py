"""L4-02 B2: D4 evaluation (pre-registered in docs/level4_l4_02_D4_prereg.md).

Stages (run on NSCC; commands in the prereg):

    python -B scripts/l4_02_d4.py process [--submit]      # CPU PBS: official sim/process.py meshes for executed runs
    python -B scripts/l4_02_d4.py process-verify --dir D   # (inside that job) hash + shape check of data.h5
    python -B scripts/l4_02_d4.py eval-prepare [--submit]  # GPU PBS request (1 x A100): freezes every input hash
    python -B scripts/l4_02_d4.py eval --request R --out O # (inside the GPU job) raw metrics only
    python -B scripts/l4_02_d4.py report [--dir D]         # CPU: pre-registered verdicts from the raw metrics

The GPU stage never simulates, never retrains and only writes raw metrics; all
verdict rules live in level4/d4_metrics.py and are applied by ``report``.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.artifacts import read_json, sha256, write_json  # noqa: E402
from level4 import d4_metrics as dm  # noqa: E402
from scripts import analyze_level4_batch as az  # noqa: E402

KIND = 'level4_l4_02_d4_eval_v1'
L3_ARTIFACTS = Path('/home/users/ntu/mingjie0/projects/doughnet/records/level3_pointcloud/'
                    'sim_check_FhY2DG/dyn_check_ml9xhm/artifacts')
VENV = '/home/users/ntu/mingjie0/venvs/doughnet/bin/activate'
CANDIDATE_GROUPS = ('frozen_seed0', 'seed1_second_group', 'd2_edge')
FIELDS = ('scene', 'frame', 'obj_verts', 'obj_faces', 'obj_vert_labels', 'obj_face_labels', 'ee_verts', 'ee_faces',
          'ee_observed', 'genus')
ROLLOUT_FRAMES = list(range(0, 61, 5))


def utc():
    return datetime.now(timezone.utc).isoformat()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def qsub(name, parent, script, resources, walltime):
    with (parent / (name + '.pbs')).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(script)
    command = ['qsub', '-N', name, '-P', 'personal', '-q', 'normal', '-l', resources, '-l', 'walltime=' + walltime,
               '-j', 'oe', '-o', str(parent / 'pbs.log'), str(parent / (name + '.pbs'))]
    result = subprocess.run(command, capture_output=True, text=True)
    print('SUBMITTED ' + result.stdout.strip() if result.returncode == 0 else 'QSUB_FAILED ' + result.stderr.strip())
    return result.returncode


# ------------------------------------------------------------------ stage 1: official processing (CPU)
def executed_candidates(project):
    runs, pool, batch = az.discover(project)
    return {rid: r for rid, r in runs.items() if r['group'] in CANDIDATE_GROUPS}, pool, batch


def process(project, submit, workers=16):
    runs, _, batch = executed_candidates(project)
    for rid, run in sorted(runs.items()):
        scene = run['evidence'] / 'execution' / rid / 'scenes' / '0000'
        if not (scene / 'config.yaml').is_file():
            raise ValueError('Executed scene has no config.yaml (needed by sim/process.py): %s' % scene)
    parent = batch / ('D4-process_' + stamp())
    staging = parent / 'scenes'
    staging.mkdir(parents=True)
    for rid, run in sorted(runs.items()):
        log = az.verified(run['evidence'], 'execution/%s/scenes/0000/log.pkl' % rid)
        (staging / rid).symlink_to(log.parent, target_is_directory=True)
    names = sorted(runs)
    write_json(parent / 'run_map.json', {'rows': names, 'source_log_sha256': {
        rid: sha256(staging / rid / 'log.pkl') for rid in names}, 'created_utc': utc(),
        'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip(),
        'process_command': 'python -u sim/process.py --config-name common base_dir=<scenes> num_processes=%d num_frames=63'
                           % workers})
    script = '\n'.join(['#!/bin/bash', 'set -uo pipefail', 'source ' + VENV, 'cd %s' % REPO,
                        'echo "commit $(git rev-parse HEAD)"',
                        'python -u sim/process.py --config-name common base_dir=%s num_processes=%d num_frames=63'
                        % (staging, workers),
                        'echo $? > %s/process_exit_code.txt' % parent,
                        'python -B scripts/l4_02_d4.py process-verify --dir %s' % parent,
                        'echo $? > %s/job_exit_code.txt' % parent, ''])
    print('PROCESS_DIR', parent, 'runs', len(names))
    if not submit:
        with (parent / 'l4_d4_process.pbs').open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(script)
        return 0
    return qsub('l4_d4_process', parent, script, 'select=1:ncpus=%d:mem=110GB' % workers, '04:00:00')


def process_verify(parent):
    import hdf5plugin  # noqa: F401
    import h5py
    run_map = read_json(parent / 'run_map.json')
    path = parent / 'scenes' / 'data.h5'
    with h5py.File(path, 'r') as f:
        shape = {k: list(f[k].shape) for k in FIELDS}
        frames = f['frame'][:]
    ok = shape['frame'] == [len(run_map['rows']), 63] and bool((frames == np.arange(63)).all())
    write_json(parent / 'processed_manifest.json', {'data_h5': str(path), 'data_h5_sha256': sha256(path),
                                                     'shapes': shape, 'rows': run_map['rows'],
                                                     'status': 'passed' if ok else 'failed', 'verified_utc': utc()})
    print('PROCESS_VERIFY', 'passed' if ok else 'failed', shape['frame'])
    return 0 if ok else 1


# ------------------------------------------------------------------ stage 2: GPU request
def eval_prepare(project, submit, l3_artifacts=L3_ARTIFACTS):
    _, pool, batch = executed_candidates(project)
    processed = az.latest(batch, 'D4-process_*', 'processed_manifest.json')
    if processed is None or read_json(processed / 'processed_manifest.json')['status'] != 'passed':
        raise ValueError('Run and verify the D4 processing job first')
    dyn = az.latest(batch, 'B-dyn_*', 'data/prediction_manifest.json') / 'data'
    l3_meta = read_json(l3_artifacts / 'metadata.json')
    pool_manifest = read_json(pool / 'pool.json')
    prereg = REPO / 'docs/level4_l4_02_D4_prereg.md'
    files = {'processed_h5': processed / 'scenes' / 'data.h5', 'run_map': processed / 'run_map.json',
             'prediction_manifest': dyn / 'prediction_manifest.json', 'dyn_request': dyn / 'request.json',
             'pool_json': pool / 'pool.json', 'planner_input': pool / 'planner_input.npz',
             'model_config': pool / 'model_config.yaml', 'grid': pool / 'grid.npz',
             'checkpoint': Path(pool_manifest['checkpoint']),
             'l3_metadata': l3_artifacts / 'metadata.json', 'l3_metrics': l3_artifacts / 'metrics.csv',
             'l3_observations': l3_artifacts / 'observations.npz', 'l3_processed': Path(l3_meta['processed']),
             'prereg': prereg, 'd4_metrics': REPO / 'level4/d4_metrics.py', 'script': Path(__file__).resolve()}
    hashes = {k: sha256(v) for k, v in files.items()}
    if hashes['l3_processed'] != l3_meta['processed_sha256']:
        raise ValueError('L3 processed data changed since the L3 diagnostic')
    if hashes['checkpoint'] != pool_manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    parent = batch / ('D4-eval_' + stamp())
    parent.mkdir()
    request = {'kind': KIND, 'created_utc': utc(), 'new_simulations': 0, 'retraining': False,
               'files': {k: str(v) for k, v in files.items()}, 'sha256': hashes, 'dyn_dir': str(dyn),
               'l3_artifacts': str(l3_artifacts), 'observation_seed': int(pool_manifest['observation_seed']),
               'chunk_size': int(read_json(dyn / 'request.json')['chunk_size']),
               'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()}
    write_json(parent / 'request.json', request)
    script = '\n'.join(['#!/bin/bash', 'set -uo pipefail', 'source ' + VENV,
                        'export PYTHONDONTWRITEBYTECODE=1', 'cd %s' % REPO, 'echo "commit $(git rev-parse HEAD)"',
                        'nvidia-smi -L', 'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"',
                        'python -B -u scripts/l4_02_d4.py eval --request %s/request.json --out %s/data' % (parent, parent),
                        'echo $? > %s/eval_exit_code.txt' % parent,
                        'python -B -u scripts/l4_02_d4.py report --dir %s' % parent,
                        'echo $? > %s/job_exit_code.txt' % parent, ''])
    print('EVAL_DIR', parent)
    if not submit:
        with (parent / 'l4_d4_eval.pbs').open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(script)
        return 0
    return qsub('l4_d4_eval', parent, script, 'select=1:ngpus=1:ncpus=8:mem=64GB', '03:00:00')


# ------------------------------------------------------------------ stage 3: GPU evaluation (raw metrics)
def eval_main(request_path, out):
    request = read_json(request_path)
    if request.get('kind') != KIND or request['new_simulations'] != 0:
        raise ValueError('Not a D4 evaluation request')
    for key, path in request['files'].items():
        if sha256(path) != request['sha256'][key]:
            raise ValueError('Input changed since the request was frozen: ' + key)
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    import torch
    if not os.environ.get('PBS_JOBID') or not mask or ',' in mask or not torch.cuda.is_available() \
            or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires exactly one PBS-allocated GPU')
    import hdf5plugin  # noqa: F401
    import h5py
    from omegaconf import OmegaConf
    from net.model.model import Predictor
    from net.pipeline.evaluater import Evaluater
    from net.pipeline.pipeline import get_label
    from net.pipeline.random import manual_seed
    from net.pipeline.renderer import Renderer
    from scripts.check_level3_dyn import canonical_model_state
    out.mkdir(parents=True, exist_ok=False)
    files = {k: Path(v) for k, v in request['files'].items()}
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = request['observation_seed']
    manual_seed(seed)
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)
    cfg = OmegaConf.load(files['model_config'])
    OmegaConf.resolve(cfg)
    model = Predictor(cfg).to(device).eval()
    state = torch.load(files['checkpoint'], map_location='cpu')['model_state_dict']
    model.load_state_dict(canonical_model_state(state), strict=True)
    del state
    renderer = Renderer(OmegaConf.create(OmegaConf.to_container(cfg.augmentation.render, resolve=True)), device=device).eval()
    evaluator = Evaluater(cfg.loss).cuda(device=device)
    # exactly the validation query grid of check_level3_dyn.py / pipeline.py
    res = dm.OFFICIAL_RESOLUTION
    bounds = torch.tensor([[-0.30, 0.30], [-0.52, 0.52], [-res / 2, 0.11 + res / 2]], device=device)
    steps = ((bounds[:, 1] - bounds[:, 0]) / res).int() + 1
    axes = [torch.linspace(bounds[i, 0], bounds[i, 1], int(steps[i]), device=device) for i in range(3)]
    query = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(1, -1, 3)
    query_np = query[0].cpu().numpy()
    below = query[0, :, 2] < 0
    np.savez_compressed(out / 'official_query.npz', query=query_np, steps=steps.cpu().numpy())

    def mesh(data, frame):
        return {k: torch.as_tensor(data[k][frame], device=device)[None] for k in
                ('obj_verts', 'obj_faces', 'obj_vert_labels', 'obj_face_labels', 'ee_verts', 'ee_faces', 'genus')}

    def truth(m):
        true_part = get_label(query, m['obj_verts'], m['obj_faces'], m['obj_face_labels'])
        in_tool = get_label(query, m['ee_verts'], m['ee_faces'], torch.ones_like(m['ee_faces'][..., :1])).squeeze(-1).bool()
        masked = true_part[0, :, 0].clone()
        masked[in_tool[0] | below] = 0
        return true_part, in_tool, masked.cpu().numpy().astype(np.int8)

    def score(logits, genus_logits, t):
        true_part, in_tool, _ = t
        genus_pred = genus_logits.argmax(dim=-1) - 1
        ev = {'query': query, 'obj_true_part_nxt': true_part, 'obj_predicted_part_nxt': logits,
              'ee_true_part_nxt': in_tool, 'genus_nxt': m_genus[0], 'predicted_genus_nxt': genus_pred}
        metrics, _ = evaluator.accuracy(ev, 'pre', '_nxt')
        probabilities = torch.softmax(logits[0], dim=-1)
        cls = probabilities.argmax(dim=-1)
        cls[in_tool[0] | below] = 0
        return ({k: float(metrics['pre_%s_nxt' % k]) * 100 for k in ('viou', 'ciou', 'accc', 'accg')},
                cls.cpu().numpy().astype(np.int8))

    def step(latent, tools_pair):
        _, _, nxt = model.predict(tools_pair, None, latent, decode=False)
        return nxt

    def pair(tools, s, t):
        x = torch.as_tensor(np.asarray(tools)[[s, t]], device=device, dtype=torch.float32)[:, None]
        if x.shape[-1] == 3:
            x = torch.cat([x, torch.zeros_like(x[..., :1])], dim=-1)
        return x

    def observe(data, frame):
        manual_seed(seed + frame)
        return renderer(mesh(data, frame), postfix='', num_samples=cfg.dataset.n_points)

    m_genus = [None]
    results = {'kind': KIND, 'started_utc': utc(), 'gpu': torch.cuda.get_device_name(0)}
    with torch.no_grad():
        # ---------------- positive control: B-dyn code path on the L3 scene and its stored inputs
        with h5py.File(files['l3_processed'], 'r') as f:
            l3 = {k: f[k][0] for k in FIELDS}
        obs = np.load(files['l3_observations'])
        frame_index = {int(f): i for i, f in enumerate(obs['frame_indices'])}
        recorded = {(r['mode'], int(r['target_frame'])): r for r in
                    __import__('csv').DictReader(files['l3_metrics'].open(encoding='utf-8'))}
        l3_obs = {f: torch.as_tensor(obs['observed'][i], device=device)[None] for f, i in frame_index.items()}
        _, _, z0 = model.reconstruct(l3_obs[0], None, decode=False)
        z_ref_abs = float(np.abs(z0[0].cpu().numpy() - obs['z_reference'][frame_index[0]]).max())
        pc = {'z0_vs_l3_reference_max_abs': z_ref_abs, 'intervals': []}
        latent, latent_abs = z0, 0.0
        for s, t in zip(ROLLOUT_FRAMES[:-1], ROLLOUT_FRAMES[1:]):
            m = mesh(l3, t)
            m_genus[0] = m['genus']
            tr = truth(m)
            latent = step(latent, pair(l3['ee_observed'], s, t))
            stored = np.load(Path(request['l3_artifacts']) / ('rollout_%03d.npz' % t))
            latent_abs = max(latent_abs, float(np.abs(latent[0].cpu().numpy() - stored['z_predicted']).max()))
            logits, g = model.decode(query, latent)
            roll, roll_cls = score(logits, g, tr)
            _, _, zs = model.reconstruct(l3_obs[s], None, decode=False)
            logits, g = model.decode(query, step(zs, pair(l3['ee_observed'], s, t)))
            single, _ = score(logits, g, tr)
            pc['intervals'].append({'source': s, 'target': t, 'rollout': roll, 'single_step': single,
                                    'recorded_rollout_viou': float(recorded[('rollout', t)]['viou']),
                                    'recorded_single_viou': float(recorded[('single_step', t)]['viou'])})
            if t == 60:
                np.savez_compressed(out / 'l3_rollout_f60_classes.npz', pred=roll_cls, truth=tr[2])
        pc['latent_vs_l3_stored_max_abs'] = latent_abs
        # same scene, the pool's L4 inputs (v3 reset observation + legacy nominal tools): informative
        planner = np.load(files['planner_input'])
        latent = torch.as_tensor(planner['z_initial'], device=device)[None]
        pc_b = []
        for i, t in enumerate(ROLLOUT_FRAMES[1:]):
            m = mesh(l3, t)
            m_genus[0] = m['genus']
            latent = step(latent, pair(planner['tools'], i, i + 1))
            logits, g = model.decode(query, latent)
            pc_b.append({'target': t, **score(logits, g, truth(m))[0]})
        pc['pool_inputs_on_l3_truth'] = pc_b
        obs_stats = {'l3_frame0': np.asarray(obs['observed'][frame_index[0]])[:, :3],
                     'pool_observed_initial': np.asarray(planner['observed_initial'])[:, :3]}
        results['positive_control'] = pc
        write_json(out / 'positive_control.json', pc)
        print('PC rollout mean', np.mean([r['rollout']['viou'] for r in pc['intervals']]),
              'single mean', np.mean([r['single_step']['viou'] for r in pc['intervals']]), 'latent', latent_abs, flush=True)
        # ---------------- AE decode of the frozen initial latent on the L4 grid (Z2)
        grid = np.load(files['grid'])
        z_init = torch.as_tensor(planner['z_initial'], device=device)[None]
        cls = []
        for start in range(0, len(grid['query']), request['chunk_size']):
            q = torch.as_tensor(grid['query'][start:start + request['chunk_size']], dtype=z_init.dtype, device=device)[None]
            logits, _ = model.decode(q, z_init)
            cls.append(logits.argmax(-1).cpu().numpy()[0])
        np.savez_compressed(out / 'ae_k0_l4_grid_classes.npz', classes=np.concatenate(cls).astype(np.int8))
        # ---------------- L4 executed candidates
        run_map = read_json(files['run_map'])
        prediction_manifest = read_json(files['prediction_manifest'])
        dyn = Path(request['dyn_dir'])
        with h5py.File(files['processed_h5'], 'r') as f:
            for row, rid in enumerate(run_map['rows']):
                data = {k: f[k][row] for k in FIELDS}
                entry = {'run_id': rid, 'rollout_frozen': {}, 'truth_masked_attributes': {}}
                truths = {}
                for t in ROLLOUT_FRAMES:
                    m = mesh(data, t)
                    truths[t] = (truth(m), m['genus'])
                masked_truth = {t: truths[t][0][2] for t in ROLLOUT_FRAMES}
                for variant in prediction_manifest['variants']:
                    key = '%s/%s' % (variant, rid)
                    path = dyn / 'predictions' / variant / rid / 'latents.npz'
                    if sha256(path) != prediction_manifest['predictions_sha256'][key]['latents.npz']:
                        raise ValueError('Frozen latent changed: ' + key)
                    z = np.load(path)['z']
                    rows, classes = [], {}
                    for k, t in enumerate(ROLLOUT_FRAMES):
                        m_genus[0] = truths[t][1]
                        logits, g = model.decode(query, torch.as_tensor(z[k], device=device)[None])
                        metrics, c = score(logits, g, truths[t][0])
                        rows.append(dict(target=t, **metrics))
                        classes[t] = c
                    entry['rollout_frozen'][variant] = rows
                    np.savez_compressed(out / ('%s_%s_classes.npz' % (rid, variant)),
                                        **{'f%02d' % t: classes[t] for t in ROLLOUT_FRAMES})
                np.savez_compressed(out / ('%s_truth_classes.npz' % rid), **{'f%02d' % t: masked_truth[t] for t in ROLLOUT_FRAMES})
                # executed observations: AE, single step and recorded-tool rollout (secondary)
                zs = {}
                for t in ROLLOUT_FRAMES:
                    o = observe(data, t)
                    if t == 0 and rid == 'candidate_000':
                        obs_stats['l4_c000_rendered_frame0'] = o[0, :, :3].cpu().numpy()
                    _, _, zs[t] = model.reconstruct(o, None, decode=False)
                ae, single, rec = [], [], []
                latent = zs[0]
                for t in ROLLOUT_FRAMES:
                    m_genus[0] = truths[t][1]
                    logits, g = model.decode(query, zs[t])
                    ae.append(dict(target=t, **score(logits, g, truths[t][0])[0]))
                for s, t in zip(ROLLOUT_FRAMES[:-1], ROLLOUT_FRAMES[1:]):
                    m_genus[0] = truths[t][1]
                    logits, g = model.decode(query, step(zs[s], pair(data['ee_observed'], s, t)))
                    single.append(dict(target=t, **score(logits, g, truths[t][0])[0]))
                    latent = step(latent, pair(data['ee_observed'], s, t))
                    logits, g = model.decode(query, latent)
                    rec.append(dict(target=t, **score(logits, g, truths[t][0])[0]))
                entry.update(ae=ae, single_step=single, rollout_recorded_tools=rec)
                write_json(out / ('%s_metrics.json' % rid), entry)
                print('CANDIDATE', rid, 'v3 f60', entry['rollout_frozen']['v3_nominal'][-1]['viou'],
                      'single mean', np.mean([r['viou'] for r in single]), flush=True)
    write_json(out / 'observation_stats.json', {k: {'z_min_mm': float(v[:, 2].min() * dm.SIM_TO_MM),
                                                    'z_q01_mm': float(np.quantile(v[:, 2], 0.01) * dm.SIM_TO_MM),
                                                    'z_max_mm': float(v[:, 2].max() * dm.SIM_TO_MM),
                                                    'centroid_mm': (v.mean(axis=0) * dm.SIM_TO_MM).tolist(),
                                                    'xy_min_mm': (v[:, :2].min(axis=0) * dm.SIM_TO_MM).tolist(),
                                                    'xy_max_mm': (v[:, :2].max(axis=0) * dm.SIM_TO_MM).tolist()}
                                                for k, v in obs_stats.items()})
    results['finished_utc'] = utc()
    write_json(out / 'eval_complete.json', results)
    return 0


# ------------------------------------------------------------------ stage 4: pre-registered verdicts (CPU)
L3_P4_ABS_MM = {'span_u_mm': 16.0, 'span_v_mm': 10.0, 'neck_mm': 1.0}   # frozen in the prereg (rollout_060.npz)


def yaw_for(rid, plan):
    return dm.NOMINAL_YAW_DEG + float(plan[rid]['parameters_mm_mm_deg'][2])


def verdicts(pc, candidates, query, plan, ae_k0_hist, rollout_hist, obs_stats, exec_attributes_a1=None):
    """Pure function: raw metrics -> every pre-registered verdict."""
    pc_status = dm.positive_control([r['rollout']['viou'] for r in pc['intervals']],
                                    [r['single_step']['viou'] for r in pc['intervals']],
                                    max(pc['latent_vs_l3_stored_max_abs'], pc['z0_vs_l3_reference_max_abs']))
    v3 = {rid: c['rollout_frozen']['v3_nominal'] for rid, c in candidates.items()}
    f60 = {rid: rows[-1]['viou'] for rid, rows in v3.items()}
    mean = {rid: float(np.mean([r['viou'] for r in rows[1:]])) for rid, rows in v3.items()}
    single_mean = {rid: float(np.mean([r['viou'] for r in c['single_step']])) for rid, c in candidates.items()}
    single_all = [r['viou'] for c in candidates.values() for r in c['single_step']]
    original = dm.p_original(f60['candidate_000'], list(f60.values()), single_all)
    corrected = dm.p_corrected(mean['candidate_000'], f60['candidate_000'], list(mean.values()), list(f60.values()),
                               list(single_mean.values()))
    errors = {k: [] for k in L3_P4_ABS_MM}
    pred_attr, exec_attr = {}, {}
    for rid, c in candidates.items():
        a_pred = c['attributes']['v3_nominal_f60']
        a_true = c['attributes']['truth_f60']
        pred_attr[rid], exec_attr[rid] = c['attributes'], a_true
        for k in errors:
            if a_pred[k] is not None and a_true[k] is not None:
                errors[k].append(abs(a_pred[k] - a_true[k]))
    p4 = dm.p4(errors, L3_P4_ABS_MM)
    ids = sorted(candidates)
    sens = {}
    for name in ('neck_mm', 'span_u_mm', 'span_v_mm'):
        sens[name] = dm.sensitivity([pred_attr[i]['v3_nominal_f60'][name] for i in ids],
                                    [pred_attr[i]['truth_f60'][name] for i in ids])
    sens['dv_max_mm'] = dm.sensitivity([pred_attr[i]['v3_nominal_dv_max']['value'] for i in ids],
                                       [pred_attr[i]['truth_dv_max']['value'] for i in ids])
    z = dm.z_checks(rollout_hist, ae_k0_hist, obs_stats['l4_c000_rendered_frame0']['z_min_mm'],
                    obs_stats['l3_frame0']['z_min_mm'])
    out = {'positive_control': pc_status, 'P_original': original, 'P_corrected': corrected, 'P4': p4,
           'D4_original': dm.d4_verdict(original, p4), 'D4_corrected': dm.d4_verdict(corrected, p4),
           'z_checks': z, 'action_sensitivity': sens,
           'per_candidate': {rid: {'rollout_f60': f60[rid], 'rollout_mean': mean[rid], 'single_mean': single_mean[rid]}
                             for rid in sorted(candidates)}}
    out['route_original'] = dm.route(out['D4_original'], pc_status['status'])
    out['route_corrected'] = dm.route(out['D4_corrected'], pc_status['status'])
    return out


def candidate_attributes(data_dir, rid, yaw, query):
    """Masked official-grid attributes (pred v3/legacy + truth) and dv_max over the 13 schedule times."""
    truth = np.load(data_dir / ('%s_truth_classes.npz' % rid))
    out = {}
    v0 = dm.attributes(truth['f00'], query, yaw)['span_v_mm']
    for name, packet in (('truth', truth), ('v3_nominal', np.load(data_dir / ('%s_v3_nominal_classes.npz' % rid))),
                         ('legacy_nominal', np.load(data_dir / ('%s_legacy_nominal_classes.npz' % rid)))):
        out[name + '_f60'] = dm.attributes(packet['f60'], query, yaw)
        spans = [dm.attributes(packet['f%02d' % t], query, yaw)['span_v_mm'] for t in ROLLOUT_FRAMES[1:]]
        out[name + '_dv_max'] = {'value': (max(s for s in spans if s is not None) - v0)
                                 if v0 is not None and any(s is not None for s in spans) else None}
    return out


def report(parent, project):
    data = parent / 'data'
    if not (data / 'eval_complete.json').is_file():
        raise ValueError('GPU evaluation did not complete: %s' % parent)
    request = read_json(parent / 'request.json')
    query = np.load(data / 'official_query.npz')['query']
    plan = {r['candidate_id']: r for r in read_json(REPO / 'configs/level4/l4_01/pool32_manifest.json')['candidates']}
    run_map = read_json(Path(request['files']['run_map']))
    candidates = {}
    for rid in run_map['rows']:
        c = read_json(data / ('%s_metrics.json' % rid))
        c['attributes'] = candidate_attributes(data, rid, yaw_for(rid, plan), query)
        candidates[rid] = c
    pc = read_json(data / 'positive_control.json')
    l3 = np.load(data / 'l3_rollout_f60_classes.npz')
    l3_check = {k: abs((dm.attributes(l3['pred'], query, 30.0)[k] or 0) - (dm.attributes(l3['truth'], query, 30.0)[k] or 0))
                for k in L3_P4_ABS_MM}
    grid = np.load(Path(request['files']['grid']))
    ae_hist = dm.subground_histogram(np.load(data / 'ae_k0_l4_grid_classes.npz')['classes'], grid['query'],
                                     float(grid['unit_to_meter']))
    dyn = Path(request['dyn_dir'])
    total = {'occupied': 0, 'below_zero': 0, 'bins': {}}
    for rid in run_map['rows']:
        with np.load(dyn / 'predictions' / 'v3_nominal' / rid / 'trajectory.npz') as packet:
            for k in range(1, packet['classes'].shape[0]):
                h = dm.subground_histogram(packet['classes'][k], grid['query'], float(grid['unit_to_meter']))
                total['occupied'] += h['occupied']
                total['below_zero'] += h['below_zero']
                for b, n in h['bins'].items():
                    total['bins'][b] = total['bins'].get(b, 0) + n
    outside = total['bins'].get('below_-12.5', 0) + total['bins'].get('-12.5_to_-7.5', 0)
    total['below_zero_share'] = total['below_zero'] / total['occupied'] if total['occupied'] else None
    total['share_outside_guaranteed_domain'] = outside / total['below_zero'] if total['below_zero'] else None
    obs_stats = read_json(data / 'observation_stats.json')
    result = verdicts(pc, candidates, query, plan, ae_hist, total, obs_stats)
    result.update(kind='L4-02_D4_report', created_utc=utc(), eval_dir=str(parent),
                  l3_p4_recomputed_abs_mm=l3_check, l3_p4_frozen_abs_mm=L3_P4_ABS_MM,
                  subground_v3_rollout_L4_grid=total, subground_ae_k0_L4_grid=ae_hist, observation_stats=obs_stats,
                  positive_control_raw=pc)
    write_json(parent / 'd4_report.json', result)
    lines = ['D4_REPORT %s' % parent,
             'PC %s rollout_mean=%.2f (L3 %.2f) single_mean=%.2f (L3 %.2f) latent_max_abs=%.2e' % (
                 result['positive_control']['status'], result['positive_control']['rollout_mean'], dm.L3_ROLLOUT_MEAN,
                 result['positive_control']['single_mean'], dm.L3_SINGLE_MEAN, result['positive_control']['latent_max_abs']),
             'P_original %s -> D4=%s | %s' % ({k: result['P_original'][k] for k in ('P1', 'P2', 'P3')},
                                             result['D4_original'], result['P_original']['values']),
             'P_corrected %s -> D4=%s | %s' % ({k: result['P_corrected'][k] for k in ('P1', 'P2', 'P3')},
                                              result['D4_corrected'], result['P_corrected']['values']),
             'P4 %s %s' % (result['P4']['P4'], {k: (v.get('status'), round(v.get('ratio', float('nan')), 3))
                                                 for k, v in result['P4']['rows'].items()}),
             'Z %s' % {k: result['z_checks'][k] for k in ('Z1', 'Z1_share_outside_guaranteed_domain', 'Z2',
                                                          'Z2_ae_k0_below_zero_share', 'Z3', 'Z3_obs_zmin_mm')},
             'SENS %s' % {k: (v['status'], round(v.get('var_ratio_pred_over_exec', float('nan')), 3),
                              round(v.get('spearman_rho', float('nan')), 3)) for k, v in result['action_sensitivity'].items()},
             'ROUTE original: %s' % result['route_original'], 'ROUTE corrected: %s' % result['route_corrected']]
    with (parent / 'd4_summary.txt').open('w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('process')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--workers', type=int, default=16)
    v = sub.add_parser('process-verify')
    v.add_argument('--dir', type=Path, required=True)
    e = sub.add_parser('eval-prepare')
    e.add_argument('--submit', action='store_true')
    g = sub.add_parser('eval')
    g.add_argument('--request', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('report')
    r.add_argument('--dir', type=Path)
    for x in (p, e, r):
        x.add_argument('--project', type=Path, default=az.PROJECT)
    args = parser.parse_args(argv)
    if args.command == 'process':
        return process(args.project, args.submit, args.workers)
    if args.command == 'process-verify':
        return process_verify(args.dir)
    if args.command == 'eval-prepare':
        return eval_prepare(args.project, args.submit)
    if args.command == 'eval':
        return eval_main(args.request, args.out)
    parent = args.dir or az.latest(args.project / az.BATCH_DIRNAME, 'D4-eval_*', 'data/eval_complete.json')
    return report(parent, args.project)


if __name__ == '__main__':
    sys.exit(main())
