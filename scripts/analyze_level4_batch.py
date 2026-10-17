"""L4-01 Phase C analysis of the Phase B batch (CPU only; run on NSCC).

    python scripts/analyze_level4_batch.py submit            # PBS CPU job (recommended)
    python scripts/analyze_level4_batch.py run --out <dir> [--workers 8]

Stage 1 (truth, resumable): for every executed run (B-pilot, B-exec-*, plus the
job-25837718 candidate_000/001), read the hash-verified raw material log and
compute, from raw material particles only: voxel occupancy on the frozen pool
grid (2 mm balls, as level4.pipeline.execute), voxel u/v spans at all 63 frames,
raw-particle u/v spans, the pinch-frame neck at every frame, and connected
components (r0 = frame0 percolation radius) at frames 0/26/32/40/50/60/62.

Stage 2 (report): joins the frozen B-dyn predictions (both tool-input variants)
with the truth for C2 (D1), C3 (D2), C4 (S0-S3a), C5 (3x3 S/V/U over a threshold
sweep, development exploration only), the S4 candidate rule and a D4
*diagnostic* voxel IoU (not the pre-registered D4 metric).  Every threshold of
every sweep is reported.  Prediction files are hash-checked against
prediction_manifest.json; truth never feeds back into any selection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.artifacts import load_grid, load_trajectory, read_json, save_packet, sha256, write_json  # noqa: E402
from level4.geometry import from_occupancy, measure, particle_occupancy  # noqa: E402
from level4 import batch_analysis as ba  # noqa: E402
from level4.material_split import component_sensitivity, neck, percolation_radius  # noqa: E402

PROJECT = Path('/home/users/ntu/mingjie0/projects/doughnet')
BATCH_DIRNAME = 'records/level4_discriminative_batch_20261016'
NOMINAL_DIRNAME = 'records/level4_shared_reset_c0bb58e9eedd4f55b633da21cdb04f50/data'
SIM_TO_MM = 200.0
SUPPORT_RADIUS_MM = 2.0
COMPONENT_FRAMES = (0, 26, 32, 40, 50, 60, 62)
COMMON_FRAMES = tuple(range(0, 61, 5))
NOMINAL_YAW_DEG = 30.0
T1_GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
T2_GRID = (0.02, 0.05, 0.10, 0.15, 0.20)
QUANTILES = (0.2, 0.4, 0.5, 0.6, 0.8)
TOLERANCES = (0.0, 1.0 / 63.0)
TOP_K = 8
RANDOM_GROUPS = ('frozen_seed0', 'seed1_second_group')
EXEC_LABELS = ('B-pilot', 'B-exec-1', 'B-exec-2', 'B-exec-3', 'B-exec-4')


def utc():
    return datetime.now(timezone.utc).isoformat()


def latest(root, pattern, required=None):
    """Newest matching directory that holds ``required`` (failed preparations are skipped)."""
    found = [p for p in sorted(root.glob(pattern)) if required is None or (p / required).is_file()]
    return found[-1] if found else None


def verified(evidence, relative):
    manifest = read_json(evidence / 'evidence_manifest.json')['files_sha256']
    if sha256(evidence / relative) != manifest[relative]:
        raise ValueError('Evidence changed: %s/%s' % (evidence, relative))
    return evidence / relative


def discover(project):
    """Executed runs with exit 0 and their batch-report checks; never re-labels failures."""
    batch = project / BATCH_DIRNAME
    nominal = project / NOMINAL_DIRNAME
    plan = {r['candidate_id']: r for r in read_json(REPO / 'configs/level4/l4_01/pool32_manifest.json')['candidates']}
    runs = {}
    for cid in ('candidate_000', 'candidate_001'):
        runs[cid] = {'run_id': cid, 'evidence': nominal, 'job': '25837718.pbs101', 'checks': 'job-25837718 fixed pair',
                     'group': plan[cid]['group'], 'candidate_id': cid,
                     'action_mm_mm_deg': plan[cid]['parameters_mm_mm_deg'], 'tool_offset_mm': [0.0, 0.0]}
    pool = None
    for label in EXEC_LABELS:
        parent = latest(batch, label + '_*', 'data/batch_report.json')
        if parent is None:
            continue
        report = read_json(parent / 'data/batch_report.json')
        request = read_json(parent / 'tools/request.json')
        pool = Path(request['source_pool'])
        rows = {r['run_id']: r for r in request['runs']}
        for rid, result in report.get('results', {}).items():
            if result.get('exit_code') != 0:
                continue
            runs[rid] = {'run_id': rid, 'evidence': parent / 'data', 'job': label, 'checks': result,
                         'action_mm_mm_deg': rows[rid]['action_mm_mm_deg'], 'tool_offset_mm': rows[rid]['tool_offset_mm'],
                         'group': rows[rid]['group'], 'candidate_id': rows[rid]['candidate_id']}
    if pool is None:
        raise ValueError('No collected Phase B execution report found')
    return runs, pool, batch


def truth_for_run(args):
    run, pool, out = args
    target = out / 'truth' / run['run_id']
    if (target / 'complete.json').is_file():
        return run['run_id'], 'cached'
    target.mkdir(parents=True, exist_ok=True)
    log = verified(run['evidence'], 'execution/%s/scenes/0000/log.pkl' % run['run_id'])
    with log.open('rb') as stream:
        rows = pickle.load(stream)
    if len(rows) != 63:
        raise ValueError('Expected 63 logged frames: ' + run['run_id'])
    grid = load_grid(pool)
    scale = float(grid['unit_to_meter'])
    radius = SUPPORT_RADIUS_MM / (1000 * scale)
    low, high = grid['query'].min(axis=0), grid['query'].max(axis=0)
    times, classes, outside = [], [], []
    spans, raw = {'u': [], 'v': [], 'reason': []}, {'u': [], 'v': []}
    for row in rows:
        p = np.asarray(row['obj']['particles']['pos'], dtype=float)
        times.append(float(row['t']))
        c = particle_occupancy(grid['query'], p, radius)
        out_flag = bool(((p.min(axis=0) - radius < low) | (p.max(axis=0) + radius > high)).any())
        classes.append(c)
        outside.append(out_flag)
        geometry = from_occupancy(grid['query'], c, grid['counts'], grid['spacing'], scale, outside_support=out_flag)
        mu, mv = measure(geometry, [1, 0, 0]), measure(geometry, [0, 1, 0])
        spans['u'].append(mu['value'] if mu['status'] == 'measured' else None)
        spans['v'].append(mv['value'] if mv['status'] == 'measured' else None)
        spans['reason'].append(mu['reason'] or mv['reason'])
        raw['u'].append(float(np.ptp(p[:, 0])) * SIM_TO_MM)
        raw['v'].append(float(np.ptp(p[:, 1])) * SIM_TO_MM)
    if not (target / 'trajectory.npz').is_file():
        save_packet(target / 'trajectory.npz', times, classes, np.asarray(outside, dtype=bool))
    yaw = NOMINAL_YAW_DEG + float(run['action_mm_mm_deg'][2])
    c_axis = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0)
    d_axis = (-c_axis[1], c_axis[0], 0.0)
    mass = np.load(pool / 'static_material_info.npz')['mass'].astype(float)
    points0 = np.asarray(rows[0]['obj']['particles']['pos'], dtype=float) * SIM_TO_MM
    r0 = percolation_radius(points0)
    necks, components = [], {}
    for f, row in enumerate(rows):
        p = np.asarray(row['obj']['particles']['pos'], dtype=float) * SIM_TO_MM
        n = neck(p, d_axis, c_axis, 4.0)
        necks.append({'status': n['status'], 'width_mm': n['neck_width_mm'], 'position_mm': n['neck_position_mm']})
        if f in COMPONENT_FRAMES:
            s = component_sensitivity(p, r0, weights=mass)
            components[str(f)] = {
                'by_multiplier': {str(r['multiplier']): {
                    'components': r['components'], 'significant_0p5pct': r['significant_components']['0.005'],
                    'debris_0p5pct': r['debris_particles']['0.005'], 'two_largest_balance': r['two_largest_balance']}
                    for r in s['rows']},
                'significant_status': s['significant_status'],
                'simulator_topology_components': int(row['topology']['components']) if 'topology' in row else None}
    write_json(target / 'attributes.json', {
        'run_id': run['run_id'], 'times_s': times, 'voxel_span_mm': spans, 'raw_particle_span_mm': raw,
        'pinch_axes': {'closing': c_axis, 'finger_width': d_axis, 'yaw_deg': yaw}, 'pinch_neck_4mm': necks,
        'components': components, 'r0_mm': r0, 'support_radius_mm': SUPPORT_RADIUS_MM,
        'source_log_sha256': sha256(log), 'truth_source': 'raw material particles only'})
    write_json(target / 'complete.json', {'run_id': run['run_id'], 'completed_utc': utc(),
                                          'trajectory_sha256': sha256(target / 'trajectory.npz')})
    return run['run_id'], 'computed'


def prediction_spans(dyn, grid, variant, cid, hashes):
    directory = dyn / 'predictions' / variant / cid
    key = '%s/%s' % (variant, cid)
    if sha256(directory / 'trajectory.npz') != hashes[key]['trajectory.npz']:
        raise ValueError('Frozen prediction changed: ' + key)
    trajectory = load_trajectory(directory / 'trajectory.npz', grid)
    out = {'times_s': [], 'u': [], 'v': []}
    for t, geometry in trajectory:
        mu, mv = measure(geometry, [1, 0, 0]), measure(geometry, [0, 1, 0])
        out['times_s'].append(t)
        out['u'].append(mu['value'] if mu['status'] == 'measured' else None)
        out['v'].append(mv['value'] if mv['status'] == 'measured' else None)
    return out, trajectory


def make_spec(rho, eta, tolerance, ref_u, ref_v, task):
    """Development sweep spec (synthetic_only calibration); never a frozen task threshold."""
    return {'schema_version': '1.0.0', 'task_id': 'L4-01_dev_sweep_%s' % task, 'object': 'whole_material',
            'semantics': 'world_axis_extent',
            'reference_frame': {'kind': 'initial_fixed_world', 'unit': 'mm', 'origin_mm': [0, 0, 0],
                                'u': [1, 0, 0], 'v': [0, 1, 0]},
            'initial_reference': {'source': 'privileged_full_material_initial_state', 'u_mm': ref_u, 'v_mm': ref_v},
            'measurement': {'representation': 'voxel_union', 'operator': 'extrema', 'minimum_reference_span_mm': 1.0,
                            'tolerance_ratio': tolerance, 'prediction_margin_ratio': 0.0},
            'time': {'primary_s': 1.92, 'auxiliary_s': 1.984,
                     'required_s': [round(f * 0.032, 6) for f in COMMON_FRAMES],
                     'execution_required_s': [round(f * 0.032, 6) for f in range(61)], 'match_tolerance_s': 1e-6},
            'terminal_hard': {'id': 'narrow_u', 'rho': rho, 'relation': 'le', 'hard': True},
            'process_hard': [] if task == 'A' else [{'id': 'protect_v', 'eta': eta, 'relation': 'le', 'hard': True}],
            'soft_preferences': ['minimum_normalized_action_l2'],
            'calibration': {'status': 'synthetic_only', 'evidence_sha256': 'development_sweep_not_frozen',
                            'split_id': 'single_initial_state_development'}}


def voxel_iou(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else None


def d1_block(table, groups, variant, attribute):
    usable = [r for r in table if r['group'] in groups and r['execution'] == 'completed'
              and r['pred_' + variant][attribute] is not None and r['exec_common'][attribute] is not None]
    excluded = [r['candidate_id'] for r in table if r['group'] in groups and r not in usable]
    if len(usable) < TOP_K + 2:
        return {'status': 'insufficient', 'n': len(usable), 'excluded': excluded}
    ids = [r['candidate_id'] for r in usable]
    pred = np.array([r['pred_' + variant][attribute] for r in usable])
    exe = np.array([r['exec_common'][attribute] for r in usable])
    residual = pred - exe
    mask = ba.top_k_mask(pred, ids, TOP_K)
    mean_test = ba.permutation_mean_difference(residual, mask)
    rank_test = ba.spearman(ba.ranks(pred), residual)
    grid_thr = T1_GRID if attribute.startswith('T1') else T2_GRID
    quant_thr = [float(np.quantile(pred, q)) for q in QUANTILES]
    return {'n': len(usable), 'excluded': excluded, 'top_k_ids': [i for i, m in zip(ids, mask) if m],
            'residual_mean': float(residual.mean()), 'residual_sd': float(residual.std(ddof=1)),
            'per_candidate': [{'candidate_id': i, 'pred': float(a), 'exec': float(b), 'residual': float(a - b)}
                              for i, a, b in zip(ids, pred, exe)],
            'top_vs_rest': mean_test, 'spearman_score_rank_vs_residual': rank_test,
            'sweep_fixed_grid': ba.threshold_sweep(pred, exe, ids, grid_thr, TOP_K),
            'sweep_prediction_quantiles': ba.threshold_sweep(pred, exe, ids, quant_thr, TOP_K),
            'verdict': ba.d1_verdict(mean_test, rank_test, None)}


def report(out, project):
    from level4.evaluate import evaluate
    from level4.selection import confusion
    runs, pool, batch = discover(project)
    dyn_parent = latest(batch, 'B-dyn_*', 'data/prediction_manifest.json')
    dyn = dyn_parent / 'data'
    manifest = read_json(dyn / 'prediction_manifest.json')
    hashes = manifest['predictions_sha256']
    grid = load_grid(pool)
    pool_manifest = read_json(pool / 'pool.json')
    ref_u = float(pool_manifest['initial_reference_mm']['u']['value'])
    ref_v = float(pool_manifest['initial_reference_mm']['v']['value'])
    plan = read_json(REPO / 'configs/level4/l4_01/pool32_manifest.json')
    meta = {r['candidate_id']: r for r in plan['candidates']}
    request = read_json(dyn / 'request.json')
    candidates = [c['candidate_id'] for c in request['candidates']]
    table, trajectories = [], {}
    for cid in candidates:
        row = {'candidate_id': cid, 'group': meta[cid]['group'], 'action_mm_mm_deg': meta[cid]['parameters_mm_mm_deg']}
        for variant in manifest['variants']:
            spans, trajectory = prediction_spans(dyn, grid, variant, cid, hashes)
            trajectories[(variant, cid)] = trajectory
            row['pred_' + variant] = ba.task_attributes(spans['times_s'], spans['u'], spans['v'], ref_u, ref_v)
            row['pred_' + variant + '_spans'] = spans
        attr_path = out / 'truth' / cid / 'attributes.json'
        if cid in runs and attr_path.is_file():
            a = read_json(attr_path)
            common = list(COMMON_FRAMES)
            row['exec_common'] = ba.task_attributes([a['times_s'][f] for f in common],
                                                    [a['voxel_span_mm']['u'][f] for f in common],
                                                    [a['voxel_span_mm']['v'][f] for f in common], ref_u, ref_v)
            row['exec_dense'] = ba.task_attributes(a['times_s'][:61], a['voxel_span_mm']['u'][:61],
                                                   a['voxel_span_mm']['v'][:61], ref_u, ref_v)
            row['execution'] = 'completed'
            truth = np.load(out / 'truth' / cid / 'trajectory.npz')['classes']
            for variant in manifest['variants']:
                pred = np.load(dyn / 'predictions' / variant / cid / 'trajectory.npz')['classes']
                row['diag_voxel_iou_frame60_' + variant] = voxel_iou(pred[-1], truth[60])
        else:
            row['execution'] = 'not_executed_or_failed'
        table.append(row)
    result = {'kind': 'L4-01_phase_c_report', 'created_utc': utc(), 'dyn_job_dir': str(dyn_parent),
              'prediction_manifest_sha256': sha256(dyn / 'prediction_manifest.json'),
              'reference_spans_mm': {'u': ref_u, 'v': ref_v}, 'top_k': TOP_K, 'candidates': table,
              'executed_runs': sorted(runs)}
    # ---- C2 / D1 --------------------------------------------------------------------------------
    d1 = {}
    for population, groups in (('random_groups', RANDOM_GROUPS), ('random_plus_d2', RANDOM_GROUPS + ('d2_edge',))):
        for variant in manifest['variants']:
            for attribute in ('T1_r_u_end', 'T2_max_abs_v_dev'):
                d1['%s|%s|%s' % (population, variant, attribute)] = d1_block(table, groups, variant, attribute)
    result['C2_D1'] = d1
    result['C2_D1_primary'] = ['random_groups|v3_nominal|T1_r_u_end', 'random_groups|v3_nominal|T2_max_abs_v_dev']
    # ---- C3 / D2 --------------------------------------------------------------------------------
    d2 = []
    for rid, run in sorted(runs.items()):
        path = out / 'truth' / rid / 'attributes.json'
        if not path.is_file():
            continue
        a = read_json(path)
        f60 = a['components']['60']['by_multiplier']
        sig = sorted({v['significant_0p5pct'] for v in f60.values()})
        balances = [v['two_largest_balance'] for v in f60.values() if v['significant_0p5pct'] >= 2]
        d2.append({'run_id': rid, 'group': run['group'], 'action_mm_mm_deg': run['action_mm_mm_deg'],
                   'tool_offset_mm': run['tool_offset_mm'],
                   'significant_components_frame60_by_multiplier': {k: v['significant_0p5pct'] for k, v in f60.items()},
                   'break_frame60': ('stable_break' if min(sig) >= 2 else
                                     'no_break' if max(sig) == 1 else 'threshold_dependent'),
                   'balance_frame60': max(balances) if balances else None,
                   'pinch_neck_mm_frame60': a['pinch_neck_4mm'][60]['width_mm'],
                   'pinch_neck_mm_frame62': a['pinch_neck_4mm'][62]['width_mm'],
                   'simulator_topology_frame60': a['components']['60']['simulator_topology_components']})
    breaks = [r for r in d2 if r['break_frame60'] != 'no_break']
    result['C3_D2'] = {'runs': d2, 'any_break': bool(breaks), 'break_runs': [r['run_id'] for r in breaks],
                       'equal_split_reachable': any((r['balance_frame60'] or 0) >= 0.8 for r in breaks),
                       'equal_split_rule': 'two largest significant pieces within 20% (balance >= 0.8)'}
    eligible = [r for r in d2 if r['group'] in RANDOM_GROUPS + ('d2_edge',)]
    unbroken = sorted([r for r in eligible if r['break_frame60'] == 'no_break' and r['pinch_neck_mm_frame60'] is not None],
                      key=lambda r: (r['pinch_neck_mm_frame60'], r['run_id']))
    broken = [r for r in eligible if r['break_frame60'] != 'no_break']
    chosen = ([broken[0]['run_id']] if broken else []) + [r['run_id'] for r in unbroken][:2 - (1 if broken else 0)]
    result['S4_selection'] = {'rule': 'A7 pre-registered: the 1-2 executed candidates with the smallest pinch neck at '
                                      'frame60, or, if any break, one broken and the thinnest unbroken candidate',
                              'breaks_present': bool(broken), 'chosen': chosen,
                              'chosen_neck_mm': {r['run_id']: r['pinch_neck_mm_frame60'] for r in unbroken[:2]}}
    # ---- C4 / S0-S3a ----------------------------------------------------------------------------
    sens = {}
    reference = out / 'truth' / 'candidate_000' / 'attributes.json'
    if reference.is_file():
        ref = read_json(reference)
        for rid in ('S0_repeat_nominal', 'S1_x_0p02', 'S1_y_0p02', 'S2_x_0p1'):
            path = out / 'truth' / rid / 'attributes.json'
            if not path.is_file():
                sens[rid] = {'status': 'not_executed'}
                continue
            a = read_json(path)
            du = [x - y for x, y in zip(a['raw_particle_span_mm']['u'], ref['raw_particle_span_mm']['u'])]
            dv = [x - y for x, y in zip(a['raw_particle_span_mm']['v'], ref['raw_particle_span_mm']['v'])]
            dn = [x['width_mm'] - y['width_mm'] for x, y in zip(a['pinch_neck_4mm'], ref['pinch_neck_4mm'])
                  if x['width_mm'] is not None and y['width_mm'] is not None]
            presence = any((x['width_mm'] is None) != (y['width_mm'] is None)
                           for x, y in zip(a['pinch_neck_4mm'], ref['pinch_neck_4mm']))
            comps = any(a['components'][f]['by_multiplier'][m]['significant_0p5pct'] !=
                        ref['components'][f]['by_multiplier'][m]['significant_0p5pct']
                        for f in a['components'] for m in a['components'][f]['by_multiplier'])
            judgement = ba.sensitivity_judgement({'span_u_mm': du, 'span_v_mm': dv, 'neck_mm': dn,
                                                  'components_changed': comps or presence})
            sens[rid] = {'judgement': judgement, 'neck_presence_changed': presence,
                         'S3a_per_frame': {'span_u_mm': du, 'span_v_mm': dv, 'neck_mm': dn},
                         'role': {'S0_repeat_nominal': 'noise floor',
                                  'S2_x_0p1': 'reference (robot scale)'}.get(rid, 'D-1 criterion')}
    result['C4_S'] = sens
    # ---- C5 / 3x3 matrices (development exploration) --------------------------------------------
    executed = [r for r in table if r['execution'] == 'completed' and r['group'] in RANDOM_GROUPS]
    truths = {r['candidate_id']: load_trajectory(out / 'truth' / r['candidate_id'] / 'trajectory.npz', grid)
              for r in executed}
    matrices = []
    for tolerance in TOLERANCES:
        for t1 in T1_GRID:
            for task, etas in (('A', (None,)), ('B', T2_GRID)):
                for eta in etas:
                    spec = make_spec(round(1 - t1, 6), eta, tolerance, ref_u, ref_v, task)
                    for variant in manifest['variants']:
                        dense, common = [], []
                        for r in executed:
                            cid = r['candidate_id']
                            p = evaluate(spec, trajectories[(variant, cid)], prediction=True)
                            dense.append({'prediction': p, 'execution': evaluate(spec, truths[cid])})
                            common.append({'prediction': p, 'execution': evaluate(spec, truths[cid], common_schedule=True)})
                        matrices.append({'task': task, 'r_u_threshold': t1, 'eta': eta, 'tolerance_ratio': tolerance,
                                         'variant': variant, 'n': len(dense), 'dense': confusion(dense),
                                         'common_schedule': confusion(common)})
    result['C5_matrices'] = {'label': 'development exploration; thresholds not frozen', 'rows': matrices}
    write_json(out / 'report' / 'phase_c_report.json', result)
    with (out / 'report' / 'candidates.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate_id', 'group', 'dx_mm', 'dy_mm', 'yaw_deg', 'execution',
                         'T1_pred_v3', 'T1_pred_legacy', 'T1_exec', 'T2_pred_v3', 'T2_pred_legacy', 'T2_exec',
                         'iou60_v3', 'iou60_legacy'])
        for r in table:
            e = r.get('exec_common', {})
            writer.writerow([r['candidate_id'], r['group'], *r['action_mm_mm_deg'], r['execution'],
                             r['pred_v3_nominal']['T1_r_u_end'], r['pred_legacy_nominal']['T1_r_u_end'],
                             e.get('T1_r_u_end'), r['pred_v3_nominal']['T2_max_abs_v_dev'],
                             r['pred_legacy_nominal']['T2_max_abs_v_dev'], e.get('T2_max_abs_v_dev'),
                             r.get('diag_voxel_iou_frame60_v3_nominal'), r.get('diag_voxel_iou_frame60_legacy_nominal')])
    return result


def summary_lines(result):
    lines = ['PHASE_C_REPORT ' + result['created_utc'] + ' executed_runs=%d' % len(result['executed_runs'])]
    for key, value in result['C2_D1'].items():
        if 'verdict' in value:
            t = value['top_vs_rest']
            lines.append('D1 %-46s n=%2d top8-rest=%+.4f p=%.4f rho=%+.3f -> %s' % (
                key, value['n'], t['selected_minus_rest_mean'], t['p_one_sided_optimistic'],
                value['spearman_score_rank_vs_residual']['rho'], value['verdict']['verdict']))
        else:
            lines.append('D1 %-46s %s n=%d' % (key, value['status'], value['n']))
    d2 = result['C3_D2']
    lines.append('D2 any_break=%s break_runs=%s equal_split_reachable=%s' % (
        d2['any_break'], d2['break_runs'], d2['equal_split_reachable']))
    necks = [r['pinch_neck_mm_frame60'] for r in d2['runs'] if r['pinch_neck_mm_frame60'] is not None]
    if necks:
        lines.append('D2 pinch neck frame60 mm: min=%.2f median=%.2f max=%.2f (n=%d)' % (
            min(necks), float(np.median(necks)), max(necks), len(necks)))
    lines.append('S4 chosen=%s' % result['S4_selection']['chosen'])
    for rid, s in result['C4_S'].items():
        if 'judgement' in s:
            j = s['judgement']
            worst = {k: (None if v is None else round(v, 4)) for k, v in j['worst_abs_change_mm'].items()}
            lines.append('S %-18s %-6s worst=%s comps_changed=%s (%s)' % (rid, j['status'], worst,
                                                                         j['components_changed'], s['role']))
        else:
            lines.append('S %-18s %s' % (rid, s['status']))
    v3 = [r['diag_voxel_iou_frame60_v3_nominal'] for r in result['candidates']
          if r['execution'] == 'completed' and r.get('diag_voxel_iou_frame60_v3_nominal') is not None]
    lg = [r['diag_voxel_iou_frame60_legacy_nominal'] for r in result['candidates']
          if r['execution'] == 'completed' and r.get('diag_voxel_iou_frame60_legacy_nominal') is not None]
    if v3 and lg:
        lines.append('D4-diagnostic voxel IoU@frame60 (not the frozen D4 metric): v3 median=%.4f legacy median=%.4f n=%d'
                     % (float(np.median(v3)), float(np.median(lg)), len(v3)))
    return lines


def run(out, project, workers):
    out.mkdir(parents=True, exist_ok=True)
    runs, pool, _ = discover(project)
    work = [(r, pool, out) for r in runs.values()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for rid, state in executor.map(truth_for_run, work):
            print('TRUTH', rid, state, flush=True)
    result = report(out, project)
    lines = summary_lines(result)
    with (out / 'report' / 'summary.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


def submit(project, workers):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    parent = project / BATCH_DIRNAME / ('C-analysis_' + stamp)
    parent.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    script = '\n'.join(['#!/bin/bash', 'set -uo pipefail',
                        'source /home/users/ntu/mingjie0/venvs/doughnet/bin/activate',
                        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1',
                        'cd %s' % REPO, 'echo "commit $(git rev-parse HEAD)"',
                        'python -B -u scripts/analyze_level4_batch.py run --out %s/data --workers %d' % (parent, workers),
                        'echo $? > %s/job_exit_code.txt' % parent, ''])
    with (parent / 'run_analysis.pbs').open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(script)
    write_json(parent / 'submission.json', {'repo_commit': commit, 'created_utc': utc(), 'workers': workers})
    command = ['qsub', '-N', 'l4_C_analysis', '-P', 'personal', '-q', 'normal', '-l',
               'select=1:ncpus=%d:mem=64GB' % workers, '-l', 'walltime=02:00:00', '-j', 'oe',
               '-o', str(parent / 'pbs.log'), str(parent / 'run_analysis.pbs')]
    result = subprocess.run(command, capture_output=True, text=True)
    print('SUBMITTED ' + result.stdout.strip() if result.returncode == 0 else 'QSUB_FAILED ' + result.stderr.strip())
    print('ANALYSIS_DIR', parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('submit')
    s.add_argument('--workers', type=int, default=16)
    for p in (r, s):
        p.add_argument('--project', type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.command == 'run':
        sys.exit(run(args.out, args.project, args.workers))
    sys.exit(submit(args.project, args.workers))


if __name__ == '__main__':
    main()
