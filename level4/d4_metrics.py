"""L4-02 D4: numpy-only measurement and pre-registered verdict rules (docs/level4_l4_02_D4_prereg.md).

Everything here is CPU/numpy so it is unit-tested locally; the GPU job
(scripts/l4_02_d4.py eval) only produces raw per-frame metrics and class arrays.
"""
from __future__ import annotations

import math

import numpy as np

from .batch_analysis import spearman
from .geometry import Geometry, measure
from .material_split import neck

SIM_TO_MM = 200.0
OFFICIAL_RESOLUTION = 0.01
OFFICIAL_BOUNDS = ((-0.30, 0.30), (-0.52, 0.52), (-OFFICIAL_RESOLUTION / 2, 0.11 + OFFICIAL_RESOLUTION / 2))
TRAINING_QUERY_Z_GUARANTEED_MM = -7.5   # box queries z >= -0.05 x scale, scale in [0.75, 1.25] -> always covered above -7.5 mm
TRAINING_QUERY_Z_MAX_MM = -12.5         # ... and never below -12.5 mm
NOMINAL_YAW_DEG = 30.0

# ---- pre-registered numbers (L4-01 section 3 and the L3 diagnostic it cites) ----
L3_ROLLOUT_MEAN = 81.95142944653828      # dyn_check_ml9xhm summary.json rollout_mean.viou (12 intervals)
L3_SINGLE_MEAN = 91.9066255291303        # single_step_mean.viou
L3_ROLLOUT_F60 = 57.33605623245239       # rollout_last (target frame 60) viou
MARGIN_PP = 3.0
P_ORIGINAL = {'P1_rollout_f60_min': L3_ROLLOUT_MEAN - MARGIN_PP, 'P2_rollout_f60_median_min': L3_ROLLOUT_MEAN - MARGIN_PP,
              'P3_single_median_min': L3_SINGLE_MEAN - MARGIN_PP, 'P2_min_executed': 20}
P_CORRECTED = {'rollout_mean_min': L3_ROLLOUT_MEAN - MARGIN_PP, 'rollout_f60_min': L3_ROLLOUT_F60 - MARGIN_PP,
               'single_mean_min': L3_SINGLE_MEAN - MARGIN_PP, 'min_executed': 20}
P4_RATIO = (0.1, 10.0)
PC_REPRODUCE_PP = 1.0      # positive control: |mean VIoU - L3 record| <= 1 pp -> pipeline reproduces
PC_LOW_PP = 3.0            # more than 3 pp below the L3 record -> pipeline abnormal
PC_LATENT_ABS = 1e-3
Z1_DOMAIN_SHARE = 0.8
Z1_INDOMAIN_SHARE = 0.2
Z2_AE_SUBGROUND_SHARE = 0.01
Z3_ALIGN_MM = 1.0
SENS_RATIO = (0.25, 4.0)
SENS_RHO = 0.5
SENS_RHO_LOW = 0.2
SENS_P = 0.05


def official_query():
    """The validation query grid of net/pipeline/pipeline.py and scripts/check_level3_dyn.py (simulator units)."""
    bounds = np.asarray(OFFICIAL_BOUNDS, dtype=np.float64)
    steps = np.rint((bounds[:, 1] - bounds[:, 0]) / OFFICIAL_RESOLUTION).astype(np.int32) + 1  # (61, 105, 13)
    axes = [np.linspace(bounds[i, 0], bounds[i, 1], int(steps[i]), dtype=np.float32) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3), steps


def mask_like_evaluater(classes, query, in_tool=None):
    """Repository Evaluater: estimates below the plane (z<0) and inside the tool count as empty."""
    out = np.asarray(classes).copy()
    out[np.asarray(query)[:, 2] < 0] = 0
    if in_tool is not None:
        out[np.asarray(in_tool, dtype=bool)] = 0
    return out


def attributes(classes, query, yaw_deg, cell=OFFICIAL_RESOLUTION, unit_to_meter=0.2):
    """u/v voxel spans (mm, extrema + cell support) and the pinch-frame neck (4 mm slices) of occupied cells."""
    occupied = np.asarray(classes) > 0
    if not occupied.any():
        return {'span_u_mm': None, 'span_v_mm': None, 'neck_mm': None}
    points = np.asarray(query, dtype=np.float64)[occupied]
    g = Geometry(points, unit_to_meter, 'voxel_union', True, True, np.full(3, cell), False)
    mu, mv = measure(g, [1, 0, 0]), measure(g, [0, 1, 0])
    c = (math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0)
    n = neck(points * 1000 * unit_to_meter, (-c[1], c[0], 0.0), c, 4.0)
    return {'span_u_mm': mu['value'], 'span_v_mm': mv['value'], 'neck_mm': n['neck_width_mm']}


def subground_histogram(classes, query, unit_to_meter=0.2):
    """Occupied cells below z=0 binned against the training query domain (mm)."""
    z = np.asarray(query)[:, 2] * 1000 * unit_to_meter
    occupied = np.asarray(classes) > 0
    below = occupied & (z < 0)
    edges = (-math.inf, TRAINING_QUERY_Z_MAX_MM, TRAINING_QUERY_Z_GUARANTEED_MM, -1.0 - 1e-9, 0.0)
    names = ('below_-12.5', '-12.5_to_-7.5', '-7.5_to_-1', '-1_to_0')
    counts = {n: int((below & (z >= lo) & (z < hi)).sum()) for n, lo, hi in zip(names, edges[:-1], edges[1:])}
    total = int(below.sum())
    return {'occupied': int(occupied.sum()), 'below_zero': total,
            'below_zero_share': total / int(occupied.sum()) if occupied.any() else None,
            'bins': counts,
            'share_outside_guaranteed_domain': ((counts['below_-12.5'] + counts['-12.5_to_-7.5']) / total) if total else None}


# ------------------------------------------------------------------ verdicts
def p_original(c000_f60, f60_all, single_all):
    """L4-01 section 3 exactly as frozen (reference = L3 rollout mean)."""
    t = P_ORIGINAL
    p1 = 'pass' if c000_f60 >= t['P1_rollout_f60_min'] else 'fail'
    if len(f60_all) < t['P2_min_executed']:
        p2 = 'uncertain'
    else:
        p2 = 'pass' if float(np.median(f60_all)) >= t['P2_rollout_f60_median_min'] else 'fail'
    p3 = 'pass' if float(np.median(single_all)) >= t['P3_single_median_min'] else 'fail'
    return {'P1': p1, 'P2': p2, 'P3': p3, 'thresholds': t,
            'values': {'P1_c000_f60': c000_f60, 'P2_median_f60': float(np.median(f60_all)), 'P2_n': len(f60_all),
                       'P3_median_single': float(np.median(single_all))}}


def p_corrected(c000_mean, c000_f60, mean_all, f60_all, single_mean_all):
    """Like-for-like reference (proposed in D4_prereg; applies only if the overview approves before the job)."""
    t = P_CORRECTED
    p1 = 'pass' if c000_mean >= t['rollout_mean_min'] and c000_f60 >= t['rollout_f60_min'] else 'fail'
    if len(mean_all) < t['min_executed']:
        p2 = 'uncertain'
    else:
        ok = float(np.median(mean_all)) >= t['rollout_mean_min'] and float(np.median(f60_all)) >= t['rollout_f60_min']
        p2 = 'pass' if ok else 'fail'
    p3 = 'pass' if float(np.median(single_mean_all)) >= t['single_mean_min'] else 'fail'
    return {'P1': p1, 'P2': p2, 'P3': p3, 'thresholds': t,
            'values': {'P1_c000_mean': c000_mean, 'P1_c000_f60': c000_f60, 'P2_median_mean': float(np.median(mean_all)),
                       'P2_median_f60': float(np.median(f60_all)), 'P2_n': len(mean_all),
                       'P3_median_single_mean': float(np.median(single_mean_all))}}


def p4(l4_abs_errors, l3_abs_errors):
    """Median absolute attribute error ratio L4 / L3 within [0.1, 10] for every attribute; 'uncertain' if unavailable."""
    rows, status = {}, []
    for key, ref in l3_abs_errors.items():
        values = [v for v in l4_abs_errors.get(key, []) if v is not None]
        if ref is None or not values:
            rows[key] = {'status': 'uncertain', 'reason': 'reference or L4 values unavailable'}
            status.append('uncertain')
            continue
        median = float(np.median(values))
        if ref == 0:
            ratio = math.inf if median > 0 else 1.0
        else:
            ratio = median / ref
        s = 'pass' if P4_RATIO[0] <= ratio <= P4_RATIO[1] else ('fail' if ratio > P4_RATIO[1] else 'pass_below')
        rows[key] = {'status': s, 'l4_median_abs_mm': median, 'l3_abs_mm': ref, 'ratio': ratio, 'n': len(values)}
        status.append('fail' if s == 'fail' else 'uncertain' if s == 'uncertain' else 'pass')
    overall = 'fail' if 'fail' in status else ('pass' if all(s == 'pass' for s in status) else 'uncertain')
    return {'P4': overall, 'rows': rows, 'ratio_window': P4_RATIO,
            'note': 'smaller-than-reference errors (ratio < 0.1) are not a failure (pass_below)'}


def d4_verdict(p, p4_result):
    """D4 = pass: P1-P3 pass and P4 pass/uncertain; fail: any of P1-P3 fail; otherwise uncertain."""
    first = [p['P1'], p['P2'], p['P3']]
    if 'fail' in first:
        return 'fail'
    if all(x == 'pass' for x in first) and p4_result['P4'] in ('pass', 'uncertain'):
        return 'pass'
    return 'uncertain'


def positive_control(rollout_viou, single_viou, latent_max_abs):
    """B-dyn code path on the L3 scene vs the L3 record (rollout/single-step 12-interval means)."""
    r, s = float(np.mean(rollout_viou)), float(np.mean(single_viou))
    dr, ds = r - L3_ROLLOUT_MEAN, s - L3_SINGLE_MEAN
    if abs(dr) <= PC_REPRODUCE_PP and abs(ds) <= PC_REPRODUCE_PP and latent_max_abs <= PC_LATENT_ABS:
        status = 'normal'
    elif dr < -PC_LOW_PP or ds < -PC_LOW_PP:
        status = 'abnormal'
    else:
        status = 'uncertain'
    return {'status': status, 'rollout_mean': r, 'single_mean': s, 'delta_rollout_pp': dr, 'delta_single_pp': ds,
            'latent_max_abs': latent_max_abs, 'rules': {'reproduce_pp': PC_REPRODUCE_PP, 'low_pp': PC_LOW_PP,
                                                       'latent_abs': PC_LATENT_ABS}}


def route(d4, pc):
    """Task sheet L4-02 B2 decision table."""
    if d4 == 'pass':
        return 'D4_pass: proceed to D1-v2 and L5 calibration'
    if d4 == 'fail' and pc == 'normal':
        return 'out_of_distribution_or_model_error: prefer in-distribution initial states (merge with B3)'
    if d4 == 'fail' and pc == 'abnormal':
        return 'pipeline_bug: fix, then rerun B-dyn (Dyn only, no new simulation)'
    return 'uncertain: report to overview'


def z_checks(hist_rollout, hist_ae_k0, obs_zmin_l4_mm, obs_zmin_l3_mm):
    share = hist_rollout.get('share_outside_guaranteed_domain')
    if share is None:
        z1 = 'no_subground'
    elif share >= Z1_DOMAIN_SHARE:
        z1 = 'query_domain_extrapolation'
    elif share <= Z1_INDOMAIN_SHARE:
        z1 = 'in_domain_subground_prediction'
    else:
        z1 = 'mixed'
    ae_share = hist_ae_k0.get('below_zero_share') or 0.0
    z2 = 'decoder_property_without_dynamics' if ae_share >= Z2_AE_SUBGROUND_SHARE else 'arises_in_dynamics'
    z3 = 'aligned' if (abs(obs_zmin_l4_mm - obs_zmin_l3_mm) <= Z3_ALIGN_MM) else 'misaligned'
    return {'Z1': z1, 'Z1_share_outside_guaranteed_domain': share, 'Z2': z2, 'Z2_ae_k0_below_zero_share': ae_share,
            'Z3': z3, 'Z3_obs_zmin_mm': {'l4': obs_zmin_l4_mm, 'l3': obs_zmin_l3_mm},
            'rules': {'Z1_domain_share': Z1_DOMAIN_SHARE, 'Z1_indomain_share': Z1_INDOMAIN_SHARE,
                      'Z2_ae_share': Z2_AE_SUBGROUND_SHARE, 'Z3_align_mm': Z3_ALIGN_MM,
                      'training_query_z_mm': [TRAINING_QUERY_Z_MAX_MM, TRAINING_QUERY_Z_GUARANTEED_MM]}}


def sensitivity(pred, exe, permutations=10000):
    """Variance ratio and Spearman rank correlation of predicted vs executed attribute over candidates."""
    pairs = [(a, b) for a, b in zip(pred, exe) if a is not None and b is not None]
    if len(pairs) < 8:
        return {'status': 'uncertain', 'reason': 'fewer than 8 paired candidates', 'n': len(pairs)}
    p, e = np.asarray(pairs, dtype=float).T
    ve = float(e.var(ddof=1))
    ratio = float(p.var(ddof=1) / ve) if ve > 0 else math.inf
    rho = spearman(p, e, permutations=permutations)
    one_sided = rho['p_two_sided'] / 2 if rho['rho'] > 0 else 1.0
    if SENS_RATIO[0] <= ratio <= SENS_RATIO[1] and rho['rho'] >= SENS_RHO and one_sided < SENS_P:
        status = 'tracks_action'
    elif ratio < SENS_RATIO[0]:
        status = 'insensitive'
    elif rho['rho'] < SENS_RHO_LOW:
        status = 'varies_but_wrong_order'
    else:
        status = 'uncertain'
    return {'status': status, 'n': len(pairs), 'var_ratio_pred_over_exec': ratio, 'spearman_rho': rho['rho'],
            'p_one_sided': one_sided, 'rules': {'ratio': SENS_RATIO, 'rho': SENS_RHO, 'rho_low': SENS_RHO_LOW, 'p': SENS_P}}
