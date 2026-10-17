"""L4-01 Phase C statistics on fixed-pool attributes (pure functions, CPU).

Attributes per candidate (task sheet C2): T1 = r_u(T) = span_u(1.92 s)/span_u(0);
T2 = max over the schedule of |r_v(t) - 1|.  Residual e = phi_pred - phi_exec, so
e < 0 means the prediction was optimistic (narrower u / steadier v than real).
All tests are pre-declared here; every threshold of a sweep is reported, none
is chosen after the fact.  Nothing here selects actions or uses truth online.
"""
from __future__ import annotations

import math

import numpy as np


def task_attributes(times, span_u, span_v, ref_u, ref_v, primary_s=1.92, tol=1e-8):
    """T1/T2 from span series; None where a required span is unknown."""
    times = np.asarray(times, dtype=float)
    end = np.flatnonzero(np.abs(times - primary_s) <= tol)
    if len(end) != 1 or ref_u <= 0 or ref_v <= 0:
        raise ValueError('Primary time missing or invalid reference spans')
    t1 = None if span_u[int(end[0])] is None else span_u[int(end[0])] / ref_u
    upto = [i for i, t in enumerate(times) if t <= primary_s + tol]
    v = [span_v[i] for i in upto]
    t2 = None if any(x is None for x in v) else max(abs(x / ref_v - 1) for x in v)
    return {'T1_r_u_end': t1, 'T2_max_abs_v_dev': t2}


def ranks(values):
    """Average ranks (1-based), ties share the mean rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranked = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranked[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranked


def spearman(x, y, permutations=10000, seed=0):
    """Spearman rho with a two-sided permutation p-value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError('Need at least three paired values')
    rx, ry = ranks(x), ranks(y)
    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = math.sqrt(float((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 0 else 0.0
    observed = rho(rx, ry)
    rng = np.random.default_rng(seed)
    hits = sum(abs(rho(rx, rng.permutation(ry))) >= abs(observed) - 1e-12 for _ in range(permutations))
    return {'rho': observed, 'p_two_sided': (hits + 1) / (permutations + 1), 'n': len(x), 'permutations': permutations}


def top_k_mask(scores, ids, k):
    """Best k by ascending predicted score; ties broken by candidate id (frozen rule)."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], ids[i]))
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def permutation_mean_difference(residuals, mask, permutations=10000, seed=0):
    """One-sided test: selected residual mean lower (more optimistic) than the rest."""
    r, m = np.asarray(residuals, dtype=float), np.asarray(mask, dtype=bool)
    if m.sum() == 0 or (~m).sum() == 0:
        raise ValueError('Both groups must be non-empty')
    observed = float(r[m].mean() - r[~m].mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        p = rng.permutation(m)
        hits += (r[p].mean() - r[~p].mean()) <= observed + 1e-12
    pooled = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return {'selected_minus_rest_mean': observed, 'p_one_sided_optimistic': (hits + 1) / (permutations + 1),
            'selected_n': int(m.sum()), 'rest_n': int((~m).sum()), 'pooled_sd': pooled,
            'standardized_effect': observed / pooled if pooled > 0 else None, 'permutations': permutations}


def false_feasible(pred_ok, exec_ok, mask):
    """Among masked candidates predicted feasible, the fraction not feasible in execution."""
    pred_ok, exec_ok, mask = (np.asarray(a, dtype=bool) for a in (pred_ok, exec_ok, mask))
    chosen = mask & pred_ok
    return None if chosen.sum() == 0 else float((chosen & ~exec_ok).sum() / chosen.sum())


def random_k_false_feasible(pred_ok, exec_ok, k, draws=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pred_ok)
    values = []
    for _ in range(draws):
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=k, replace=False)] = True
        value = false_feasible(pred_ok, exec_ok, mask)
        if value is not None:
            values.append(value)
    return np.asarray(values)


def threshold_sweep(pred, exe, ids, thresholds, k=8, draws=10000, seed=0):
    """C2(3): for each pre-listed threshold, top-k vs random-k 'predicted ok, actually not ok' rates."""
    pred, exe = np.asarray(pred, dtype=float), np.asarray(exe, dtype=float)
    top = top_k_mask(pred, ids, k)
    rows = []
    for threshold in thresholds:
        pred_ok, exec_ok = pred <= threshold, exe <= threshold
        top_rate = false_feasible(pred_ok, exec_ok, top)
        random_rates = random_k_false_feasible(pred_ok, exec_ok, k, draws, seed)
        rows.append({'threshold': float(threshold), 'predicted_ok': int(pred_ok.sum()), 'executed_ok': int(exec_ok.sum()),
                     'top_k_predicted_ok': int((top & pred_ok).sum()), 'top_k_false_feasible': top_rate,
                     'random_k_mean': float(random_rates.mean()) if len(random_rates) else None,
                     'random_k_q05_q95': np.quantile(random_rates, [.05, .95]).tolist() if len(random_rates) else None,
                     'p_random_ge_top': (float((random_rates >= top_rate - 1e-12).mean()) if len(random_rates)
                                         and top_rate is not None else None),
                     'random_draws_defined': int(len(random_rates))})
    return rows


def samples_per_group(effect, sd, alpha=0.05, power=0.8):
    """Normal-approximation n per group for a one-sided two-sample mean difference."""
    if sd is None or sd <= 0 or effect == 0:
        return None
    z = {0.05: 1.6448536269514722}[alpha] + {0.8: 0.8416212335729143}[power]
    return int(math.ceil(2 * (z * sd / abs(effect)) ** 2))


def d1_verdict(mean_test, spearman_result, sweep, alpha=0.05, minimum_selected=8):
    """Support / not support / uncertain, with the frozen decision rule stated in the output."""
    rule = ('support: top-k minus rest residual mean < 0 (optimistic) with one-sided permutation p < 0.05; '
            'not_support: mean difference >= 0, or |standardized effect| < 0.2; uncertain: otherwise, or fewer than '
            'minimum_selected candidates.  Spearman and the threshold sweep are reported, not used in the verdict.')
    diff, p = mean_test['selected_minus_rest_mean'], mean_test['p_one_sided_optimistic']
    effect = mean_test['standardized_effect']
    if mean_test['selected_n'] < minimum_selected:
        verdict = 'uncertain_insufficient_sample'
    elif p < alpha and diff < 0:
        verdict = 'support'
    elif diff >= 0 or (effect is not None and abs(effect) < 0.2):
        verdict = 'not_support'
    else:
        verdict = 'uncertain'
    return {'verdict': verdict, 'rule': rule,
            'n_per_group_for_80pct_power': samples_per_group(diff, mean_test['pooled_sd'])}


def sensitivity_judgement(differences, limit_mm=0.1):
    """D-1 section 3 interim criterion for one S-run against its reference run.

    differences: {'span_u_mm': [...], 'span_v_mm': [...], 'neck_mm': [...], 'components_changed': bool}
    """
    # An empty list means the quantity is undefined in both runs (e.g. no waist yet); a
    # presence mismatch must be passed in through components_changed by the caller.
    worst = {k: (None if not v else float(np.max(np.abs(v)))) for k, v in differences.items() if k != 'components_changed'}
    ok = all(w is None or w < limit_mm for w in worst.values()) and not differences['components_changed']
    return {'status': 'passed' if ok else 'failed', 'worst_abs_change_mm': worst, 'limit_mm': limit_mm,
            'not_applicable': sorted(k for k, w in worst.items() if w is None),
            'components_changed': bool(differences['components_changed']),
            'criterion': 'D-1 interim: span and neck change < 0.1 mm and component count unchanged'}
