"""L4-02 A1/A2 (CPU, login node): execution attribute distribution and T1'/T2' feasibility drafts.

Reads only existing Phase C truth (``C-analysis_*/data/truth/<run>/``, raw-particle
derived) plus the hash-verified execution logs for the simulator's own topology
labels.  No prediction is read.  Thresholds printed here are development candidates
(quantiles of this pool), never frozen task thresholds.

    python -B scripts/l4_02_attributes.py [--analysis <C-analysis dir>]

Writes ``<analysis>/l4_02_A1A2_<stamp>/``: attributes.csv, attribute_distribution.md,
task_v2_draft.md, a1a2.json and (if matplotlib is importable) three PNG scatter plots.
"""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.artifacts import read_json, sha256, write_json  # noqa: E402
from scripts import analyze_level4_batch as az  # noqa: E402

PRIMARY = 60
CANDIDATE_GROUPS = ('frozen_seed0', 'seed1_second_group', 'd2_edge')
QUANTILES = (0.25, 0.5, 0.75)
DIFFERENT_ACTION = (5.0, 5.0, 5.0)   # |d dx| mm, |d dy| mm, |d yaw| deg: any one at/above -> different action
PROVISIONAL_NECK_NOISE_MM = 0.0608   # S0 (L4-01); B1 replaces it
ATTRIBUTES = ('neck60_mm', 'neck62_mm', 'neck_min_0_60_mm', 'r_u60', 'r_v60', 'dv60_mm', 'dv_max_mm',
              'dv_max_ratio', 'du60_mm', 'raw_u60_mm', 'raw_v60_mm')


def simulator_labels(rows):
    """The simulator's SceneGraph topology labels (the DoughNet training labels)."""
    comps = [int(r['topology']['components']) for r in rows]
    genus = [list(r['topology']['genus']) for r in rows]
    split = next((f for f, c in enumerate(comps) if c >= 2), None)
    parts = np.unique(np.asarray(rows[PRIMARY]['obj']['particles']['idx']), return_counts=True)[1]
    balance = float(parts.min() / parts.max()) if len(parts) > 1 else None
    first_genus_change = next((f for f, g in enumerate(genus) if g != genus[0]), None)
    return {'sim_components60': comps[PRIMARY], 'sim_components62': comps[-1], 'sim_genus0': genus[0],
            'sim_genus60': genus[PRIMARY], 'sim_first_split_frame': split,
            'sim_first_genus_change_frame': first_genus_change, 'sim_part_balance60': balance}


def run_row(run, truth, ref_u, ref_v, rows=None):
    a = read_json(truth / 'attributes.json')
    u, v = a['voxel_span_mm']['u'], a['voxel_span_mm']['v']
    if any(x is None for x in u[:PRIMARY + 1] + v[:PRIMARY + 1]):
        raise ValueError('Unknown voxel span in executed truth: ' + run['run_id'])
    necks = [n['width_mm'] for n in a['pinch_neck_4mm']]
    dv = [x - v[0] for x in v[:PRIMARY + 1]]
    dx, dy, yaw = run['action_mm_mm_deg']
    row = {'run_id': run['run_id'], 'candidate_id': run['candidate_id'], 'group': run['group'],
           'dx_mm': dx, 'dy_mm': dy, 'yaw_deg': yaw, 'tool_offset_mm': run['tool_offset_mm'],
           'neck60_mm': necks[PRIMARY], 'neck62_mm': necks[-1],
           'neck_min_0_60_mm': min([n for n in necks[:PRIMARY + 1] if n is not None], default=None),
           'neck_status60': a['pinch_neck_4mm'][PRIMARY]['status'],
           'r_u60': u[PRIMARY] / ref_u, 'r_v60': v[PRIMARY] / ref_v, 'du60_mm': u[PRIMARY] - u[0],
           'dv60_mm': dv[-1], 'dv_max_mm': max(dv), 'dv_max_ratio': max(dv) / ref_v,
           'raw_u60_mm': a['raw_particle_span_mm']['u'][PRIMARY], 'raw_v60_mm': a['raw_particle_span_mm']['v'][PRIMARY],
           'l4_significant60_by_multiplier': {k: item['significant_0p5pct']
                                              for k, item in a['components']['60']['by_multiplier'].items()}}
    if rows is not None:
        row.update(simulator_labels(rows))
    return row


def describe(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(x):
        return {'n': 0}
    q = np.quantile(x, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    return {'n': int(len(x)), 'min': float(q[0]), 'q10': float(q[1]), 'q25': float(q[2]), 'median': float(q[3]),
            'q75': float(q[4]), 'q90': float(q[5]), 'max': float(q[6]), 'mean': float(x.mean()),
            'sd': float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def different(a, b):
    d = (abs(a['dx_mm'] - b['dx_mm']), abs(a['dy_mm'] - b['dy_mm']), abs(a['yaw_deg'] - b['yaw_deg']))
    return any(x >= t for x, t in zip(d, DIFFERENT_ACTION)), d


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else None


def feasibility(rows, satisfied, final_states):
    chosen = [r for r in rows if satisfied(r)]
    pairs = list(itertools.combinations(chosen, 2))
    diff = [different(a, b) for a, b in pairs]
    ious = [(a['candidate_id'], b['candidate_id'], iou(final_states[a['candidate_id']], final_states[b['candidate_id']]))
            for a, b in pairs if a['candidate_id'] in final_states and b['candidate_id'] in final_states]
    ious = [x for x in ious if x[2] is not None]
    worst = min(ious, key=lambda x: x[2]) if ious else None
    return {'n_satisfying': len(chosen), 'candidates': [r['candidate_id'] for r in chosen],
            'actions': {r['candidate_id']: [r['dx_mm'], r['dy_mm'], r['yaw_deg']] for r in chosen},
            'n_pairs': len(pairs), 'n_pairs_different_action': int(sum(d[0] for d in diff)),
            'action_range': {k: ([min(r[k] for r in chosen), max(r[k] for r in chosen)] if chosen else None)
                             for k in ('dx_mm', 'dy_mm', 'yaw_deg')},
            'final_state_min_iou_pair': list(worst) if worst else None,
            'final_state_iou_median': float(np.median([x[2] for x in ious])) if ious else None}


def a2(rows, final_states, ref_v):
    necks = [r['neck60_mm'] for r in rows if r['neck60_mm'] is not None]
    dvs = [r['dv_max_ratio'] for r in rows]
    w_values = [round(float(np.quantile(necks, q)), 1) for q in QUANTILES] if necks else []
    eta_values = [round(float(np.quantile(dvs, q)), 3) for q in QUANTILES]
    out = {'w_candidates_mm': dict(zip(map(str, QUANTILES), w_values)),
           'eta_candidates_ratio': dict(zip(map(str, QUANTILES), eta_values)),
           'eta_candidates_mm': {str(q): round(e * ref_v, 2) for q, e in zip(QUANTILES, eta_values)},
           'unknown_neck60': [r['candidate_id'] for r in rows if r['neck60_mm'] is None],
           'T1prime': {}, 'T2prime': {}}
    for w in w_values:
        f = feasibility(rows, lambda r, w=w: r['neck60_mm'] is not None and r['neck60_mm'] <= w, final_states)
        f['within_3x_noise_of_w'] = [r['candidate_id'] for r in rows if r['neck60_mm'] is not None
                                     and abs(r['neck60_mm'] - w) <= 3 * PROVISIONAL_NECK_NOISE_MM]
        out['T1prime'][str(w)] = f
        for eta in eta_values:
            out['T2prime']['%s|%s' % (w, eta)] = feasibility(
                rows, lambda r, w=w, eta=eta: r['neck60_mm'] is not None and r['neck60_mm'] <= w
                and r['dv_max_ratio'] <= eta, final_states)
    return out


def plots(rows, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return 'matplotlib unavailable: %r' % exc
    split = [(r.get('sim_components60') or 1) >= 2 for r in rows]
    colors = ['#c2410c' if s else '#2563eb' for s in split]
    for x, y, name in (('neck60_mm', 'dv_max_mm', 'A1_neck60_vs_dvmax.png'), ('r_u60', 'r_v60', 'A1_ru60_vs_rv60.png')):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([np.nan if r[x] is None else r[x] for r in rows], [r[y] for r in rows], c=colors, s=22)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title('orange = simulator label split at frame60', fontsize=9)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter([r['dx_mm'] for r in rows], [r['dy_mm'] for r in rows],
                    c=[np.nan if r['neck60_mm'] is None else r['neck60_mm'] for r in rows], cmap='viridis', s=30)
    for r, s in zip(rows, split):
        if s:
            ax.scatter([r['dx_mm']], [r['dy_mm']], facecolors='none', edgecolors='#c2410c', s=80)
    fig.colorbar(sc, label='neck60 mm')
    ax.set_xlabel('dx mm')
    ax.set_ylabel('dy mm')
    ax.set_title('actions; ring = simulator split', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'A1_actions_neck60.png', dpi=150)
    plt.close(fig)
    return 'A1_neck60_vs_dvmax.png, A1_ru60_vs_rv60.png, A1_actions_neck60.png'


def fmt(x, digits=3):
    if x is None:
        return '—'
    return '%.*f' % (digits, x) if isinstance(x, float) else str(x)


def markdown(result):
    lines = ['# L4-02 A1：现有执行的属性分布', '',
             '- 来源：`%s`（原始粒子真值；未读取任何预测）' % result['source_report'],
             '- 总体：%d 个候选执行（%s）；敏感性运行只在 attributes.csv 中列出，不进分布。' % (
                 len(result['candidates']), '/'.join(CANDIDATE_GROUPS)),
             '- r_u/r_v = 体素跨度 / 初态参照（u %.1f mm，v %.1f mm）；dv = v 跨度相对 frame0 的增量；'
             'dv_max = 0–1.92 s 内最大值；颈宽 = 捏合坐标系 4 mm 切片最小宽度。' % (result['ref_u_mm'], result['ref_v_mm']), '',
             '| 属性 | n | min | q10 | q25 | 中位 | q75 | q90 | max |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in ATTRIBUTES:
        d = result['distribution'][k]
        if d['n']:
            lines.append('| %s | %d | %s |' % (k, d['n'], ' | '.join(
                fmt(d[q]) for q in ('min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'))))
    sim = result['simulator_labels']
    lines += ['', '## 两种“断开”口径（frame60）', '',
              '| 口径 | 1 块 | ≥2 块 |', '|---|---:|---:|',
              '| 模拟器 SceneGraph 标签（DoughNet 训练标签） | %d | %d |' % (sim['components60_eq1'], sim['components60_ge2']),
              '| L4 粒子连通（显著块，全部连通倍数） | %d | %d |' % (sim['l4_one'], sim['l4_ge2']), '',
              '- 模拟器首次标为分裂的帧：%s' % sim['first_split_frames'],
              '- 模拟器标签分裂候选的两块质量比（小/大）：%s' % sim['balances'], '',
              '## 文件', '', '- `attributes.csv`：全部运行逐项数值；散点图：%s。' % result['plots']]
    return '\n'.join(lines) + '\n'


def draft_markdown(result):
    s = result['A2']
    lines = ['# L4-02 A2：T1′/T2′ 规格草案（开发候选，未冻结）', '',
             '阈值 w、η 取本池 %d 个候选执行的分位数（%s），只用于可行性统计，**不是冻结阈值**。'
             'ε 需 ≥ 噪声底的 3 倍（审核二.2）；噪声底以 B1 为准，暂用 S0 的 %.4f mm。' % (
                 len(result['candidates']), '/'.join(map(str, QUANTILES)), PROVISIONAL_NECK_NOISE_MM), '',
             '## 规格（schema 2.0.0 草案）', '', '```json', json.dumps(result['spec_drafts'], ensure_ascii=False, indent=1),
             '```', '', '## T1′ 捏出腰：颈宽(frame60) ≤ w', '',
             '| w (mm) | 满足数 | 动作不同的对数/总对数 | dx 范围 | dy 范围 | yaw 范围 | 终态 IoU 中位 | 最不相似一对 | 距 w ≤3×噪声 |',
             '|---:|---:|---:|---|---|---|---:|---|---:|']
    for w, f in s['T1prime'].items():
        worst = f['final_state_min_iou_pair']
        lines.append('| %s | %d | %d/%d | %s | %s | %s | %s | %s | %d |' % (
            w, f['n_satisfying'], f['n_pairs_different_action'], f['n_pairs'], f['action_range']['dx_mm'],
            f['action_range']['dy_mm'], f['action_range']['yaw_deg'], fmt(f['final_state_iou_median']),
            '—' if worst is None else '%s/%s IoU=%.3f' % tuple(worst), len(f['within_3x_noise_of_w'])))
    lines += ['', '## T2′ 捏出腰且控制摊开：颈宽 ≤ w 且 max_t Δv(t)/v0 ≤ η', '',
              '| w (mm) | η | η (mm) | 满足数 | 动作不同的对数/总对数 | 终态 IoU 中位 | 最不相似一对 |',
              '|---:|---:|---:|---:|---:|---:|---|']
    for key, f in s['T2prime'].items():
        w, eta = key.split('|')
        worst = f['final_state_min_iou_pair']
        lines.append('| %s | %s | %.2f | %d | %d/%d | %s | %s |' % (
            w, eta, float(eta) * result['ref_v_mm'], f['n_satisfying'], f['n_pairs_different_action'], f['n_pairs'],
            fmt(f['final_state_iou_median']), '—' if worst is None else '%s/%s IoU=%.3f' % tuple(worst)))
    lines += ['', '“动作不同”：|Δdx| ≥ %.0f mm 或 |Δdy| ≥ %.0f mm 或 |Δyaw| ≥ %.0f°。' % DIFFERENT_ACTION
              + '终态 IoU 为两次执行 frame60 体素占据（1 mm 网格，2 mm 粒子球）的 IoU。',
              '颈宽 unknown（无腰）的候选不计满足：%s。' % (s['unknown_neck60'] or '无')]
    return '\n'.join(lines) + '\n'


def spec_drafts(ref_u, ref_v):
    common = {'schema_version': '2.0.0-draft', 'object': 'whole_material',
              'reference_frame': {'kind': 'pinch_frame', 'unit': 'mm', 'closing_axis': 'yaw = 30 deg + action yaw',
                                  'scan_axis': 'finger-width axis (perpendicular to closing, in plane)'},
              'initial_reference': {'source': 'privileged_full_material_initial_state', 'u_mm': ref_u, 'v_mm': ref_v},
              'time': {'primary_s': 1.92, 'auxiliary_s': 1.984, 'execution_required_s': 'frames 0..60 (0.032 s)'},
              'measurement': {'neck': {'representation': 'raw_material_particles', 'operator': 'waist_min_width',
                                       'slice_mm': 4.0, 'unknown_if': 'no waist (monotone width profile)'},
                              'v_span': {'representation': 'voxel_union', 'operator': 'extrema', 'grid_mm': 1.0,
                                         'support_radius_mm': 2.0, 'unknown_if': 'grid boundary contact'}},
              'tolerance': {'epsilon_mm': None, 'rule': 'epsilon >= 3 x repeat-noise floor (B1)'},
              'unknown_policy': 'unknown never counts as satisfied',
              'calibration': {'status': 'development_candidate_not_frozen'}}
    t1 = dict(common, task_id='T1prime_pinch_waist', terminal_hard={'id': 'neck_width', 'relation': 'le', 'w_mm': None},
              process_hard=[])
    t2 = dict(common, task_id='T2prime_pinch_waist_limit_spread',
              terminal_hard={'id': 'neck_width', 'relation': 'le', 'w_mm': None},
              process_hard=[{'id': 'v_span_increase', 'relation': 'le', 'eta_ratio_of_v0': None,
                             'over': 'every execution_required_s'}])
    return [t1, t2]


def build(rows, ref_u, ref_v, final_states, source_report, source_sha):
    candidates = [r for r in rows if r['group'] in CANDIDATE_GROUPS]
    l4_one = sum(all(v == 1 for v in r['l4_significant60_by_multiplier'].values()) for r in candidates)
    result = {'kind': 'L4-02_A1A2', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_report': str(source_report), 'source_report_sha256': source_sha,
              'ref_u_mm': ref_u, 'ref_v_mm': ref_v, 'candidates': [r['candidate_id'] for r in candidates],
              'distribution': {k: describe([r[k] for r in candidates]) for k in ATTRIBUTES},
              'simulator_labels': {
                  'components60_eq1': sum(r.get('sim_components60') == 1 for r in candidates),
                  'components60_ge2': sum((r.get('sim_components60') or 0) >= 2 for r in candidates),
                  'l4_one': l4_one, 'l4_ge2': len(candidates) - l4_one,
                  'first_split_frames': sorted({r.get('sim_first_split_frame') for r in candidates} - {None}),
                  'balances': sorted(round(r['sim_part_balance60'], 3) for r in candidates if r.get('sim_part_balance60'))},
              'spec_drafts': spec_drafts(ref_u, ref_v), 'rows': rows}
    result['A2'] = a2(candidates, final_states, ref_v)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--project', type=Path, default=az.PROJECT)
    args = parser.parse_args(argv)
    batch = args.project / az.BATCH_DIRNAME
    analysis = args.analysis or az.latest(batch, 'C-analysis_*', 'data/report/phase_c_report.json')
    report_path = analysis / 'data/report/phase_c_report.json'
    report = read_json(report_path)
    ref_u, ref_v = report['reference_spans_mm']['u'], report['reference_spans_mm']['v']
    runs, _, _ = az.discover(args.project)
    rows, final_states = [], {}
    for rid in sorted(runs):
        truth = analysis / 'data' / 'truth' / rid
        if not (truth / 'attributes.json').is_file():
            continue
        log = az.verified(runs[rid]['evidence'], 'execution/%s/scenes/0000/log.pkl' % rid)
        with log.open('rb') as stream:
            logged = pickle.load(stream)
        row = run_row(runs[rid], truth, ref_u, ref_v, logged)
        rows.append(row)
        if runs[rid]['group'] in CANDIDATE_GROUPS:
            with np.load(truth / 'trajectory.npz', allow_pickle=False) as packet:
                final_states[row['candidate_id']] = packet['classes'][PRIMARY] > 0
        print('ROW', rid, row['neck60_mm'], round(row['dv_max_mm'], 2), row['sim_components60'], flush=True)
    result = build(rows, ref_u, ref_v, final_states, report_path, sha256(report_path))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = analysis / ('l4_02_A1A2_' + stamp)
    out.mkdir()
    result['plots'] = plots([r for r in rows if r['group'] in CANDIDATE_GROUPS], out)
    write_json(out / 'a1a2.json', result)
    keys = ['run_id', 'candidate_id', 'group', 'dx_mm', 'dy_mm', 'yaw_deg', 'neck_status60'] + list(ATTRIBUTES) + [
        'sim_components60', 'sim_components62', 'sim_first_split_frame', 'sim_first_genus_change_frame',
        'sim_part_balance60', 'sim_genus60']
    with (out / 'attributes.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(keys + ['l4_significant60_by_multiplier'])
        for r in rows:
            writer.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), list) else r.get(k) for k in keys]
                            + [json.dumps(r['l4_significant60_by_multiplier'])])
    for name, text in (('attribute_distribution.md', markdown(result)), ('task_v2_draft.md', draft_markdown(result))):
        with (out / name).open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(markdown(result))
    print(draft_markdown(result))
    print('A1A2_DIR', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
