"""L4-01 POST-HOC EXPLORATORY D1 (not pre-registered; report to the overview as a deviation).

Why: in the pre-registered Phase C analysis every Dyn prediction at t>0 occupies
the z-min face of the frozen query grid (material predicted below the table), so
level4.geometry.measure() returns unknown and D1 has n=0 in every cell.  That
result stands and is reported as is.  This script adds two clearly labelled
exploratory readings of the SAME frozen predictions, written to a new directory
next to the existing analysis; nothing pre-registered is changed or overwritten.

  E1 lateral_faces_only : unknown only if occupied cells touch an x/y face of the
                          grid (or the packet says incomplete/outside_support);
                          z-face contact is ignored.  u/v extents are unchanged.
  E2 clip_below_k0_floor: drop predicted cells below the lowest occupied z layer of
                          the same prediction at k=0 (the model's own input
                          observation; no execution truth), then apply the
                          frozen all-face boundary rule.

Execution truth (exec_common) is read from the existing phase_c_report.json and is
used only for residuals, never for selection.  Also reports, per prediction frame,
the share of occupied cells below the k0 floor (a D4-relevant diagnostic).

    python -B scripts/explore_level4_d1_lateral.py [--analysis <C-analysis_dir>]
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.artifacts import load_grid, read_json, sha256, write_json  # noqa: E402
from level4.geometry import Geometry, measure  # noqa: E402
from level4 import batch_analysis as ba  # noqa: E402
from scripts import analyze_level4_batch as az  # noqa: E402

LABEL = 'POST_HOC_EXPLORATORY_NOT_PREREGISTERED'
RULES = ('E1_lateral_faces_only', 'E2_clip_below_k0_floor')


def lateral_touch(occupied):
    return any(np.take(occupied, e, axis=i).any() for i in (0, 1) for e in (0, -1))


def all_touch(occupied):
    return any(np.take(occupied, e, axis=i).any() for i in range(3) for e in (0, -1))


def spans(grid, classes, complete, outside, rule):
    counts = tuple(int(c) for c in grid['counts'])
    query, step, scale = grid['query'], np.asarray(grid['spacing'], float), float(grid['unit_to_meter'])
    occ = [(c > 0).reshape(counts) for c in classes]
    floor = int(np.nonzero(occ[0].any(axis=(0, 1)))[0].min()) if occ[0].any() else 0
    out = {'u': [], 'v': [], 'reason': [], 'below_floor_share': [], 'k0_floor_index': floor}
    for o, ok, outs in zip(occ, complete, outside):
        n = int(o.sum())
        out['below_floor_share'].append(float(o[:, :, :floor].sum() / n) if n else None)
        if rule == RULES[1]:
            o = o.copy(); o[:, :, :floor] = False
            truncated = all_touch(o)
        else:
            truncated = lateral_touch(o)
        geometry = Geometry(query[o.reshape(-1)], scale, 'voxel_union', True, bool(ok), step,
                            bool(truncated or outs))
        mu, mv = measure(geometry, [1, 0, 0]), measure(geometry, [0, 1, 0])
        out['u'].append(mu['value'] if mu['status'] == 'measured' else None)
        out['v'].append(mv['value'] if mv['status'] == 'measured' else None)
        out['reason'].append(mu['reason'] or mv['reason'])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--project', type=Path, default=az.PROJECT)
    args = parser.parse_args(argv)
    batch = args.project / az.BATCH_DIRNAME
    analysis = args.analysis or az.latest(batch, 'C-analysis_*', 'data/report/phase_c_report.json')
    report_path = analysis / 'data/report/phase_c_report.json'
    report = read_json(report_path)
    dyn = Path(report['dyn_job_dir']) / 'data'
    if sha256(dyn / 'prediction_manifest.json') != report['prediction_manifest_sha256']:
        raise ValueError('prediction_manifest changed since the pre-registered report')
    manifest = read_json(dyn / 'prediction_manifest.json')
    hashes = manifest['predictions_sha256']
    _, pool, _ = az.discover(args.project)
    grid = load_grid(pool)
    ref_u, ref_v = report['reference_spans_mm']['u'], report['reference_spans_mm']['v']
    result = {'kind': 'L4-01_exploratory_d1', 'label': LABEL, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_report': str(report_path), 'source_report_sha256': sha256(report_path),
              'prediction_manifest_sha256': report['prediction_manifest_sha256'], 'rules': {}, 'D1': {}}
    for rule in RULES:
        table = []
        for row in report['candidates']:
            new = {k: row[k] for k in ('candidate_id', 'group', 'execution') if k in row}
            if 'exec_common' in row:
                new['exec_common'] = row['exec_common']
            for variant in manifest['variants']:
                path = dyn / 'predictions' / variant / row['candidate_id'] / 'trajectory.npz'
                if sha256(path) != hashes['%s/%s' % (variant, row['candidate_id'])]['trajectory.npz']:
                    raise ValueError('Frozen prediction changed: %s' % path)
                with np.load(path, allow_pickle=False) as p:
                    s = spans(grid, p['classes'], p['complete'], p['outside_support'], rule)
                    times = [float(t) for t in p['times_s']]
                new['pred_' + variant] = ba.task_attributes(times, s['u'], s['v'], ref_u, ref_v)
                new['pred_' + variant + '_spans'] = s
            table.append(new)
        result['rules'][rule] = table
        for population, groups in (('random_groups', az.RANDOM_GROUPS),
                                   ('random_plus_d2', az.RANDOM_GROUPS + ('d2_edge',))):
            for variant in manifest['variants']:
                for attribute in ('T1_r_u_end', 'T2_max_abs_v_dev'):
                    key = '%s|%s|%s|%s' % (rule, population, variant, attribute)
                    result['D1'][key] = az.d1_block(table, groups, variant, attribute)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = analysis / ('exploratory_d1_' + stamp)
    out.mkdir()
    write_json(out / 'exploratory_d1.json', result)
    lines = [LABEL + ' source=' + str(report_path)]
    for key, value in result['D1'].items():
        if 'verdict' in value:
            t = value['top_vs_rest']
            lines.append('D1x %-62s n=%2d top8-rest=%+.4f p=%.4f rho=%+.3f -> %s' % (
                key, value['n'], t['selected_minus_rest_mean'], t['p_one_sided_optimistic'],
                value['spearman_score_rank_vs_residual']['rho'], value['verdict']['verdict']))
        else:
            lines.append('D1x %-62s %s n=%d' % (key, value['status'], value['n']))
    for rule in RULES:
        shares = [s for r in result['rules'][rule] for s in r['pred_v3_nominal_spans']['below_floor_share'][-1:]
                  if s is not None]
        if shares:
            lines.append('%s v3 k12 share of predicted cells below k0 floor: median=%.3f max=%.3f n=%d' % (
                rule, float(np.median(shares)), max(shares), len(shares)))
    with (out / 'summary.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('EXPLORATORY_DIR', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
