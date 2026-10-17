"""L4-02 A3: simulator topology labels vs L4 particle connectivity on the same logs (CPU, numpy only).

For each log and frame: the simulator's SceneGraph label (components, genus, particle
part ids), the L4 connected-component count at {1, 1.25, 1.5, 2} x r0 (r0 = frame-0
percolation radius), and, where the simulator reports >=2 parts, the minimum distance
between particles of different simulator parts and the number of cross-part particle
pairs within r0 (how much material still touches across the simulator's 'split').

    python -B scripts/l4_02_component_check.py --log NAME=path/to/log.pkl [...] --out <new dir>
"""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.material_split import component_sensitivity, neck, percolation_radius  # noqa: E402

SIM_TO_MM = 200.0
FRAMES = (0, 30, 45, 49, 50, 54, 56, 57, 60, 62)


def cross_part_contact(points, parts, radius, chunk=512):
    """Minimum distance between different simulator parts and pair count within radius."""
    labels = np.unique(parts)
    if len(labels) < 2:
        return {'min_distance_mm': None, 'pairs_within_r0': None}
    best, pairs = math.inf, 0
    a = points[parts == labels[0]]
    b = points[parts != labels[0]]
    for start in range(0, len(a), chunk):
        d2 = ((a[start:start + chunk, None, :] - b[None, :, :]) ** 2).sum(-1)
        best = min(best, float(np.sqrt(d2.min())))
        pairs += int((d2 <= radius * radius).sum())
    return {'min_distance_mm': best, 'pairs_within_r0': pairs}


def analyse(rows, yaw_deg=30.0, frames=FRAMES):
    c_axis = (math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0)
    d_axis = (-c_axis[1], c_axis[0], 0.0)
    r0 = percolation_radius(np.asarray(rows[0]['obj']['particles']['pos'], float) * SIM_TO_MM)
    out = []
    for f in frames:
        row = rows[f]
        p = np.asarray(row['obj']['particles']['pos'], float) * SIM_TO_MM
        parts = np.asarray(row['obj']['particles']['idx'])
        sens = component_sensitivity(p, r0)
        n = neck(p, d_axis, c_axis, 4.0)
        ids, counts = np.unique(parts, return_counts=True)
        out.append({'frame': f, 'step': int(row['step']),
                    'simulator': {'components': row['topology']['components'], 'genus': list(row['topology']['genus']),
                                  'part_ids': ids.tolist(), 'part_particles': counts.tolist()},
                    'l4_components_by_multiplier': {str(r['multiplier']): r['components'] for r in sens['rows']},
                    'l4_significant_0p5pct_by_multiplier': {str(r['multiplier']): r['significant_components']['0.005']
                                                            for r in sens['rows']},
                    'pinch_neck_mm': n['neck_width_mm'],
                    'cross_part': cross_part_contact(p, parts, r0)})
    return {'r0_mm': r0, 'frames': out}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', action='append', required=True, help='NAME=path/to/log.pkl')
    parser.add_argument('--yaw', type=float, default=30.0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'kind': 'L4-02_A3_component_definition_check', 'logs': {}}
    for item in args.log:
        name, path = item.split('=', 1)
        with open(path, 'rb') as stream:
            rows = pickle.load(stream)
        result['logs'][name] = dict(analyse(rows, args.yaw), source=path, frames_logged=len(rows))
        for r in result['logs'][name]['frames']:
            print(name, r['frame'], r['simulator']['components'], r['simulator']['genus'], r['simulator']['part_particles'],
                  'L4', r['l4_components_by_multiplier'], 'neck', r['pinch_neck_mm'] and round(r['pinch_neck_mm'], 2),
                  'cross', r['cross_part'], flush=True)
    with (args.out / 'component_check.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
