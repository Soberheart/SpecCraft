"""L4-02 A4 (CPU, login node): where the L4 initial state and pinch sit in the DoughNet training data.

Reads the processed training ``dataset.h5`` (all groups, e.g. train/val/test; the
'real' group has no meshes and is skipped).  Per scene it summarizes, from the
stored meshes and simulator topology labels only:

* initial shape (frame 0): bbox extents (mm), height, number of components and
  genus per component (simulator label), a coarse shape class;
* the topology change over the sequence (split = more components at the end than
  at frame 0; merge; genus change) and the frame of the first change;
* the pinch: finger closing direction (yaw, from the two finger point clusters of
  ``ee_observed``), closing centre relative to the frame-0 object centroid, and
  the final finger gap.

The L4 initial state (from ``--l4-log`` frame 0, the same rows the batch used) is
summarized the same way and placed among the training scenes: nearest training
scenes by initial bbox + genus, and the split rate among similar scenes.

    python -B scripts/l4_02_training_distribution.py --dataset <dataset.h5> --l4-log <log.pkl> --out <new dir>
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

SIM_TO_MM = 200.0


def finger_pose(points):
    """Closing axis yaw (deg, mod 180), centre (mm) and gap (mm) from 1024 tool samples (2 x 512)."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 4:
        return None
    left, right = p[:len(p) // 2], p[len(p) // 2:]  # get_ee_samples stacks finger_left then finger_right
    cl, cr = left.mean(axis=0), right.mean(axis=0)
    axis = (cr - cl)[:2]
    norm = float(np.hypot(*axis))
    if norm == 0:
        return None
    a = axis / norm
    gap = float(((right[:, :2] - cl[:2]) @ a).min() - ((left[:, :2] - cl[:2]) @ a).max())
    yaw = math.degrees(math.atan2(a[1], a[0])) % 180.0
    return {'yaw_deg': yaw, 'centre_mm': ((cl + cr) / 2 * SIM_TO_MM).tolist(), 'gap_mm': gap * SIM_TO_MM}


def mesh_extent(verts, faces):
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces)
    tri = v[f]
    used = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0
    pts = v[np.unique(f[used])] * SIM_TO_MM
    return pts.min(axis=0), pts.max(axis=0), pts.mean(axis=0)


def shape_class(extent_mm, genus):
    """Coarse label from the simulator genus label and the bbox aspect (documented heuristic)."""
    comps = len(genus)
    if comps >= 2:
        return 'multi_%d' % comps
    if genus[0] >= 1:
        return 'ring_genus%d' % genus[0]
    x, y, z = sorted(extent_mm[:2])[::-1] + [extent_mm[2]]
    if x > 2.0 * y:
        return 'elongated'
    return 'compact'


def topology(genus_rows):
    labels = [sorted(int(g) for g in row if g >= 0) for row in genus_rows]
    comps = [len(x) for x in labels]
    change = next((i for i, x in enumerate(labels) if x != labels[0]), None)
    return {'genus0': labels[0], 'genus_end': labels[-1], 'components0': comps[0], 'components_end': comps[-1],
            'split': comps[-1] > comps[0], 'merge': comps[-1] < comps[0], 'genus_changed': labels[-1] != labels[0],
            'first_change_frame': change}


def scene_summary(verts0, faces0, genus_rows, tools0, tools_end):
    low, high, centroid = mesh_extent(verts0, faces0)
    extent = (high - low).tolist()
    topo = topology(genus_rows)
    start, end = finger_pose(tools0), finger_pose(tools_end)
    out = {'extent_mm': extent, 'height_mm': float(high[2]), 'z_min_mm': float(low[2]), 'centroid_mm': centroid.tolist(),
           'shape_class': shape_class(extent, topo['genus0']), **topo}
    if end is not None:
        out['pinch'] = {'yaw_deg': end['yaw_deg'], 'final_gap_mm': end['gap_mm'],
                        'centre_offset_mm': (np.asarray(end['centre_mm'][:2]) - centroid[:2]).tolist(),
                        'initial_gap_mm': None if start is None else start['gap_mm']}
    return out


def l4_summary(rows):
    p = np.asarray(rows[0]['obj']['particles']['pos'], dtype=float) * SIM_TO_MM
    genus = list(rows[0]['topology']['genus'])
    extent = (p.max(axis=0) - p.min(axis=0)).tolist()
    r = np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean())
    labels = [sorted(r_['topology']['genus']) for r_ in rows]
    comps = [int(r_['topology']['components']) for r_ in rows]
    return {'extent_mm': extent, 'height_mm': float(p[:, 2].max()), 'z_min_mm': float(p[:, 2].min()),
            'centroid_mm': p.mean(axis=0).tolist(), 'radial_min_mm': float(r.min()), 'radial_max_mm': float(r.max()),
            'genus0': genus, 'shape_class': shape_class(extent, genus), 'components_end': comps[-1],
            'genus_end': labels[-1], 'split': comps[-1] > comps[0],
            'first_change_frame': next((i for i, x in enumerate(labels) if x != labels[0]), None)}


def place(l4, scenes, k=10):
    same_topology = [s for s in scenes if s['genus0'] == sorted(l4['genus0'])]
    def dist(s):
        return float(np.linalg.norm(np.asarray(s['extent_mm']) - np.asarray(l4['extent_mm'])))
    nearest = sorted(same_topology, key=dist)[:k]
    similar = [s for s in same_topology if dist(s) <= 10.0]
    return {'n_same_initial_genus': len(same_topology), 'n_similar_bbox_within_10mm': len(similar),
            'split_rate_similar': (sum(s['split'] for s in similar) / len(similar)) if similar else None,
            'genus_change_rate_similar': (sum(s['genus_changed'] for s in similar) / len(similar)) if similar else None,
            'nearest': [{'group': s['group'], 'row': s['row'], 'extent_mm': s['extent_mm'], 'genus0': s['genus0'],
                         'genus_end': s['genus_end'], 'split': s['split'], 'pinch': s.get('pinch'),
                         'bbox_distance_mm': dist(s)} for s in nearest]}


def aggregate(scenes):
    classes = Counter(s['shape_class'] for s in scenes)
    by_class = {}
    for name in sorted(classes):
        member = [s for s in scenes if s['shape_class'] == name]
        ext = np.asarray([s['extent_mm'] for s in member])
        by_class[name] = {'n': len(member), 'split_rate': float(np.mean([s['split'] for s in member])),
                          'genus_change_rate': float(np.mean([s['genus_changed'] for s in member])),
                          'extent_mm_median': np.median(ext, axis=0).tolist(),
                          'extent_mm_min': ext.min(axis=0).tolist(), 'extent_mm_max': ext.max(axis=0).tolist(),
                          'final_gap_mm_median': float(np.median([s['pinch']['final_gap_mm'] for s in member
                                                                  if s.get('pinch')] or [np.nan])),
                          'first_change_frames': sorted(Counter(s['first_change_frame'] for s in member
                                                                if s['first_change_frame'] is not None).items())[:20]}
    return {'n_scenes': len(scenes), 'shape_classes': dict(classes), 'by_class': by_class,
            'initial_genus': {json.dumps(k): v for k, v in Counter(json.dumps(s['genus0']) for s in scenes).items()},
            'transitions': dict(Counter('%s -> %s' % (s['genus0'], s['genus_end']) for s in scenes).most_common(20))}


def read_dataset(path, max_scenes=None):
    import hdf5plugin  # noqa: F401  (registers the Blosc2 filter)
    import h5py
    scenes = []
    with h5py.File(path, 'r', libver='latest') as f:
        groups = [g for g in f.keys() if isinstance(f[g], h5py.Group)] or ['']
        for g in groups:
            d = f[g] if g else f
            if 'obj_verts' not in d:
                continue
            n = d['scene'].shape[0] if max_scenes is None else min(max_scenes, d['scene'].shape[0])
            for i in range(n):
                genus = d['genus'][i]
                scenes.append(dict(scene_summary(d['obj_verts'][i, 0], d['obj_faces'][i, 0], genus,
                                                 d['ee_observed'][i, 0], d['ee_observed'][i, -1]),
                                   group=g, row=i, scene_id=int(d['scene'][i, 0])))
    return scenes


def markdown(result):
    a, l4, pl = result['aggregate'], result['l4'], result['placement']
    lines = ['# L4-02 A4：训练分布检查', '', '- 数据：`%s`（%d 个场景，组：%s）' % (
        result['dataset'], a['n_scenes'], ', '.join(sorted({s['group'] for s in result['scenes']}))),
             '- 口径：初态取 frame0 网格（去除零面积填充）；拓扑取模拟器 SceneGraph 标签（训练标签）；'
             '捏合姿态取末帧 ee_observed 两指点云。形状类别为启发式（genus 标签 + 包围盒长宽比）。', '',
             '## 形状类别', '', '| 类别 | 场景数 | 分裂比例 | genus 改变比例 | 包围盒中位 (mm) | 末帧指间距中位 (mm) |',
             '|---|---:|---:|---:|---|---:|']
    for name, s in a['by_class'].items():
        lines.append('| %s | %d | %.2f | %.2f | %s | %.1f |' % (
            name, s['n'], s['split_rate'], s['genus_change_rate'], [round(x, 1) for x in s['extent_mm_median']],
            s['final_gap_mm_median']))
    lines += ['', '## 拓扑变化（前 20）', '', '| 初态 genus → 末态 genus | 场景数 |', '|---|---:|']
    lines += ['| %s | %d |' % kv for kv in a['transitions'].items()]
    lines += ['', '## L4 初态', '',
              '- 包围盒 %s mm，高 %.1f mm，环内半径 %.1f mm、外半径 %.1f mm；genus 标签 %s → 类别 %s。' % (
                  [round(x, 1) for x in l4['extent_mm']], l4['height_mm'], l4['radial_min_mm'], l4['radial_max_mm'],
                  l4['genus0'], l4['shape_class']),
              '- 名义动作执行后模拟器标签：末态 genus %s，分裂=%s，首次拓扑变化帧 %s。' % (
                  l4['genus_end'], l4['split'], l4['first_change_frame']),
              '- 训练中初态 genus 相同的场景 %d 个；包围盒差 ≤10 mm 的 %d 个，其中分裂比例 %s、genus 改变比例 %s。' % (
                  pl['n_same_initial_genus'], pl['n_similar_bbox_within_10mm'],
                  'n/a' if pl['split_rate_similar'] is None else '%.2f' % pl['split_rate_similar'],
                  'n/a' if pl['genus_change_rate_similar'] is None else '%.2f' % pl['genus_change_rate_similar']), '',
              '| 最近训练场景 | 包围盒 (mm) | genus 0→末 | 分裂 | 捏合 yaw (°) | 中心偏移 (mm) | 末帧指距 (mm) | 距离 (mm) |',
              '|---|---|---|---|---:|---|---:|---:|']
    for s in pl['nearest']:
        p = s.get('pinch') or {}
        lines.append('| %s/%d | %s | %s→%s | %s | %s | %s | %s | %.1f |' % (
            s['group'], s['row'], [round(x, 1) for x in s['extent_mm']], s['genus0'], s['genus_end'], s['split'],
            '—' if not p else '%.0f' % p['yaw_deg'], '—' if not p else [round(x, 1) for x in p['centre_offset_mm']],
            '—' if not p else '%.1f' % p['final_gap_mm'], s['bbox_distance_mm']))
    lines += ['', '## 候选“可断初态”', '', '按“训练中分裂比例高、且初态类别在训练中常见”排序（自动列出，需人工复核）：', '']
    ranked = sorted(a['by_class'].items(), key=lambda kv: (-kv[1]['split_rate'], -kv[1]['n']))
    for name, s in ranked[:3]:
        lines.append('- **%s**：%d 个场景，分裂比例 %.2f，包围盒中位 %s mm。' % (
            name, s['n'], s['split_rate'], [round(x, 1) for x in s['extent_mm_median']]))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--l4-log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-scenes', type=int, help='per group, for a quick look only')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    scenes = read_dataset(args.dataset, args.max_scenes)
    with args.l4_log.open('rb') as stream:
        rows = pickle.load(stream)
    l4 = l4_summary(rows)
    result = {'kind': 'L4-02_A4_training_distribution', 'dataset': str(args.dataset), 'l4_log': str(args.l4_log),
              'aggregate': aggregate(scenes), 'l4': l4, 'placement': place(l4, scenes), 'scenes': scenes}
    with (args.out / 'training_distribution.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    text = markdown(result)
    with (args.out / 'training_distribution_check.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(text)
    print('A4_DIR', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
