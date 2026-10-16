"""Model tool input (``ee_observed``) from logged gripper states, as in training.

The Dyn model was trained on ``sim/process.py``: per frame, both finger meshes
are posed at the logged ``ee`` summary (``pos_left``/``pos_right``/``quat``),
z-scaled to 0.15 and sampled uniformly (512 points per finger).  To build the
v3 nominal tool input without re-implementing that code, the three functions
``load_ee_mesh``, ``get_ee_in_frame`` and ``get_ee_samples`` are extracted from
the archived runtime's ``sim/process.py`` by AST and executed in a namespace
with numpy/open3d/copy/os only (the module's heavy imports are never run).
Sampling is random in open3d; it is seeded per frame (seed + frame) and the
seed is recorded.  Points are simulator units, shape (frames, 1024, 3).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path

import numpy as np

FUNCTIONS = ('load_ee_mesh', 'get_ee_in_frame', 'get_ee_samples')
ZMAX = 0.15
NUM_POINTS = 1024


def extract_process_functions(process_path, o3d):
    """Return {name: function} executed from the exact source text of sim/process.py."""
    process_path = Path(process_path)
    source = process_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS}
    if set(nodes) != set(FUNCTIONS):
        raise ValueError('sim/process.py no longer defines the training tool sampling functions')
    namespace = {'np': np, 'o3d': o3d, 'copy': copy, 'os': os, '__file__': str(process_path)}
    module = ast.Module(body=[nodes[name] for name in FUNCTIONS], type_ignores=[])
    exec(compile(module, str(process_path), 'exec'), namespace)
    segments = {name: ast.get_source_segment(source, nodes[name]) for name in FUNCTIONS}
    return ({name: namespace[name] for name in FUNCTIONS},
            {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in segments.items()})


def observed_tool_sequence(rows, frames, scene_cfg, process_path, seed, o3d=None):
    """ee_observed for the given logged frames; returns (points, provenance)."""
    if o3d is None:
        import open3d as o3d
    functions, source_hashes = extract_process_functions(process_path, o3d)
    if scene_cfg.ee.type != 'gripper':
        raise ValueError('Expected the two-finger gripper used in training')
    meshes = [functions['load_ee_mesh'](scene_cfg.ee.entities.finger_left),
              functions['load_ee_mesh'](scene_cfg.ee.entities.finger_right)]
    clouds = []
    for frame in frames:
        row = rows[frame]
        if int(row['frame_idx']) != frame:
            raise ValueError('Log row order differs from frame index')
        _, fingers = functions['get_ee_in_frame'](meshes, row['ee'])
        o3d.utility.random.seed(int(seed) + int(frame))
        points = functions['get_ee_samples'](fingers, ZMAX, NUM_POINTS)
        if points.shape != (NUM_POINTS, 3) or not np.isfinite(points).all():
            raise ValueError('Unexpected tool sample at frame %d' % frame)
        clouds.append(points.astype(np.float32))
    return np.stack(clouds), {'process_function_sha256': source_hashes, 'seed': int(seed),
                              'frames': [int(f) for f in frames], 'zmax': ZMAX, 'num_points': NUM_POINTS,
                              'pose_source': "logged ee summary (pos_left/pos_right/quat), as sim/process.py"}


def compare_clouds(a, b):
    """Per-frame geometric agreement of two random samplings of the same posed fingers (mm)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('Tool sequences must share shape (T, N, 3)')
    rows = []
    for t in range(len(a)):
        rows.append({'bbox_max_abs_mm': float(np.max(np.abs(np.r_[a[t].min(0) - b[t].min(0), a[t].max(0) - b[t].max(0)]))) * 200,
                     'centroid_abs_mm': float(np.max(np.abs(a[t].mean(0) - b[t].mean(0)))) * 200})
    return {'frames': rows, 'max_bbox_mm': max(r['bbox_max_abs_mm'] for r in rows),
            'max_centroid_mm': max(r['centroid_abs_mm'] for r in rows)}
