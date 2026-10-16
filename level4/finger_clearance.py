"""Initial finger/material clearance for a candidate grasp pose (CPU geometry).

L4-01 A6.  The gripper starts at its grasp pose at table level with the recorded
initial opening, so a candidate whose finger cuboid already contains material
particles at frame0 would start interpenetrated.  The pool's historical
``legality_scope`` only checks pose bounds; this adds a geometric pre-screen
from the scene config.  It is a screening diagnostic, not a simulator collision
model: particle centres only, no particle radius unless ``margin_mm`` is set.

Geometry (all mm, world frame, from ``source_scene.yaml``):
* ee centre = nominal grasp ``to_pos`` + (dx, dy, 0); yaw = nominal yaw + yaw_deg
* finger entity positions = centre -/+ R(open/2, 0, 0) (inner faces)
* each finger cuboid = board.obj [-0.5,0.5]^2 x [-0.125,0.125] scaled by
  ``geom.scale`` and shifted by ``geom.offset_pos`` in the ee frame.
"""
from __future__ import annotations

import math

import numpy as np

SIM_TO_MM = 1000 * 0.2
BOARD_HALF_EXTENT_UNSCALED = (0.5, 0.5, 0.125)  # board-board.provenance bounds, rounded (<0.001 sim)


def _yaw_from_wxyz(q):
    w, x, y, z = (float(v) for v in q)
    if abs(x) > 1e-9 or abs(y) > 1e-9:
        raise ValueError('Grasp quaternion is not a pure world-z yaw')
    return math.degrees(2 * math.atan2(z, w))


def finger_geometry(scene):
    """Extract finger cuboids and grasp pose from a parsed scene config (dict)."""
    grasp = scene['actions']['grasp']
    entities = scene['ee']['entities']
    fingers = []
    for name, sign in (('finger_left', -1.0), ('finger_right', 1.0)):
        geom = entities[name]['geom']
        half = np.multiply(BOARD_HALF_EXTENT_UNSCALED, geom['scale']) * SIM_TO_MM
        offset = np.asarray(geom['offset_pos'], dtype=float) * SIM_TO_MM
        if any(abs(v) > 1e-12 for v in geom.get('offset_euler', (0, 0, 0))):
            raise ValueError('Rotated finger offsets are not supported')
        fingers.append({'name': name, 'sign': sign, 'half_extent_mm': half.tolist(), 'offset_mm': offset.tolist()})
    return {'centre_mm': (np.asarray(grasp['to_pos'], dtype=float) * SIM_TO_MM).tolist(),
            'yaw_deg': _yaw_from_wxyz(grasp['to_quat']),
            'open_mm': float(grasp['init_d']) * SIM_TO_MM, 'fingers': fingers}


def _box_signed_distance(local, half):
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def initial_clearance(particles_mm, geometry, action, margins_mm=(0.0, 1.0, 2.0)):
    """Signed clearance (mm, negative = inside) of frame0 particles to both fingers."""
    points = np.asarray(particles_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid particle array')
    dx, dy, yaw = (float(v) for v in action)
    psi = math.radians(geometry['yaw_deg'] + yaw)
    rot = np.array([[math.cos(psi), -math.sin(psi), 0.0], [math.sin(psi), math.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    centre = np.asarray(geometry['centre_mm']) + [dx, dy, 0.0]
    rows = {}
    for finger in geometry['fingers']:
        local_centre = np.array([finger['sign'] * geometry['open_mm'] / 2, 0.0, 0.0]) + finger['offset_mm']
        local = (points - centre) @ rot - local_centre  # world -> ee frame (rot is orthonormal)
        distance = _box_signed_distance(local, np.asarray(finger['half_extent_mm']))
        rows[finger['name']] = {'min_clearance_mm': float(distance.min()),
                                'particles_within': {str(m): int((distance <= m).sum()) for m in margins_mm},
                                'box_centre_world_mm': (centre + rot @ local_centre).tolist()}
    worst = min(row['min_clearance_mm'] for row in rows.values())
    return {'action_mm_mm_deg': [dx, dy, yaw], 'fingers': rows, 'min_clearance_mm': worst,
            'interpenetrating_at_frame0': worst <= 0.0,
            'scope': 'frame0 particle centres vs finger cuboids; screening only, not a simulator collision test'}
