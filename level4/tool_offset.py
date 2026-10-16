"""Whole-trajectory tool offset (delta_x, delta_y) for the S0-S4 sensitivity runs.

L4-01 A7.  The offset emulates an unmodelled execution-interface error: the
executed scene's grasp anchor ``actions.grasp.to_pos`` is shifted by
(delta_x, delta_y) mm in the world frame while the registered *action* (used for
prediction) is unchanged.  Every gripper pose in the run is generated from
``to_pos`` (``Push.get_from = to_pos + R from_offset``; ``get_to = to_pos``) and
the per-axis PID acts on pose errors, which are translation invariant, so the
whole tool trajectory is translated by the offset.  A zero offset returns the
scene unchanged (no field is rewritten), so S0 differs from the nominal scene
only in its log directory.
"""
from __future__ import annotations

import copy
import math


def apply_tool_offset(scene, delta_mm, unit_to_meter):
    """Return a new scene dict whose tool trajectory is shifted by delta_mm (x, y)."""
    if len(delta_mm) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in delta_mm):
        raise ValueError('tool offset must be two finite mm values (delta_x, delta_y)')
    if not (isinstance(unit_to_meter, (int, float)) and math.isfinite(unit_to_meter) and unit_to_meter > 0):
        raise ValueError('invalid unit_to_meter')
    if any(abs(v) > 1.0 for v in delta_mm):
        raise ValueError('sensitivity offsets above 1 mm are outside the S0-S4 design')
    shifted = copy.deepcopy(scene)
    if delta_mm[0] == 0 and delta_mm[1] == 0:
        return shifted
    position = shifted['actions']['grasp']['to_pos']
    if len(position) != 3:
        raise ValueError('grasp.to_pos must be xyz')
    position[0] = float(position[0]) + float(delta_mm[0]) / (1000 * unit_to_meter)
    position[1] = float(position[1]) + float(delta_mm[1]) / (1000 * unit_to_meter)
    return shifted
