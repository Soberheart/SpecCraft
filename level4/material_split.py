"""D2 measures from raw material particles: components, neck width, volume share.

L4-01 A8.  Inputs are *raw material particle centres in mm* (simulator units
x1000x0.2), never tool/table points or predicted occupancy.  All length
parameters are explicit development parameters, not calibrated task thresholds:

* connectivity: particles i, j are linked when ||p_i - p_j|| <= link_radius_mm.
  The radius is ``multiplier x r0`` where r0 is the *percolation radius of the
  initial (frame0) material of the same run* (smallest radius that makes the
  initial piece one component; ``percolation_radius``).  r0 is fixed per run so
  that stretching never changes the rule.  The job-25837718 material has median
  NN spacing 1.26 mm but r0 = 2.64 mm (irregular sampling), so a median-based
  rule would fragment the untouched initial piece.  ``component_sensitivity``
  reports every multiplier; a count that changes across multipliers is
  ``unstable``.
* significant components: components holding at least ``min_share`` of the mass;
  smaller ones (e.g. single stray particles) are reported separately as debris
  and never counted as a break.
* neck: slice along a fixed scan axis (default u) into bins of ``slice_mm``; the
  cross-section width of a bin is the extent of particle centres along the width
  axis (default v).  A bin is a waist when strictly larger widths exist on both
  sides; the neck is the narrowest waist.  An empty bin between occupied bins is
  a gap (width 0).  Widths are centre extents (no particle radius added).
* volume proxy: per-component particle mass (or count) share of the total.
"""
from __future__ import annotations

import numpy as np

from .geometry import axis_vector

DEFAULT_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0)
DEFAULT_MIN_SHARES = (0.001, 0.005, 0.02)


def _points(points_mm):
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected at least two finite material points of shape (N, 3) in mm')
    return points


def _pairs_within(points, radius, chunk=256):
    """Yield (i, j) index arrays, i < j, with distance <= radius (exact, chunked)."""
    r2 = radius * radius
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        i, j = np.nonzero(d2 <= r2)
        i = i + start
        keep = i < j
        yield i[keep], j[keep]


def nearest_neighbour_spacing(points_mm, chunk=256):
    points = _points(points_mm)
    nearest = np.empty(len(points))
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        d2[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
        nearest[start:start + len(block)] = np.sqrt(d2.min(axis=1))
    return {'median_mm': float(np.median(nearest)), 'p10_mm': float(np.quantile(nearest, .1)),
            'p90_mm': float(np.quantile(nearest, .9)), 'max_mm': float(nearest.max())}


def connected_components(points_mm, link_radius_mm):
    """Return component labels 0..K-1, ordered by decreasing size (ties: lowest index)."""
    points = _points(points_mm)
    if not np.isfinite(link_radius_mm) or link_radius_mm <= 0:
        raise ValueError('link radius must be positive finite mm')
    edges = [np.stack(pair) for pair in _pairs_within(points, float(link_radius_mm)) if len(pair[0])]
    labels = np.arange(len(points))
    if edges:
        i, j = np.concatenate(edges, axis=1)
        while True:  # min-label propagation to a fixed point
            new = labels.copy()
            np.minimum.at(new, i, labels[j])
            np.minimum.at(new, j, labels[i])
            new = new[new]
            if np.array_equal(new, labels):
                break
            labels = new
    roots, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    first = np.array([np.flatnonzero(inverse == k)[0] for k in range(len(roots))])
    order = np.lexsort((first, -sizes))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank[inverse]


def component_summary(points_mm, link_radius_mm, weights=None, min_shares=DEFAULT_MIN_SHARES):
    points = _points(points_mm)
    labels = connected_components(points, link_radius_mm)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(points),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('weights must be positive finite, one per particle')
    count = int(labels.max()) + 1
    share = np.array([weights[labels == k].sum() for k in range(count)]) / weights.sum()
    significant = {str(m): int((share >= m).sum()) for m in min_shares}
    return {'link_radius_mm': float(link_radius_mm), 'components': count,
            'significant_components': significant,
            'debris_particles': {str(m): int(sum((labels == k).sum() for k in range(count) if share[k] < m))
                                 for m in min_shares},
            'particle_counts': [int((labels == k).sum()) for k in range(min(count, 8))],
            'volume_share': [float(s) for s in share[:8]],
            'two_largest_balance': float(share[1] / share[0]) if count > 1 else None,
            'centroids_mm': [points[labels == k].mean(axis=0).tolist() for k in range(min(count, 4))]}


def percolation_radius(points_mm, tolerance_mm=0.005):
    """Smallest link radius (within tolerance) that makes the points one component."""
    points = _points(points_mm)
    low = nearest_neighbour_spacing(points)['max_mm']  # every point needs at least one link
    if connected_components(points, low).max() == 0:
        return float(low)
    high = 2 * low
    while connected_components(points, high).max() > 0:
        high *= 2
    while high - low > tolerance_mm:
        mid = (low + high) / 2
        if connected_components(points, mid).max() == 0:
            high = mid
        else:
            low = mid
    return float(high)


def component_sensitivity(points_mm, r0_mm, multipliers=DEFAULT_MULTIPLIERS, weights=None,
                          min_shares=DEFAULT_MIN_SHARES):
    if not np.isfinite(r0_mm) or r0_mm <= 0:
        raise ValueError('r0 must be positive finite mm (initial percolation radius)')
    rows = [dict(component_summary(points_mm, m * r0_mm, weights, min_shares), multiplier=float(m))
            for m in multipliers]
    counts = sorted({row['components'] for row in rows})
    significant = sorted({row['significant_components'][str(m)] for row in rows for m in min_shares})
    return {'r0_mm': float(r0_mm), 'rows': rows, 'component_counts': counts,
            'significant_component_counts': significant,
            'status': 'stable' if len(counts) == 1 else 'unstable',
            'significant_status': 'stable' if len(significant) == 1 else 'unstable'}


def cross_section_profile(points_mm, scan_axis=(1, 0, 0), width_axis=(0, 1, 0), slice_mm=4.0):
    points = _points(points_mm)
    scan, width = axis_vector(scan_axis), axis_vector(width_axis)
    if abs(float(scan @ width)) > 1e-10:
        raise ValueError('scan and width axes must be orthogonal')
    if not np.isfinite(slice_mm) or slice_mm <= 0:
        raise ValueError('slice thickness must be positive finite mm')
    s, w = points @ scan, points @ width
    edges = np.arange(s.min(), s.max() + slice_mm, slice_mm)
    if len(edges) < 2:
        edges = np.array([s.min(), s.min() + slice_mm])
    index = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, len(edges) - 2)
    widths, counts = [], []
    for k in range(len(edges) - 1):
        member = w[index == k]
        counts.append(int(member.size))
        widths.append(float(member.max() - member.min()) if member.size else 0.0)
    return {'scan_axis': scan.tolist(), 'width_axis': width.tolist(), 'slice_mm': float(slice_mm),
            'centres_mm': ((edges[:-1] + edges[1:]) / 2).tolist(), 'widths_mm': widths, 'counts': counts}


def neck(points_mm, scan_axis=(1, 0, 0), width_axis=(0, 1, 0), slice_mm=4.0):
    profile = cross_section_profile(points_mm, scan_axis, width_axis, slice_mm)
    widths = np.asarray(profile['widths_mm']); counts = np.asarray(profile['counts'])
    waist = []
    for k in range(1, len(widths) - 1):
        if widths[:k].max() > widths[k] and widths[k + 1:].max() > widths[k]:
            waist.append(k)
    gaps = [k for k in range(1, len(widths) - 1) if counts[k] == 0]
    if not waist:
        return {'status': 'no_waist', 'neck_width_mm': None, 'neck_position_mm': None,
                'gap_slices': gaps, 'profile': profile}
    k = min(waist, key=lambda i: (widths[i], i))
    return {'status': 'gap' if counts[k] == 0 else 'waist', 'neck_width_mm': float(widths[k]),
            'neck_position_mm': float(profile['centres_mm'][k]), 'gap_slices': gaps,
            'max_width_mm': float(widths.max()), 'profile': profile}


def d2_measures(points_mm, r0_mm, weights=None, multipliers=DEFAULT_MULTIPLIERS,
                slice_options_mm=(3.0, 4.0, 6.0), scan_axis=(1, 0, 0), width_axis=(0, 1, 0)):
    """All D2 quantities with their sensitivity rows; nothing here is a task verdict."""
    return {'scope': 'development D2 measures from raw material particles; not task thresholds',
            'components': component_sensitivity(points_mm, r0_mm, multipliers, weights),
            'neck': {str(h): {k: v for k, v in neck(points_mm, scan_axis, width_axis, h).items()
                              if k != 'profile'} for h in slice_options_mm}}
