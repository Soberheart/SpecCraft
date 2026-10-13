"""Deterministic material-support measurements, in millimetres.

Primary representation: equal-volume occupied cells of an axis-aligned grid.
No tool masks, PCA rotation, outlier removal, or density-dependent quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def finite_array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Invalid shape or nonfinite geometry")
    return result


def axis_vector(value):
    axis = finite_array(value, (3,))
    if not np.isclose(np.linalg.norm(axis), 1., rtol=0, atol=1e-10):
        raise ValueError("Reference axes must be explicitly normalized")
    return axis


@dataclass
class Geometry:
    points: np.ndarray
    unit_to_meter: float
    representation: str
    material_only: bool
    complete: bool
    cell_size: np.ndarray = field(default_factory=lambda: np.zeros(3))
    truncated: bool = False
    reason: str = ""
    weights: np.ndarray | None = None


def measure(geometry, axis, origin=(0., 0., 0.), quantiles=None):
    """Unknown inputs stay unknown; extrema include full occupied-cell support.

    origin is expressed in the geometry's coordinate units. Projection spans
    are translation invariant. Quantiles are a sensitivity diagnostic only.
    """
    out = {"status": "unknown", "value": None, "unit": "mm", "reason": ""}
    try:
        u, o = axis_vector(axis), finite_array(origin, (3,))
        p = finite_array(geometry.points)
        if p.ndim != 2 or p.shape[1] != 3 or not len(p):
            raise ValueError("empty_or_malformed_points")
        scale = float(geometry.unit_to_meter)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid_unit_conversion")
        if not geometry.material_only:
            raise ValueError("tool_or_ground_contamination")
        if not geometry.complete:
            raise ValueError(geometry.reason or "incomplete_or_occluded_material")
        if geometry.truncated:
            raise ValueError("occupied_query_boundary_or_outside_support")
        if geometry.representation not in ("points", "mesh_vertices", "voxel_union"):
            raise ValueError("unsupported_representation")
        step = finite_array(geometry.cell_size, (3,))
        if (step < 0).any() or (geometry.representation == "voxel_union" and (step <= 0).any()):
            raise ValueError("invalid_cell_size")
        if geometry.representation != "voxel_union" and step.any():
            raise ValueError("cell_size_requires_voxel_union")
        projected = (p - o) @ u
        if quantiles is None:
            low, high = projected.min(), projected.max()
        else:
            q = finite_array(quantiles, (2,))
            if not 0 <= q[0] < q[1] <= 1:
                raise ValueError("invalid_quantiles")
            if geometry.representation == "voxel_union":
                weights = np.ones(len(p))
            elif geometry.weights is not None:
                weights = finite_array(geometry.weights, (len(p),))
            else:
                raise ValueError("quantiles_require_explicit_sampling_weights")
            if (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("invalid_sampling_weights")
            positive = weights > 0
            order = np.argsort(projected[positive], kind="stable")
            values, weights = projected[positive][order], weights[positive][order]
            cdf = np.cumsum(weights) / weights.sum()
            indices = np.minimum(np.searchsorted(cdf, q, side="left"), len(values)-1)
            low, high = values[indices]
        cell_support = np.abs(u) @ step
        out.update(status="measured", value=float((high-low+cell_support)*1000*scale),
                   cell_projection_mm=float(cell_support*1000*scale),
                   operator="extrema" if quantiles is None else "weighted_inverse_cdf", reason="")
    except (TypeError, ValueError, OverflowError) as exc:
        out["reason"] = str(exc)
    return out


def from_mesh(vertices, faces, unit_to_meter, *, material_only, complete):
    """Ignore padding/unreferenced vertices, never discard contact material."""
    try:
        p = finite_array(vertices)
        f = np.asarray(faces)
        if p.ndim != 2 or p.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("invalid_mesh_shape")
        if not np.issubdtype(f.dtype, np.integer) or not f.size or f.min() < 0 or f.max() >= len(p):
            raise ValueError("invalid_mesh_faces")
        tri = p[f]
        nonpadding = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) > 0
        points = p[np.unique(f[nonpadding])]
        return Geometry(points, unit_to_meter, "mesh_vertices", material_only, complete)
    except (TypeError, ValueError) as exc:
        return Geometry(np.empty((0,3)), unit_to_meter, "mesh_vertices", material_only, False, reason=str(exc))


def grid_coordinates(bounds, spacing, max_cells=4_000_000):
    bounds = finite_array(bounds, (3,2))
    step = finite_array(spacing, (3,))
    if (step <= 0).any() or (bounds[:,1] <= bounds[:,0]).any():
        raise ValueError("Invalid grid bounds/spacing")
    counts = np.ceil((bounds[:,1]-bounds[:,0])/step).astype(int)+1
    if np.prod(counts, dtype=np.int64) > max_cells:
        raise ValueError("Grid exceeds declared memory/query budget")
    # Last centre is on or beyond the requested upper bound; bounds saved exactly.
    axes = [bounds[i,0] + np.arange(counts[i])*step[i] for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1,3), counts


def from_occupancy(query, classes, counts, spacing, unit_to_meter, *, complete=True,
                   ignored_mask=None, outside_support=False):
    """Class 0 is background, 1..4 material; tool/ground must not be encoded."""
    try:
        p = finite_array(query)
        c, n = np.asarray(classes), np.asarray(counts)
        step = finite_array(spacing, (3,))
        if n.shape != (3,) or not np.issubdtype(n.dtype, np.integer) or (n < 2).any():
            raise ValueError("invalid_grid_counts")
        if p.shape != (int(np.prod(n)),3) or c.shape != (len(p),):
            raise ValueError("occupancy_grid_shape_mismatch")
        if not np.isfinite(c).all() or ((c < 0)|(c > 4)|(c != np.rint(c))).any():
            raise ValueError("invalid_material_classes")
        if (step <= 0).any():
            raise ValueError("invalid_cell_size")
        origin = p[0]
        expected = np.stack(np.meshgrid(*[origin[i]+np.arange(n[i])*step[i] for i in range(3)],
                                       indexing="ij"), axis=-1).reshape(-1,3)
        if not np.allclose(p, expected, rtol=0, atol=max(step)*1e-5):
            raise ValueError("query_is_not_declared_regular_grid")
        if ignored_mask is not None and np.asarray(ignored_mask).any():
            raise ValueError("masked_contact_or_unobserved_cells")
        occupied = (c > 0).reshape(tuple(n))
        boundary = any(np.take(occupied, edge, axis=i).any() for i in range(3) for edge in (0,-1))
        return Geometry(p[c > 0], unit_to_meter, "voxel_union", True, complete, step,
                        bool(boundary or outside_support))
    except (TypeError, ValueError) as exc:
        return Geometry(np.empty((0,3)), unit_to_meter, "voxel_union", True, False, reason=str(exc))


def particle_occupancy(query, particles, radius):
    """Unmasked raw simulator material: union of equal-radius balls.

    scipy is imported only by this optional execution-data adapter.
    Radius is a representation choice, not a task tolerance or particle stress.
    """
    p = finite_array(particles)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid material particles/radius")
    q = finite_array(query)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("Invalid occupancy query")
    try:
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(p).query(q)
    except ImportError:
        if len(q)*len(p)>2_000_000:
            raise RuntimeError("Install scipy for large particle occupancy queries")
        distance=np.sqrt(((q[:,None,:]-p[None,:,:])**2).sum(axis=-1).min(axis=1))
    return (distance <= radius).astype(np.uint8)
