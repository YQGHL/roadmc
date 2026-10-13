"""Final geometric normal estimation for synthetic surface point clouds.

Road/cross-section grouping is the caller's responsibility.  The regular-grid
path preserves input row alignment (including NaN dropout rows); the scattered
path uses local PCA with bounded per-batch covariance memory.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _unit_normals(dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    normals = np.stack((-dzdx, -dzdy, np.ones_like(dzdx)), axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals


def _grid_normals(
    points: np.ndarray, shape: tuple[int, int], spacing: tuple[float, float], order: str
) -> np.ndarray:
    nx, ny = shape
    if nx < 1 or ny < 1 or nx * ny != len(points):
        raise ValueError("grid_shape must be positive and its product must equal len(points)")
    if order not in ("xy", "yx"):
        raise ValueError("grid_order must be 'xy' or 'yx'")
    sx, sy = spacing
    grid = points.reshape((nx, ny, 3) if order == "xy" else (ny, nx, 3))
    if order == "yx":
        grid = grid.transpose(1, 0, 2)
    z = grid[..., 2]
    valid = np.isfinite(grid).all(axis=-1)
    dx = np.zeros((nx, ny), dtype=np.float64)
    dy = np.zeros_like(dx)

    # Estimate from immediate finite grid neighbors only. A missing sample
    # creates a true hole; no derivative reaches across it.
    for axis, derivative, step in ((0, dx, sx), (1, dy, sy)):
        before_valid = np.zeros_like(valid)
        after_valid = np.zeros_like(valid)
        before_z = np.zeros_like(z)
        after_z = np.zeros_like(z)
        before_slice = [slice(None), slice(None)]
        center_slice = [slice(None), slice(None)]
        before_slice[axis] = slice(0, -1)
        center_slice[axis] = slice(1, None)
        before_valid[tuple(center_slice)] = valid[tuple(before_slice)]
        before_z[tuple(center_slice)] = z[tuple(before_slice)]
        after_valid[tuple(before_slice)] = valid[tuple(center_slice)]
        after_z[tuple(before_slice)] = z[tuple(center_slice)]
        has_before = valid & before_valid
        has_after = valid & after_valid
        central = has_before & has_after
        forward = ~has_before & has_after
        backward = has_before & ~has_after
        derivative[central] = (after_z[central] - before_z[central]) / (2.0 * step)
        derivative[forward] = (after_z[forward] - z[forward]) / step
        derivative[backward] = (z[backward] - before_z[backward]) / step

    result = _unit_normals(dx, dy).reshape(-1, 3) if order == "xy" else _unit_normals(dx, dy).transpose(1, 0, 2).reshape(-1, 3)
    result[~valid.reshape(-1) if order == "xy" else ~valid.transpose(1, 0).reshape(-1)] = (0.0, 0.0, 1.0)
    return result


def _pca_normals(points: np.ndarray, k_neighbors: int) -> np.ndarray:
    if not isinstance(k_neighbors, (int, np.integer)) or k_neighbors < 3:
        raise ValueError("k_neighbors must be an integer of at least 3")
    finite = np.isfinite(points).all(axis=1)
    result = np.tile((0.0, 0.0, 1.0), (len(points), 1)).astype(np.float64)
    indices = np.flatnonzero(finite)
    cloud = points[indices]
    if len(cloud) < 4:
        return result
    k = min(int(k_neighbors), len(cloud) - 1)
    tree = cKDTree(cloud)
    # Bound temporary neighbor/covariance/eigensystem arrays independently of
    # total scan size. cKDTree query itself is also issued per batch.
    batch_size = max(1, min(8192, 131072 // (k + 1)))
    for start in range(0, len(cloud), batch_size):
        stop = min(start + batch_size, len(cloud))
        _, neighbor_idx = tree.query(cloud[start:stop], k=k + 1)
        # Equal-distance duplicates need not put the query itself first.
        # Remove by identity, then keep k neighbors in distance order.
        is_self = neighbor_idx == np.arange(start, stop)[:, None]
        columns = np.argsort(is_self, axis=1, kind="stable")[:, :k]
        neighbor_idx = np.take_along_axis(neighbor_idx, columns, axis=1)
        neighbors = cloud[neighbor_idx]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("bki,bkj->bij", centered, centered) / max(k - 1, 1)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]
            normals[normals[:, 2] < 0.0] *= -1.0
            lengths = np.linalg.norm(normals, axis=1)
            good = np.isfinite(normals).all(axis=1) & (lengths > 1e-12)
            # Numerical conditioning criteria, not physical surface priors:
            # require two supported tangent directions and a distinct normal.
            scale = eigenvalues[:, 2]
            good &= (scale > 0.0) & (eigenvalues[:, 1] > 1e-12 * scale)
            good &= (eigenvalues[:, 1] - eigenvalues[:, 0]) > 1e-8 * scale
            normals[good] /= lengths[good, None]
            normals[~good] = (0.0, 0.0, 1.0)
            result[indices[start:stop]] = normals
        except np.linalg.LinAlgError:
            # Preserve the finite, upward fallback for a failed batch.
            continue
    return result


def estimate_surface_normals(
    points: np.ndarray,
    *,
    grid_shape: tuple[int, int] | None = None,
    grid_spacing_m: float | tuple[float, float] | None = None,
    grid_order: str = "xy",
    k_neighbors: int = 16,
) -> np.ndarray:
    """Estimate final surface normals, returning one normal per input row.

    For a regular height grid, ``grid_shape=(nx, ny)`` identifies x/y sample
    counts and spacing is in meters. ``grid_order='xy'`` means x-major then
    y-major C-order; ``'yx'`` means y-major then x-major. NaN rows represent
    dropout and remain aligned in output; derivatives use only adjacent valid
    cells, with central differences when both sides exist and one-sided
    differences otherwise. Isolated cells receive zero slope.

    Without grid metadata, local PCA estimates normals for an irregular cloud.
    Degenerate or isotropic neighborhoods receive the upward fallback.
    Both modes orient normals toward positive z; non-finite input rows return
    the neutral ``[0, 0, 1]`` fallback for the caller to filter.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {cloud.shape}")
    if grid_shape is None:
        return _pca_normals(cloud, k_neighbors)
    if grid_spacing_m is None:
        raise ValueError("grid_spacing_m is required when grid_shape is provided")
    if np.isscalar(grid_spacing_m):
        spacing = (float(grid_spacing_m), float(grid_spacing_m))
    else:
        spacing = tuple(float(value) for value in grid_spacing_m)
        if len(spacing) != 2:
            raise ValueError("grid_spacing_m must be a scalar or (dx, dy)")
    if any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("grid_spacing_m values must be finite and positive")
    return _grid_normals(cloud, tuple(map(int, grid_shape)), spacing, grid_order)
