"""Explicit, grid-independent crack graphs and their visible surface loss fields."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import Voronoi


def validate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the physical graph arrays in a v2 crack event."""
    if not isinstance(graph, dict) or set(graph) != {"nodes", "edges", "closed_cells_xy_m"}:
        raise ValueError("graph requires nodes, edges and closed_cells_xy_m")
    nodes, edges, cells = graph["nodes"], graph["edges"], graph["closed_cells_xy_m"]
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(cells, list):
        raise ValueError("graph nodes/edges/closed_cells_xy_m must be arrays")
    if len(nodes) < 2 or not edges:
        raise ValueError("crack graph requires at least two nodes and one edge")
    node_map: dict[str, np.ndarray] = {}
    node_coordinates: set[tuple[float, float]] = set()
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {"id", "xy_m", "boundary"}:
            raise ValueError("each graph node requires id, xy_m and boundary")
        ident = node["id"]
        xy = np.asarray(node["xy_m"], dtype=float)
        if (
            not isinstance(ident, str)
            or not ident
            or ident in node_map
            or xy.shape != (2,)
            or not np.isfinite(xy).all()
        ):
            raise ValueError(f"invalid or duplicate graph node {ident!r}")
        if not isinstance(node["boundary"], bool):
            raise ValueError(f"node {ident}: boundary must be boolean")
        coordinate = (float(xy[0]), float(xy[1]))
        if coordinate in node_coordinates:
            raise ValueError(
                f"node {ident}: duplicate coordinate must reuse the existing shared node id"
            )
        node_coordinates.add(coordinate)
        node_map[ident] = xy
    edge_ids: set[str] = set()
    referenced_nodes: set[str] = set()
    normalized_edges = []
    for edge in edges:
        required = {
            "id",
            "start",
            "end",
            "points_m",
            "half_width_left_m",
            "half_width_right_m",
            "visible_depth_m",
        }
        if not isinstance(edge, dict) or set(edge) != required:
            raise ValueError(
                "each graph edge requires id/start/end and explicit physical profile arrays"
            )
        ident, start, end = edge["id"], edge["start"], edge["end"]
        if not isinstance(ident, str) or not ident or ident in edge_ids:
            raise ValueError(f"invalid or duplicate graph edge {ident!r}")
        edge_ids.add(ident)
        if start not in node_map or end not in node_map or start == end:
            raise ValueError(f"edge {ident}: start/end must reference distinct existing nodes")
        referenced_nodes.update((start, end))
        points = np.asarray(edge["points_m"], dtype=float)
        left = np.asarray(edge["half_width_left_m"], dtype=float)
        right = np.asarray(edge["half_width_right_m"], dtype=float)
        depth = np.asarray(edge["visible_depth_m"], dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"edge {ident}: points_m must be finite Nx2 with N>=2")
        if any(v.shape != (len(points),) or not np.isfinite(v).all() for v in (left, right, depth)):
            raise ValueError(f"edge {ident}: width/depth arrays must be finite and match points_m")
        if np.any(left <= 0) or np.any(right <= 0) or np.any(depth < 0):
            raise ValueError(
                f"edge {ident}: half-widths must be positive and visible depth nonnegative"
            )
        if not np.array_equal(points[0], node_map[start]) or not np.array_equal(
            points[-1], node_map[end]
        ):
            raise ValueError(
                f"edge {ident}: polyline endpoints must exactly equal referenced node coordinates"
            )
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
            raise ValueError(f"edge {ident}: consecutive polyline points must differ")
        normalized_edges.append((ident, start, end, points, left, right, depth))
    if referenced_nodes != set(node_map):
        raise ValueError("every graph node must be an endpoint of at least one edge")
    for cell in cells:
        poly = np.asarray(cell, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            raise ValueError(
                "closed cell must be finite polygon coordinates with at least three vertices"
            )
        area = 0.5 * abs(
            float(
                np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1))
            )
        )
        if area <= 1e-15:
            raise ValueError("closed cell polygon must have positive area")
    _reject_unshared_crossings(normalized_edges)
    return {"nodes": nodes, "edges": normalized_edges, "closed_cells_xy_m": cells}


def _reject_unshared_crossings(edges: list[tuple]) -> None:
    """Reject proper segment crossings unless both lines meet at a shared graph node."""
    eps = 1e-12

    def cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    segments = []
    for ei, edge in enumerate(edges):
        for si, (p0, p1) in enumerate(zip(edge[3][:-1], edge[3][1:], strict=True)):
            segments.append((ei, si, p0, p1))
    for ix, (ai, asi, p0, p1) in enumerate(segments):
        r = p1 - p0
        for bi, bsi, q0, q1 in segments[ix + 1 :]:
            s = q1 - q0
            den = cross2(r, s)
            qp = q0 - p0
            if abs(den) <= eps:
                if abs(cross2(qp, r)) > eps:
                    continue
                axis = int(np.argmax(np.abs(r)))
                p_lo, p_hi = sorted((p0[axis], p1[axis]))
                q_lo, q_hi = sorted((q0[axis], q1[axis]))
                overlap = min(p_hi, q_hi) - max(p_lo, q_lo)
                if overlap > eps:
                    raise ValueError(
                        f"edges {edges[ai][0]} and {edges[bi][0]} have collinear overlapping segments"
                    )
                if overlap < -eps:
                    continue
                shared_axis = max(p_lo, q_lo)
                xy = next(
                    point.copy()
                    for point in (p0, p1, q0, q1)
                    if abs(point[axis] - shared_axis) <= eps
                )
            else:
                t = cross2(qp, s) / den
                u = cross2(qp, r) / den
                if not (-eps <= t <= 1 + eps and -eps <= u <= 1 + eps):
                    continue
                xy = p0 + t * r
            a, b = edges[ai], edges[bi]
            shared = {a[1], a[2]} & {b[1], b[2]}
            is_shared_node = any(
                np.linalg.norm(xy - (a[3][0] if a[1] == nid else a[3][-1])) <= eps
                and np.linalg.norm(xy - (b[3][0] if b[1] == nid else b[3][-1])) <= eps
                for nid in shared
            )
            is_adjacent_vertex = (
                ai == bi
                and abs(asi - bsi) == 1
                and any(np.linalg.norm(xy - point) <= eps for point in (a[3][min(asi, bsi) + 1],))
            )
            if not (is_shared_node or is_adjacent_vertex):
                raise ValueError(
                    f"edges {a[0]} and {b[0]} cross/touch without an explicit shared node"
                )


def render_graph(
    graph: dict[str, Any], x: np.ndarray, y: np.ndarray, profile: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return maximum depth-loss, supporting mask, and winning stable edge indices."""
    if profile not in {"legacy_v1_gaussian", "compact"}:
        raise ValueError("profile must be legacy_v1_gaussian or compact")
    parsed = validate_graph(graph)
    shape = x.shape
    loss = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    owner = np.full(shape, -1, dtype=np.int32)
    for ei, edge in enumerate(parsed["edges"]):
        _, _, _, points, left, right, depths = edge
        edge_mask = np.zeros(shape, dtype=bool)
        edge_loss = np.zeros(shape, dtype=float)
        for k, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
            vx, vy = b - a
            denom = vx * vx + vy * vy
            t = np.clip(((x - a[0]) * vx + (y - a[1]) * vy) / denom, 0.0, 1.0)
            px, py = a[0] + t * vx, a[1] + t * vy
            dx, dy = x - px, y - py
            dist = np.hypot(dx, dy)
            cross = vx * (y - a[1]) - vy * (x - a[0])
            left_k = left[k] + t * (left[k + 1] - left[k])
            right_k = right[k] + t * (right[k + 1] - right[k])
            half_width = np.where(cross >= 0, left_k, right_k)
            u = dist / half_width
            d = depths[k] + t * (depths[k + 1] - depths[k])
            segment_mask = u <= 1.0
            if profile == "legacy_v1_gaussian":
                segment_loss = np.where(segment_mask, d * np.exp(-(u**2)), 0.0)
            else:
                segment_loss = np.where(
                    segment_mask, d * np.maximum(0.0, 1.0 - u**2) ** 2, 0.0
                )
            # Every segment contributes support, even if a narrower segment is nearer.
            edge_mask |= segment_mask
            np.maximum(edge_loss, segment_loss, out=edge_loss)
        wins = edge_loss > loss  # stable earlier edge wins ties
        loss[wins] = edge_loss[wins]
        owner[wins] = ei
        mask |= edge_mask
    return loss, mask, owner


def build_bezier_graph(
    control_points_xy_m: list[list[float]],
    knot_spacing_m: float,
    half_width_left_m: list[float],
    half_width_right_m: list[float],
    visible_depth_m: list[float],
    *,
    seed: int,
    perturbation_rms_m: float = 0.0,
) -> dict[str, Any]:
    """Sample one cubic Bezier centerline into a grid-independent explicit graph.

    Width/depth inputs are explicit endpoint knots and are linearly interpolated
    along the sampled curve. Optional jitter is a seeded engineering hypothesis.
    """
    control = np.asarray(control_points_xy_m, dtype=float)
    if control.shape != (4, 2) or not np.isfinite(control).all() or knot_spacing_m <= 0:
        raise ValueError("Bezier requires four finite XY controls and positive knot_spacing_m")
    if seed < 0 or perturbation_rms_m < 0 or not np.isfinite(perturbation_rms_m):
        raise ValueError("seed and perturbation_rms_m are invalid")
    width_l = np.asarray(half_width_left_m, dtype=float)
    width_r = np.asarray(half_width_right_m, dtype=float)
    depth = np.asarray(visible_depth_m, dtype=float)
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in (width_l, width_r, depth)):
        raise ValueError(
            "Bezier half-width/depth arrays must each contain two finite endpoint values"
        )
    if np.array_equal(control[0], control[-1]):
        raise ValueError("Bezier line endpoints must differ")
    if np.any(width_l <= 0) or np.any(width_r <= 0) or np.any(depth < 0):
        raise ValueError("Bezier half-widths must be positive and depth nonnegative")
    estimate = float(sum(np.linalg.norm(np.diff(control, axis=0), axis=1)))
    if estimate <= 0:
        raise ValueError("Bezier centerline must have positive length")
    t_dense = np.linspace(0.0, 1.0, max(256, int(np.ceil(estimate / knot_spacing_m * 16)) + 1))
    omt = 1.0 - t_dense
    dense = (
        omt[:, None] ** 3 * control[0]
        + 3 * omt[:, None] ** 2 * t_dense[:, None] * control[1]
        + 3 * omt[:, None] * t_dense[:, None] ** 2 * control[2]
        + t_dense[:, None] ** 3 * control[3]
    )
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    targets = np.r_[np.arange(0.0, arc[-1], knot_spacing_m), arc[-1]]
    points = np.column_stack([np.interp(targets, arc, dense[:, axis]) for axis in range(2)])
    t = np.interp(targets, arc, t_dense)
    normalized_arc = targets / arc[-1]
    if perturbation_rms_m:
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0.0, perturbation_rms_m, size=points.shape)
        taper = np.sin(np.pi * t)[:, None]
        points[1:-1] += jitter[1:-1] * taper[1:-1]
    points[0], points[-1] = control[0], control[-1]
    return {
        "nodes": [
            {"id": "n0", "xy_m": points[0].tolist(), "boundary": False},
            {"id": "n1", "xy_m": points[-1].tolist(), "boundary": False},
        ],
        "edges": [
            {
                "id": "e0",
                "start": "n0",
                "end": "n1",
                "points_m": points.tolist(),
                "half_width_left_m": np.interp(normalized_arc, [0, 1], width_l).tolist(),
                "half_width_right_m": np.interp(normalized_arc, [0, 1], width_r).tolist(),
                "visible_depth_m": np.interp(normalized_arc, [0, 1], depth).tolist(),
            }
        ],
        "closed_cells_xy_m": [],
    }


def build_alligator_graph(
    region_bounds_m: list[float],
    block_size_m: float,
    inhibition_ratio: float,
    half_width_m: float,
    visible_depth_m: float,
    *,
    seed: int,
) -> dict[str, Any]:
    """Build bounded Voronoi-cell cracks from explicit region/spacing assumptions."""
    from .primitives import _generate_alligator_seeds

    if len(region_bounds_m) != 4:
        raise ValueError("region_bounds_m must be [xmin,xmax,ymin,ymax]")
    xmin, xmax, ymin, ymax = map(float, region_bounds_m)
    if (
        not np.isfinite(
            [xmin, xmax, ymin, ymax, block_size_m, inhibition_ratio, half_width_m, visible_depth_m]
        ).all()
        or min(xmax - xmin, ymax - ymin, block_size_m, half_width_m) <= 0
        or inhibition_ratio < 0
        or visible_depth_m < 0
    ):
        raise ValueError("invalid alligator geometry parameters")
    area = (xmax - xmin) * (ymax - ymin)
    expected = max(4, int(area / (block_size_m**2)))
    rng = np.random.default_rng(seed)
    seeds = _generate_alligator_seeds(
        xmin, xmax, ymin, ymax, expected, block_size_m * inhibition_ratio, rng
    )
    # Qhull ridge geometry supplies a shared-node network; cells are separately
    # clipped from half-planes so no unbounded or hidden coordinates are needed.
    pad = max(xmax - xmin, ymax - ymin) * 2
    extras = np.array(
        [
            [xmin - pad, ymin - pad],
            [xmin - pad, ymax + pad],
            [xmax + pad, ymin - pad],
            [xmax + pad, ymax + pad],
        ]
    )
    vor = Voronoi(np.vstack((seeds, extras)))
    roi = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
    polygons = []
    for i, si in enumerate(seeds):
        poly = roi.copy()
        for j, sj in enumerate(seeds):
            if i == j or len(poly) == 0:
                continue
            normal = 2 * (sj - si)
            limit = float(np.dot(sj, sj) - np.dot(si, si))
            out = []
            for p, q in zip(poly, np.roll(poly, -1, axis=0), strict=True):
                fp = float(np.dot(normal, p) - limit)
                fq = float(np.dot(normal, q) - limit)
                pin, qin = fp <= 1e-12, fq <= 1e-12
                if pin:
                    out.append(p)
                if pin != qin:
                    t = fp / (fp - fq)
                    out.append(p + t * (q - p))
            poly = np.asarray(out, dtype=float)
        if len(poly) >= 3:
            polygons.append(poly)
    nodes: list[dict[str, Any]] = []
    node_ids: dict[tuple[float, float], str] = {}
    edges = []
    cells = []
    edge_ids = {}

    def node_id(point: np.ndarray) -> str:
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key not in node_ids:
            nid = f"n{len(nodes)}"
            node_ids[key] = nid
            boundary = (
                abs(key[0] - xmin) < 1e-9
                or abs(key[0] - xmax) < 1e-9
                or abs(key[1] - ymin) < 1e-9
                or abs(key[1] - ymax) < 1e-9
            )
            nodes.append(
                {"id": nid, "xy_m": [float(point[0]), float(point[1])], "boundary": boundary}
            )
        return node_ids[key]

    for poly in polygons:
        if len(poly) < 3:
            continue
        cells.append(poly.tolist())
    center = vor.points.mean(axis=0)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        if p1 >= len(seeds) and p2 >= len(seeds):
            continue
        if v1 >= 0 and v2 >= 0:
            a, b = vor.vertices[v1], vor.vertices[v2]
        else:
            finite = v1 if v1 >= 0 else v2
            tangent = vor.points[p2] - vor.points[p1]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            a = vor.vertices[finite]
            b = a + direction * (pad * 4)
        # Liang-Barsky clipping of the ridge segment to the explicit ROI.
        direction = b - a
        t0, t1 = 0.0, 1.0
        for p, q in (
            (-direction[0], a[0] - xmin),
            (direction[0], xmax - a[0]),
            (-direction[1], a[1] - ymin),
            (direction[1], ymax - a[1]),
        ):
            if abs(p) < 1e-15:
                if q < 0:
                    t0, t1 = 1, 0
                    break
            else:
                r = q / p
                if p < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if t1 <= t0:
            continue
        a, b = a + t0 * direction, a + t1 * direction
        na, nb = node_id(a), node_id(b)
        key = tuple(sorted((na, nb)))
        if na == nb or key in edge_ids:
            continue
        eid = f"e{len(edges)}"
        edge_ids[key] = eid
        a_xy = np.asarray(nodes[int(na[1:])]["xy_m"])
        b_xy = np.asarray(nodes[int(nb[1:])]["xy_m"])
        edges.append(
            {
                "id": eid,
                "start": na,
                "end": nb,
                "points_m": [a_xy.tolist(), b_xy.tolist()],
                "half_width_left_m": [half_width_m] * 2,
                "half_width_right_m": [half_width_m] * 2,
                "visible_depth_m": [visible_depth_m] * 2,
            }
        )
    return {"nodes": nodes, "edges": edges, "closed_cells_xy_m": cells}


def build_block_graph(
    x_nodes_m: list[float],
    y_nodes_m: list[float],
    *,
    direction: str,
    perturbation_rms_m: float,
    half_width_m: float,
    visible_depth_m: float,
    seed: int,
) -> dict[str, Any]:
    """Create a finite near-rectangular network from explicit node axes."""
    xs, ys = np.asarray(x_nodes_m, dtype=float), np.asarray(y_nodes_m, dtype=float)
    if (
        xs.ndim != 1
        or ys.ndim != 1
        or len(xs) < 2
        or len(ys) < 2
        or not np.isfinite(xs).all()
        or not np.isfinite(ys).all()
        or np.any(np.diff(xs) <= 0)
        or np.any(np.diff(ys) <= 0)
    ):
        raise ValueError(
            "block graph axes must be finite strictly increasing arrays with >=2 values"
        )
    if (
        direction not in {"x", "y"}
        or perturbation_rms_m < 0
        or half_width_m <= 0
        or visible_depth_m < 0
        or seed < 0
    ):
        raise ValueError("invalid block graph direction or physical parameters")
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack((xx, yy), axis=-1)
    if perturbation_rms_m:
        rng = np.random.default_rng(seed)
        jitter = np.zeros_like(coords)
        jitter[..., 1 if direction == "x" else 0] = rng.normal(
            0, perturbation_rms_m, size=coords.shape[:2]
        )
        jitter[[0, -1], :, :] = 0
        jitter[:, [0, -1], :] = 0
        coords += jitter
    nodes = []
    for i in range(len(xs)):
        for j in range(len(ys)):
            nodes.append(
                {
                    "id": f"n{i}_{j}",
                    "xy_m": coords[i, j].tolist(),
                    "boundary": i in {0, len(xs) - 1} or j in {0, len(ys) - 1},
                }
            )
    edges = []
    for i in range(len(xs)):
        for j in range(len(ys)):
            for di, dj in ((1, 0), (0, 1)):
                if i + di >= len(xs) or j + dj >= len(ys):
                    continue
                a, b = f"n{i}_{j}", f"n{i + di}_{j + dj}"
                p0, p1 = coords[i, j], coords[i + di, j + dj]
                edges.append(
                    {
                        "id": f"e{len(edges)}",
                        "start": a,
                        "end": b,
                        "points_m": [p0.tolist(), p1.tolist()],
                        "half_width_left_m": [half_width_m] * 2,
                        "half_width_right_m": [half_width_m] * 2,
                        "visible_depth_m": [visible_depth_m] * 2,
                    }
                )
    cells = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cells.append(
                [
                    coords[i, j].tolist(),
                    coords[i + 1, j].tolist(),
                    coords[i + 1, j + 1].tolist(),
                    coords[i, j + 1].tolist(),
                ]
            )
    return {"nodes": nodes, "edges": edges, "closed_cells_xy_m": cells}
