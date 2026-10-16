"""Resolution-aware geometry diagnostics for explicit crack graphs.

All inputs use metres and x-major regular grids.  Vector-graph measurements
are kept separate from measurements extracted from the height residual; the
former describe the supplied model, while the latter describe only visible
sampled geometry.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from scipy import ndimage


def _grid_inputs(
    x_m: np.ndarray,
    y_m: np.ndarray,
    reference_z_m: np.ndarray,
    final_z_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    reference = np.asarray(reference_z_m, dtype=np.float64)
    final = np.asarray(final_z_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("x_m and y_m must be 1D axes with at least two samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("grid coordinates must be finite")
    dxs, dys = np.diff(x), np.diff(y)
    if np.any(dxs <= 0.0) or np.any(dys <= 0.0):
        raise ValueError("grid coordinates must be strictly increasing")
    dx, dy = float(dxs[0]), float(dys[0])
    if not np.allclose(dxs, dx, rtol=1e-9, atol=1e-12):
        raise ValueError("x_m must be regularly spaced")
    if not np.allclose(dys, dy, rtol=1e-9, atol=1e-12):
        raise ValueError("y_m must be regularly spaced")
    shape = (len(x), len(y))
    if reference.shape != shape or final.shape != shape:
        raise ValueError(f"height arrays must have x-major shape {shape}")
    if not np.isfinite(reference).all() or not np.isfinite(final).all():
        raise ValueError("reference and final heights must be finite")
    return x, y, reference, final, dx, dy


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _find_root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _polygon_metrics(vertices: Any) -> dict[str, Any]:
    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("cell polygons must be finite Nx2 arrays with at least three vertices")
    rolled = np.roll(polygon, -1, axis=0)
    area = 0.5 * abs(float(np.sum(polygon[:, 0] * rolled[:, 1] - rolled[:, 0] * polygon[:, 1])))
    perimeter = float(np.linalg.norm(rolled - polygon, axis=1).sum())
    edge_vectors = rolled - polygon
    edge_angles = np.arctan2(edge_vectors[:, 1], edge_vectors[:, 0])
    candidates = []
    for angle in edge_angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray(((c, s), (-s, c)))
        projected = polygon @ rotation.T
        spans = np.ptp(projected, axis=0)
        candidates.append((float(np.prod(spans)), float(angle), spans))
    bbox_area, major_angle, spans = min(candidates, key=lambda item: item[0])
    axis_ratio = float(max(spans) / max(min(spans), 1e-30))
    return {
        "area_m2": area,
        "perimeter_m": perimeter,
        "rectangularity": min(area / bbox_area, 1.0) if bbox_area else None,
        "orientation_deg": float(np.degrees(major_angle)) if axis_ratio > 1.05 else None,
        "orientation_status": "available" if axis_ratio > 1.05 else "ambiguous_near_square",
        "major_minor_axis_ratio": axis_ratio,
    }


def _background_cell_metrics(mask: np.ndarray, dx: float, dy: float) -> dict[str, Any]:
    """Measure only fully enclosed background components in the sampled ROI."""
    # 8-connected crack pixels pair with 4-connected background pixels to
    # avoid diagonal corner contacts simultaneously opening and closing cells.
    labels, count = ndimage.label(~mask, structure=ndimage.generate_binary_structure(2, 1))
    if not count:
        return {"closed_cell_count": 0, "closed_cells": [], "roi_open_background_components": 0}
    boundary_ids = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist())
    closed = []
    open_count = 0
    for component_id in range(1, count + 1):
        region = labels == component_id
        coords = np.argwhere(region)
        if component_id in boundary_ids:
            open_count += 1
            continue
        xy = np.column_stack((coords[:, 0] * dx, coords[:, 1] * dy))
        centered = xy - xy.mean(axis=0)
        covariance = centered.T @ centered / max(len(xy), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, -1]
        projected = centered @ eigenvectors
        spans = np.ptp(projected, axis=0) + max(dx, dy)
        axis_ratio = float(max(spans) / max(min(spans), 1e-30))
        box_area = float(np.prod(spans))
        closed.append({
            "area_m2": float(len(coords) * dx * dy),
            "rectangularity": float(min(len(coords) * dx * dy / box_area, 1.0)) if box_area else None,
            "orientation_deg": float(np.degrees(np.arctan2(major[1], major[0]))) if axis_ratio > 1.05 else None,
            "orientation_status": "available" if axis_ratio > 1.05 else "ambiguous_near_square",
            "major_minor_axis_ratio": axis_ratio,
            "measurement_caveat": "grid-pixel area and PCA orientation proxy; resolution-quantized, unlike nominal vector-polygon metrics",
        })
    return {
        "closed_cell_count": len(closed),
        "closed_cells": closed,
        "roi_open_background_components": open_count,
        "cell_metric_caveat": "only fully enclosed threshold-mask holes are measured; ROI-open regions are censored, not counted as complete cells",
    }


def _nominal_graph_metrics(
    graph: dict[str, Any], roi: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Measure normalized graph primitives; raw JSON parsing belongs in an adapter."""
    xmin, xmax, ymin, ymax = roi
    lines = []
    network_summary = []
    for event in graph.get("line_events", []):
        if not isinstance(event, dict):
            raise ValueError("each normalized line graph must be an object")
        event_id = str(event.get("event_id", ""))
        line = np.asarray(event.get("centerline_xy_m"), dtype=np.float64)
        if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 or not np.isfinite(line).all():
            raise ValueError(f"line {event_id!r} centerline_xy_m must be finite Nx2")
        length = _polyline_length(line)
        chord = float(np.linalg.norm(line[-1] - line[0]))
        width = np.asarray(event.get("support_width_m", []), dtype=np.float64)
        if width.ndim != 1 or (width.size not in (0, len(line))) or not np.isfinite(width).all():
            raise ValueError(f"line {event_id!r} support_width_m must match centerline vertices")
        endpoints = line[[0, -1]]
        censored = int(np.sum(
            (endpoints[:, 0] <= xmin) | (endpoints[:, 0] >= xmax)
            | (endpoints[:, 1] <= ymin) | (endpoints[:, 1] >= ymax)
        ))
        lines.append({
            "event_id": event_id,
            "variant": event.get("variant"),
            "label": event.get("label"),
            "centerline_length_m": length,
            "end_to_end_chord_m": chord,
            "tortuosity": length / chord if chord > 0.0 else None,
            "geometry_support_width_m": {
                "status": "available" if width.size else "unavailable",
                "reason": None if width.size else "graph has no width samples",
                "min": float(width.min()) if width.size else None,
                "median": float(np.median(width)) if width.size else None,
                "max": float(width.max()) if width.size else None,
            },
            "label_width_m": (
                {"status": "nominal_label_metadata", "value_m": event["label_width_m"]}
                if event.get("label_width_m") is not None
                else {"status": "unavailable", "reason": "graph schema has no label bandwidth; taxonomy label does not determine geometry width"}
            ),
            "roi_censored_endpoints": max(censored, int(event.get("roi_censored_endpoint_count", 0))),
            "visible_depth_m": _optional_summary(event.get("visible_depth_m")),
        })
    for event in graph.get("network_events", []):
        if not isinstance(event, dict):
            raise ValueError("each normalized network graph must be an object")
        event_id = str(event.get("event_id", ""))
        nodes = np.asarray(event.get("nodes_xy_m"), dtype=np.float64)
        edges = np.asarray(event.get("edges_node_indices"), dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
            raise ValueError(f"network {event_id!r} nodes_xy_m must be finite Nx2")
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"network {event_id!r} edges_node_indices must be Mx2")
        if edges.size and (edges.min() < 0 or edges.max() >= len(nodes)):
            raise ValueError(f"network {event_id!r} edge references an absent node")
        degrees = np.bincount(edges.ravel(), minlength=len(nodes))
        edge_points = event.get("edge_points_xy_m")
        if edge_points is not None and len(edge_points) != len(edges):
            raise ValueError(f"network {event_id!r} edge_points_xy_m must match edge count")
        lengths = np.asarray([
            _polyline_length(np.asarray(points, dtype=np.float64))
            for points in edge_points
        ]) if edge_points is not None else (
            np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1) if len(edges) else np.empty(0)
        )
        censored = int(np.sum(
            (nodes[:, 0] <= xmin) | (nodes[:, 0] >= xmax)
            | (nodes[:, 1] <= ymin) | (nodes[:, 1] >= ymax)
        ))
        parent = list(range(len(nodes)))
        for left, right in edges:
            root_left, root_right = _find_root(parent, int(left)), _find_root(parent, int(right))
            if root_left != root_right:
                parent[root_right] = root_left
        graph_components = len({_find_root(parent, index) for index in range(len(nodes))}) if len(nodes) else 0
        polygons = event.get("closed_cells_xy_m")
        cell_metrics = [_polygon_metrics(poly) for poly in polygons] if polygons is not None else []
        network_summary.append({
            "event_id": event_id,
            "variant": event.get("variant"),
            "label": event.get("label"),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "endpoint_count": int(np.sum(degrees == 1)),
            "branch_node_count": int(np.sum(degrees >= 3)),
            "intersection_node_count": int(np.sum(degrees >= 3)),
            "connected_component_count": int(graph_components),
            "cycle_rank": int(max(0, len(edges) - len(nodes) + graph_components)),
            "junction_degree_histogram": {
                str(degree): int(np.sum(degrees == degree))
                for degree in sorted(set(degrees.tolist()))
            },
            "total_edge_length_m": float(lengths.sum()),
            "roi_censored_nodes": max(censored, int(event.get("roi_censored_node_count", 0))),
            "partial_edge_count": int(event.get("partial_edge_count", 0)),
            "closed_cells": cell_metrics,
            "closed_cell_count": len(cell_metrics) if polygons is not None else None,
            "closed_cell_status": "available" if polygons is not None else "unavailable",
            "closed_cell_reason": None if polygons is not None else "normalized graph has no closed cell polygons",
        })
    return {"line_events": lines, "network_events": network_summary}


def _optional_summary(values: Any) -> dict[str, Any]:
    if values is None:
        return {"status": "unavailable", "reason": "vector graph has no visible depth samples"}
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return {"status": "unavailable", "reason": "visible depth samples are empty or non-finite"}
    return {
        "status": "nominal_graph_value",
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "sample_count": int(array.size),
    }


def normalize_v2_graph_json(graph_json: str | dict[str, Any]) -> dict[str, Any]:
    """Adapt replay v2 graph JSON to this module's stable diagnostic contract.

    Topology comes only from explicit ``start``/``end`` node identifiers;
    coordinate proximity is never used to invent graph connectivity.
    """
    graph_doc = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    if not isinstance(graph_doc, dict) or not isinstance(graph_doc.get("events"), list):
        raise ValueError("graph_json must be an object with an events list")
    lines: list[dict[str, Any]] = []
    networks: list[dict[str, Any]] = []
    for event in graph_doc["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("graph"), dict):
            raise ValueError("each graph_json event must contain a graph object")
        event_id = str(event.get("instance_id", ""))
        variant = str(event.get("variant", "unknown"))
        graph = event["graph"]
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError(f"event {event_id!r} needs nodes and edges lists")
        node_index: dict[str, int] = {}
        node_xy = []
        node_boundary = []
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or "id" not in node:
                raise ValueError(f"event {event_id!r} contains a node without id")
            identifier = str(node["id"])
            if identifier in node_index:
                raise ValueError(f"event {event_id!r} contains duplicate node id {identifier!r}")
            xy = np.asarray(node.get("xy_m"), dtype=np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f"event {event_id!r} node {identifier!r} needs finite xy_m")
            node_index[identifier] = index
            node_xy.append(xy.tolist())
            node_boundary.append(bool(node.get("boundary", False)))
        if not node_xy:
            raise ValueError(f"event {event_id!r} graph has no nodes")
        edge_rows = []
        edge_points = []
        partial_edge_count = 0
        for edge in edges:
            if not isinstance(edge, dict):
                raise ValueError(f"event {event_id!r} contains a non-object edge")
            start, end = str(edge.get("start")), str(edge.get("end"))
            if start not in node_index or end not in node_index:
                raise ValueError(f"event {event_id!r} edge references a missing endpoint id")
            points = np.asarray(edge.get("points_m"), dtype=np.float64)
            left = np.asarray(edge.get("half_width_left_m"), dtype=np.float64)
            right = np.asarray(edge.get("half_width_right_m"), dtype=np.float64)
            depth = np.asarray(edge.get("visible_depth_m"), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError(f"event {event_id!r} edge {edge.get('id')!r} has invalid points_m")
            if any(values.ndim != 1 or len(values) != len(points) or not np.isfinite(values).all()
                   for values in (left, right, depth)):
                raise ValueError(f"event {event_id!r} edge profile arrays must match points_m")
            edge_rows.append((node_index[start], node_index[end]))
            edge_points.append(points.tolist())
            if node_boundary[node_index[start]] or node_boundary[node_index[end]]:
                partial_edge_count += 1
            if variant in {"longitudinal", "transverse"}:
                lines.append({
                    "event_id": f"{event_id}:{edge.get('id', len(lines))}",
                    "variant": variant,
                    "label": event.get("label", -1),
                    "centerline_xy_m": points.tolist(),
                    "support_width_m": (left + right).tolist(),
                    "visible_depth_m": depth.tolist(),
                    "roi_censored_endpoint_count": int(
                        node_boundary[node_index[start]] + node_boundary[node_index[end]]
                    ),
                })
        if variant in {"alligator", "block"}:
            networks.append({
                "event_id": event_id,
                "variant": variant,
                "label": event.get("label", -1),
                "nodes_xy_m": node_xy,
                "edges_node_indices": edge_rows,
                "edge_points_xy_m": edge_points,
                "partial_edge_count": partial_edge_count,
                "roi_censored_node_count": int(sum(node_boundary)),
            "closed_cells_xy_m": graph.get("closed_cells_xy_m"),
            })
        elif variant not in {"longitudinal", "transverse"}:
            raise ValueError(f"event {event_id!r} has unsupported crack variant {variant!r}")
    return {"line_events": lines, "network_events": networks}


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning for a 2D binary array, with no optional packages."""
    image = np.asarray(mask, dtype=bool).copy()
    if image.ndim != 2:
        raise ValueError("skeleton extraction requires a 2D mask")
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(image, 1)
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(v.astype(np.uint8) for v in neighbors)
            transitions = sum(
                ((~neighbors[i]) & neighbors[(i + 1) % 8]).astype(np.uint8)
                for i in range(8)
            )
            if phase == 0:
                condition = (~p2 | ~p4 | ~p6) & (~p4 | ~p6 | ~p8)
            else:
                condition = (~p2 | ~p4 | ~p8) & (~p2 | ~p6 | ~p8)
            remove = image & (count >= 2) & (count <= 6) & (transitions == 1) & condition
            if remove.any():
                image[remove] = False
                changed = True
    return image


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    """Use the same corner-suppressed edges as length and component profiles.

    Keep a diagonal only when neither intervening orthogonal pixel exists.
    Otherwise that diagonal shortcuts a turn and invents high-degree pixels.
    The remaining orthogonal route preserves connectivity without the shortcut.
    """
    padded = np.pad(skeleton, 1)
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    neighbors = (
        north,
        south,
        west,
        east,
        padded[:-2, :-2] & ~north & ~west,
        padded[:-2, 2:] & ~north & ~east,
        padded[2:, :-2] & ~south & ~west,
        padded[2:, 2:] & ~south & ~east,
    )
    return sum(neighbor.astype(np.uint8) for neighbor in neighbors) * skeleton


def _skeleton_metrics(skeleton: np.ndarray, dx: float, dy: float) -> dict[str, Any]:
    skeleton = np.asarray(skeleton, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    components, n_components = ndimage.label(skeleton, structure=structure)
    degree = _skeleton_degree(skeleton)
    endpoints = skeleton & (degree == 1)
    branches = skeleton & (degree >= 3)
    # A one-pixel dilation groups the several high-degree pixels that a single
    # sampled junction can produce; the source skeleton itself is unchanged.
    branch_groups = ndimage.binary_dilation(branches, structure=structure, iterations=1)
    _, branch_clusters = ndimage.label(branch_groups, structure=structure)
    boundary = np.zeros_like(skeleton)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    censored_endpoints = int(np.sum(endpoints & boundary))
    # Four primary directions plus diagonals. Suppress a diagonal when either
    # intervening orthogonal pixel exists to avoid double-counting a corner.
    edge_length = 0.0
    for di, dj, weight in ((1, 0, dx), (0, 1, dy), (1, 1, float(np.hypot(dx, dy))), (1, -1, float(np.hypot(dx, dy)))):
        a = skeleton[max(0, -di) : skeleton.shape[0] - max(0, di), max(0, -dj) : skeleton.shape[1] - max(0, dj)]
        b = skeleton[max(0, di) : skeleton.shape[0] - max(0, -di), max(0, dj) : skeleton.shape[1] - max(0, -dj)]
        if di and dj:
            cross_a = skeleton[max(0, -di) : skeleton.shape[0] - max(0, di), max(0, dj) : skeleton.shape[1] - max(0, -dj)]
            cross_b = skeleton[max(0, di) : skeleton.shape[0] - max(0, -di), max(0, -dj) : skeleton.shape[1] - max(0, dj)]
            edge_length += float(np.sum(a & b & ~cross_a & ~cross_b)) * weight
        else:
            edge_length += float(np.sum(a & b)) * weight
    return {
        "pixel_component_count": int(n_components),
        "pixel_skeleton_length_m": edge_length,
        "endpoint_count_in_roi": int(endpoints.sum()),
        "roi_censored_endpoints": censored_endpoints,
        "branch_pixel_count": int(branches.sum()),
        "branch_cluster_count": int(branch_clusters),
        "junction_degree_histogram": {
            str(value): int(np.sum(degree[skeleton] == value))
            for value in sorted(set(degree[skeleton].tolist()))
        },
        "topology_graph_rule": "8-neighbor edges with orthogonal corner shortcuts suppressed",
        "branch_grouping_rule": "one-pixel dilation followed by 8-connected clustering",
        "topology_warning": "corner-suppressed pixel topology and one-pixel branch grouping remain resolution- and threshold-dependent",
    }


def _component_profiles(
    skeleton: np.ndarray, distance: np.ndarray, loss: np.ndarray, dx: float, dy: float
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    profiles = []
    height, width = skeleton.shape
    for component_id in range(1, count + 1):
        component = labels == component_id
        coords = [tuple(row) for row in np.argwhere(component)]
        coord_set = set(coords)
        adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for i, j in coords:
            adjacent = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    point = (i + di, j + dj)
                    if point not in coord_set:
                        continue
                    if di and dj and ((i + di, j) in coord_set or (i, j + dj) in coord_set):
                        continue
                    adjacent.append(point)
            adjacency[(i, j)] = adjacent
        endpoints = [point for point, adjacent in adjacency.items() if len(adjacent) == 1]
        branches = [point for point, adjacent in adjacency.items() if len(adjacent) >= 3]
        censored = sum(i in (0, height - 1) or j in (0, width - 1) for i, j in endpoints)
        item: dict[str, Any] = {
            "component_id": component_id,
            "pixel_count": len(coords),
            "endpoint_count": len(endpoints),
            "branch_pixel_count": len(branches),
            "tortuosity": None,
            "roi_censored_endpoints": censored,
            "profile_status": "roi_censored" if censored else "unavailable",
        }
        if len(endpoints) == 2 and not branches and censored == 0:
            ordered = [endpoints[0]]
            previous = None
            while True:
                next_nodes = [node for node in adjacency[ordered[-1]] if node != previous]
                if not next_nodes:
                    break
                following = next_nodes[0]
                if following in ordered:
                    break
                previous = ordered[-1]
                ordered.append(following)
                if following == endpoints[1]:
                    break
            if ordered[-1] == endpoints[1]:
                coordinates = np.asarray(ordered, dtype=np.int64)
                steps = np.diff(coordinates, axis=0) * np.asarray((dx, dy))
                arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(steps, axis=1))))
                chord = float(np.linalg.norm((coordinates[-1] - coordinates[0]) * np.asarray((dx, dy))))
                keep = np.linspace(0, len(ordered) - 1, min(len(ordered), 256), dtype=np.int64)
                item.update({
                    "tortuosity": float(arclength[-1] / chord) if chord > 0.0 else None,
                    "profile_status": "available",
                    "profile_sample_count": int(len(keep)),
                    "profile_s_m": arclength[keep].tolist(),
                    "effective_width_m": (2.0 * distance[coordinates[keep, 0], coordinates[keep, 1]]).tolist(),
                    "visible_loss_depth_m": loss[coordinates[keep, 0], coordinates[keep, 1]].tolist(),
                    "profile_sampling": "uniform index samples along independently extracted skeleton; at most 256 per component",
                })
        profiles.append(item)
    return profiles


def analyze_crack_geometry(
    x_m: np.ndarray,
    y_m: np.ndarray,
    reference_z_m: np.ndarray,
    final_z_m: np.ndarray,
    normalized_graph: dict[str, Any],
    *,
    loss_thresholds_m: tuple[float, ...],
    roi_bounds_m: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Compare nominal vector-graph quantities with independent surface extraction.

    ``loss_thresholds_m`` are diagnostic thresholds only, not calibrated crack
    definitions.  Surface masks and skeletons are derived solely from the
    reference-to-final height residual; graph, labels, and event masks are not
    used to create measured geometry.
    """
    x, y, reference, final, dx, dy = _grid_inputs(x_m, y_m, reference_z_m, final_z_m)
    roi = roi_bounds_m or (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    if len(roi) != 4 or not np.isfinite(roi).all() or roi[1] <= roi[0] or roi[3] <= roi[2]:
        raise ValueError("roi_bounds_m must be finite (xmin, xmax, ymin, ymax)")
    thresholds = tuple(float(value) for value in loss_thresholds_m)
    if not thresholds or any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("loss_thresholds_m must contain positive finite values")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("loss_thresholds_m must be unique and increasing")
    loss = reference - final
    nominal = _nominal_graph_metrics(normalized_graph, roi)
    minimum_support_widths = [
        event["geometry_support_width_m"]["min"]
        for event in nominal["line_events"]
        if event["geometry_support_width_m"]["min"] is not None
    ]
    width_threshold_m = 4.0 * max(dx, dy)
    for event in nominal["line_events"]:
        width_summary = event["geometry_support_width_m"]
        if width_summary["min"] is not None:
            width_summary["resolution_threshold_m"] = width_threshold_m
            width_summary["resolution_status"] = (
                "under_sampled" if width_summary["min"] < width_threshold_m else "resolved_for_numerical_test"
            )
    extracted: dict[str, Any] = {}
    for threshold in thresholds:
        mask = loss >= threshold
        if int(mask.sum()) < 3:
            extracted[f"{threshold:g}"] = {
                "status": "unavailable",
                "reason": "fewer than three grid samples exceed the fixed loss threshold",
                "threshold_m": threshold,
                "sample_count": int(mask.sum()),
            }
            continue
        skeleton = _thin(mask)
        distance, nearest_outside = ndimage.distance_transform_edt(
            mask, sampling=(dx, dy), return_indices=True
        )
        widths = 2.0 * distance[skeleton]
        boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        if boundary.any():
            nearest_loss = loss[nearest_outside[0][boundary], nearest_outside[1][boundary]]
            edge_jump = np.abs(loss[boundary] - nearest_loss)
        else:
            edge_jump = np.empty(0)
        # The edge statistic records a grid-scale transition proxy; it is not
        # interpreted as a physical crack-lip discontinuity.
        skeleton_stats = _skeleton_metrics(skeleton, dx, dy)
        extracted[f"{threshold:g}"] = {
            "status": "available",
            "threshold_m": threshold,
            "loss_sample_count": int(mask.sum()),
            "component_count": skeleton_stats["pixel_component_count"],
            "centerline_length_m": skeleton_stats["pixel_skeleton_length_m"],
            "component_profiles": _component_profiles(skeleton, distance, loss, dx, dy),
            "roi_censored_endpoints": skeleton_stats["roi_censored_endpoints"],
            "branch_pixel_count": skeleton_stats["branch_pixel_count"],
            "branch_cluster_count": skeleton_stats["branch_cluster_count"],
            "junction_degree_histogram": skeleton_stats["junction_degree_histogram"],
            "topology_graph_rule": skeleton_stats["topology_graph_rule"],
            "branch_grouping_rule": skeleton_stats["branch_grouping_rule"],
            "effective_width_m": {
                "definition": "threshold-mask local inscribed-diameter proxy at independently thinned skeleton pixels",
                "sample_count": int(widths.size),
                "min": float(widths.min()) if widths.size else None,
                "median": float(np.median(widths)) if widths.size else None,
                "max": float(widths.max()) if widths.size else None,
                "status": (
                    "under_sampled"
                    if minimum_support_widths and min(minimum_support_widths) < width_threshold_m
                    else "available_for_numerical_test"
                ),
                "reason": (
                    "minimum nominal support width is below four grid intervals"
                    if minimum_support_widths and min(minimum_support_widths) < width_threshold_m
                    else None
                ),
                "resolution_threshold_m": width_threshold_m,
                "resolution_caveat": "four grid intervals is an engineering numerical-test condition, not sensor precision",
                "physical_width_caveat": "not a normal-crossing width; curved, asymmetric, and branched masks can differ from physical opening width",
            },
            "visible_loss_depth_m": {
                "min": float(loss[mask].min()),
                "median": float(np.median(loss[mask])),
                "q90": float(np.quantile(loss[mask], 0.9)),
                "max": float(loss[mask].max()),
            },
            "grid_edge_transition_proxy_m": {
                "definition": "absolute loss difference to 3x3 local maximum at threshold boundary; not a physical lip estimate",
                "median": float(np.median(edge_jump)) if edge_jump.size else None,
                "q90": float(np.quantile(edge_jump, 0.9)) if edge_jump.size else None,
            },
            **_background_cell_metrics(mask, dx, dy),
            "topology_warning": skeleton_stats["topology_warning"],
        }
    return {
        "status": "diagnostic_only",
        "real_gap_status": "real_gap_not_evaluated",
        "real_gap_not_evaluated": True,
        "coordinate_units": "m",
        "grid_layout": "x-major",
        "grid_spacing_m": {"x": dx, "y": dy},
        "reference_definition": "caller-supplied frozen reference surface",
        "residual_definition": "reference_z_m - final_z_m; positive values are visible depressions",
        "residual_may_include_texture_changes": True,
        "roi_bounds_m": list(roi),
        "loss_thresholds_m": list(thresholds),
        "loss_threshold_source": "explicit caller-supplied engineering thresholds; not a standard or calibrated crack definition",
        "nominal_vector_graph": nominal,
        "surface_extracted": extracted,
        "scope_limit": "single-valued visible surface only; no subsurface depth, hidden topology, or overhang inference",
    }


__all__ = ["analyze_crack_geometry", "normalize_v2_graph_json"]
