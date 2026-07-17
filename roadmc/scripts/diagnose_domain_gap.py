"""Compare observable geometry statistics between two point-cloud domains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.real.dataset import RealRoadDataset
from roadmc.domain_gap import OBSERVABLE_NAMES, PointCloudRecord, compare_domains, dominant_ground_surface


def _scene_paths(root: Path, split: str, pattern: str, max_scenes: int) -> list[Path]:
    candidate = root / split
    directory = candidate if candidate.exists() else root
    paths = sorted(directory.glob(pattern))
    return paths[:max_scenes]


def load_synthetic_records(root: Path, split: str, max_scenes: int) -> tuple[list[PointCloudRecord], list[str]]:
    records: list[PointCloudRecord] = []
    warnings: list[str] = []
    for path in _scene_paths(root, split, "scene_*.npz", max_scenes):
        with np.load(path, allow_pickle=False) as data:
            points = data["points"].astype(np.float64)
            if "coordinates_normalized" in data.files and bool(data["coordinates_normalized"]):
                if "coordinate_center" not in data.files or "coordinate_scale" not in data.files:
                    warnings.append(f"{path.name}: missing inverse normalization fields")
                else:
                    points = points * float(data["coordinate_scale"]) + data["coordinate_center"]
            else:
                warnings.append(
                    f"{path.name}: no metric inverse normalization metadata; density uses stored coordinates"
                )
            intensities = data["feats"][:, 0] if "feats" in data.files else None
            normals = data["normals"] if "normals" in data.files else None
        records.append(PointCloudRecord(points, intensities, normals, name=path.name))
    if not records:
        raise FileNotFoundError(f"No synthetic scene_*.npz files found under {root / split}")
    return records, warnings


def load_real_records(
    root: Path,
    pattern: str,
    max_scenes: int,
    require_metadata: bool,
) -> list[PointCloudRecord]:
    paths = _scene_paths(root, split="", pattern=pattern, max_scenes=max_scenes)
    records = []
    for path in paths:
        points, _, normals, intensities, _ = RealRoadDataset.load_scene(
            path, require_metadata=require_metadata
        )
        records.append(PointCloudRecord(points, intensities, normals, name=path.name))
    if not records:
        raise FileNotFoundError(f"No supported real files matching {pattern!r} under {root}")
    return records


def _load_domain(
    kind: str,
    root: Path,
    split: str,
    pattern: str,
    max_scenes: int,
    require_metadata: bool,
) -> tuple[list[PointCloudRecord], list[str]]:
    if kind == "synthetic":
        return load_synthetic_records(root, split, max_scenes)
    return load_real_records(root, pattern, max_scenes, require_metadata), []


def _print_summary(report: dict) -> None:
    print("=" * 88)
    print("RoadMC Observable Domain-Gap Report")
    print("=" * 88)
    print(f"Source scenes: {report['source_scene_count']}  Target scenes: {report['target_scene_count']}")
    print(f"Joint RBF-MMD: {report['joint_rbf_mmd']}")
    print("Descriptor                 W1             Energy         RBF-MMD")
    for name in OBSERVABLE_NAMES:
        item = report["descriptors"][name]
        distances = item["distances"]
        if not item["available"]:
            print(f"{name:<25.25} unavailable")
            continue
        print(
            f"{name:<25.25} {distances['wasserstein_1']:<14.6g} "
            f"{distances['energy']:<14.6g} {distances['rbf_mmd']:<14.6g}"
        )
    print("=" * 88)


def _apply_ground_filter(
    records: Iterable[PointCloudRecord], distance_threshold: float, seed: int
) -> tuple[list[PointCloudRecord], list[dict]]:
    filtered = []
    details = []
    for index, record in enumerate(records):
        selected, info = dominant_ground_surface(
            record,
            distance_threshold=distance_threshold,
            seed=seed + index,
        )
        filtered.append(selected)
        details.append({"name": record.name, **info})
    return filtered, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--source-kind", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--target-kind", choices=("synthetic", "real"), default="real")
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--target-split", default="val")
    parser.add_argument("--source-pattern", default="*")
    parser.add_argument("--target-pattern", default="*")
    parser.add_argument("--max-scenes", type=int, default=100)
    parser.add_argument("--max-points-per-scene", type=int, default=4096)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--mmd-max-samples", type=int, default=512)
    parser.add_argument("--source-ground-plane", action="store_true")
    parser.add_argument("--target-ground-plane", action="store_true")
    parser.add_argument("--ground-distance-threshold", type=float, default=0.15)
    parser.add_argument("--require-real-metadata", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    source, source_warnings = _load_domain(
        args.source_kind,
        Path(args.source_dir),
        args.source_split,
        args.source_pattern,
        args.max_scenes,
        args.require_real_metadata,
    )
    target, target_warnings = _load_domain(
        args.target_kind,
        Path(args.target_dir),
        args.target_split,
        args.target_pattern,
        args.max_scenes,
        args.require_real_metadata,
    )
    source_ground = []
    target_ground = []
    if args.source_ground_plane:
        source, source_ground = _apply_ground_filter(
            source, args.ground_distance_threshold, args.seed
        )
    if args.target_ground_plane:
        target, target_ground = _apply_ground_filter(
            target, args.ground_distance_threshold, args.seed + 100000
        )
    report = compare_domains(
        source,
        target,
        k_neighbors=args.k_neighbors,
        max_points_per_scene=args.max_points_per_scene,
        mmd_max_samples=args.mmd_max_samples,
        seed=args.seed,
    )
    report["source_kind"] = args.source_kind
    report["target_kind"] = args.target_kind
    report["warnings"] = source_warnings + target_warnings
    report["source_ground_filter"] = source_ground
    report["target_ground_filter"] = target_ground
    _print_summary(report)
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"JSON report saved to: {output}")


if __name__ == "__main__":
    main()
