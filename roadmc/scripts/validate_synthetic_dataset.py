"""Validate RoadMC synthetic scenes before a reported training experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.features import (
    DEFAULT_GEOMETRY_K_NEIGHBORS,
    OBSERVABLE_FEATURE_SCHEMA,
    compute_observable_features,
    has_observable_feature_schema,
)


def validate_scene(path: Path, *, reconstruct_features: bool = False) -> dict:
    """Check one file against the RoadMC physical-data and feature contracts."""

    result: dict[str, object] = {"path": str(path), "errors": [], "warnings": []}
    errors: list[str] = result["errors"]  # type: ignore[assignment]
    warnings: list[str] = result["warnings"]  # type: ignore[assignment]
    try:
        with np.load(path, allow_pickle=False) as scene:
            required = {"points", "labels", "feats", "normals", "feature_schema"}
            missing = sorted(required - set(scene.files))
            if missing:
                errors.append(f"missing keys: {missing}")
                return result
            points = np.asarray(scene["points"], dtype=np.float32)
            labels = np.asarray(scene["labels"], dtype=np.int64).reshape(-1)
            feats = np.asarray(scene["feats"], dtype=np.float32)
            normals = np.asarray(scene["normals"], dtype=np.float32)
            if not has_observable_feature_schema(scene["feature_schema"]):
                errors.append(f"feature_schema is not {OBSERVABLE_FEATURE_SCHEMA}")
            if points.ndim != 2 or points.shape[1] != 3:
                errors.append(f"points shape must be (N, 3), got {points.shape}")
                return result
            if labels.shape != (len(points),):
                errors.append(f"labels shape {labels.shape} does not match ({len(points)},)")
            if feats.shape != (len(points), 3):
                errors.append(f"feats shape {feats.shape} does not match ({len(points)}, 3)")
            if normals.shape != (len(points), 3):
                errors.append(f"normals shape {normals.shape} does not match ({len(points)}, 3)")
            finite_arrays = (
                np.isfinite(points).all()
                and np.isfinite(feats).all()
                and np.isfinite(normals).all()
            )
            if not finite_arrays:
                errors.append("points, feats, and normals must all be finite")
            if np.any((labels < 0) | (labels > 37)):
                errors.append("labels must lie in [0, 37]")
            if len(feats) and (np.any(feats[:, 0] < -1e-6) or np.any(feats[:, 0] > 1.0 + 1e-6)):
                errors.append("normalized intensity is outside [0, 1]")
            if len(feats) and (np.any(feats[:, 1] < -1e-6) or np.any(feats[:, 1] > 1.0 + 1e-6)):
                errors.append("PCA curvature is outside [0, 1]")
            if len(feats) and np.any(np.abs(feats[:, 2]) > 3.0 + 1e-6):
                errors.append("scaled local-height residual exceeds configured clip")
            if len(normals):
                lengths = np.linalg.norm(normals, axis=1)
                if np.quantile(np.abs(lengths - 1.0), 0.95) > 5e-3:
                    warnings.append("more than 5% of normals differ from unit length by > 0.005")
            if "target_label" in scene.files:
                target = int(scene["target_label"])
                if target >= 1 and not np.any(labels == target):
                    errors.append(f"forced target label {target} is absent")
            if reconstruct_features and not errors:
                expected = compute_observable_features(
                    points,
                    feats[:, 0],
                    k_neighbors=DEFAULT_GEOMETRY_K_NEIGHBORS,
                )
                if not np.allclose(feats, expected, rtol=3e-5, atol=3e-6):
                    max_error = float(np.max(np.abs(feats - expected)))
                    errors.append(
                        f"observable feature reconstruction mismatch (max_abs={max_error:.6g})"
                    )
            result["point_count"] = int(len(points))
            result["labels"] = np.unique(labels).astype(int).tolist()
    except Exception as exc:  # pragma: no cover - caller receives file-specific evidence
        errors.append(f"unreadable scene: {exc!r}")
    return result


def validate_dataset(
    data_dir: str | Path,
    *,
    split: str,
    feature_check_scenes: int = 0,
) -> dict:
    """Validate one split and return a JSON-safe report."""

    root = Path(data_dir) / split
    paths = sorted(root.glob("scene_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No scene_*.npz files in {root}")
    checked = set(paths[: max(0, feature_check_scenes)])
    reports = [validate_scene(path, reconstruct_features=path in checked) for path in paths]
    errors = [
        {"path": report["path"], "errors": report["errors"]}
        for report in reports
        if report["errors"]
    ]
    return {
        "data_dir": str(Path(data_dir)),
        "split": split,
        "feature_schema": OBSERVABLE_FEATURE_SCHEMA,
        "scene_count": len(paths),
        "feature_reconstruction_scenes": len(checked),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors[:50],
        "warnings": [
            {"path": report["path"], "warnings": report["warnings"]}
            for report in reports
            if report["warnings"]
        ][:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument(
        "--feature-check-scenes",
        type=int,
        default=32,
        help="Recompute features for this many deterministic scenes per split",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    splits = ("train", "val") if args.split == "both" else (args.split,)
    reports = {
        split: validate_dataset(
            args.data_dir,
            split=split,
            feature_check_scenes=args.feature_check_scenes,
        )
        for split in splits
    }
    output = {
        "data_dir": str(Path(args.data_dir)),
        "feature_schema": OBSERVABLE_FEATURE_SCHEMA,
        "reports": reports,
        "valid": all(report["valid"] for report in reports.values()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report: {output_path}")
    if not output["valid"]:
        raise SystemExit(
            "Synthetic data validation failed; inspect the JSON report before training."
        )


if __name__ == "__main__":
    main()
