"""Statistically credible offline evaluation for RoadMC checkpoints.

The evaluator accumulates one global confusion matrix over a fixed split and
uses complete scenes as bootstrap blocks.  It deliberately does not average
per-batch mIoU because classes absent from a batch make that estimate unstable
for long-tailed segmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roadmc.data.curriculum import (
    class_names_for_stage,
    normalize_label_stage,
    num_classes_for_stage,
    stage_for_num_classes,
)
from roadmc.data.features import require_observable_checkpoint_schema
from roadmc.metrics import (
    CalibrationAccumulator,
    bootstrap_scene_confidence_intervals,
    confusion_matrix_from_predictions,
    metrics_from_confusion,
    scan_binary_thresholds,
)

LABEL_NAMES = {
    0: "Background", 1: "Alligator Light", 2: "Alligator Severe",
    3: "Block Crack Light", 4: "Block Crack Severe",
    5: "Longitudinal Crack Light", 6: "Longitudinal Crack Severe",
    7: "Transverse Crack Light", 8: "Transverse Crack Severe",
    9: "Pothole Light", 10: "Pothole Severe",
    11: "Raveling Light", 12: "Raveling Severe",
    13: "Depression Light", 14: "Depression Severe",
    15: "Rutting Light", 16: "Rutting Severe",
    17: "Corrugation Light", 18: "Corrugation Severe",
    19: "Bleeding", 20: "Patching (Asphalt)",
    21: "Slab Shatter Light", 22: "Slab Shatter Severe",
    23: "Concrete Crack Light", 24: "Concrete Crack Severe",
    25: "Corner Break Light", 26: "Corner Break Severe",
    27: "Faulting Light", 28: "Faulting Severe",
    29: "Pumping", 30: "Edge Spall Light", 31: "Edge Spall Severe",
    32: "Joint Damage Light", 33: "Joint Damage Severe",
    34: "Pitting", 35: "Blowup", 36: "Exposed Aggregate",
    37: "Patching (Concrete)",
}


def _parse_labels(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _checkpoint_num_classes(checkpoint: Path, fallback: int | None) -> int:
    if fallback is not None:
        return fallback
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        hyper_parameters = payload.get("hyper_parameters", {})
        value = hyper_parameters.get("num_classes")
        if value is not None:
            return int(value)
    except Exception:
        pass
    return 38


def _checkpoint_input_point_count(payload: dict) -> int | None:
    hyper_parameters = payload.get("hyper_parameters", {})
    value = hyper_parameters.get("input_point_count") if isinstance(hyper_parameters, dict) else None
    return int(value) if value is not None else None


def _threshold_values(start: float, end: float, step: float) -> list[float]:
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0 or step <= 0.0:
        raise ValueError("threshold range must satisfy 0 <= start/end <= 1 and step > 0")
    if start > end:
        raise ValueError("threshold start must be <= threshold end")
    return [float(value) for value in np.arange(start, end + step * 0.25, step)]


def _report_lines(
    metrics: dict,
    calibration: dict,
    bootstrap: dict | None,
    label_stage: str,
) -> list[str]:
    lines = ["=" * 82, "RoadMC Global Evaluation Report", "=" * 82]
    lines.append(f"Total evaluated points:      {metrics['total_points']:,}")
    lines.append(f"Supported classes:           {metrics['supported_labels']}")
    lines.append(f"Supported-class mIoU:        {metrics['supported_miou']:.4f}")
    lines.append(f"All non-background mIoU:     {metrics['all_non_background_miou']:.4f}")
    lines.append(f"Foreground IoU:              {metrics['foreground_iou']:.4f}")
    lines.append(f"Foreground precision/recall: {metrics['foreground_precision']:.4f} / {metrics['foreground_recall']:.4f}")
    lines.append(f"ECE / Brier / NLL:           {calibration['ece']:.4f} / {calibration['brier']:.4f} / {calibration['nll']:.4f}")
    if bootstrap and bootstrap.get("metrics"):
        ci = bootstrap["metrics"].get("supported_miou")
        if ci and "lower" in ci and "upper" in ci:
            lines.append(
                f"Scene bootstrap {bootstrap['confidence']:.0%} CI: "
                f"[{ci['lower']:.4f}, {ci['upper']:.4f}] (n={bootstrap['n_scenes']}, B={bootstrap['n_bootstrap']})"
            )
    lines.append("-" * 82)
    lines.append("ID  Name                         Support      IoU     Recall  Precision  Included")
    class_names = class_names_for_stage(label_stage)
    for item in metrics["per_class"]:
        class_id = item["class_id"]
        if label_stage == "full38":
            name = LABEL_NAMES.get(class_id, f"Class {class_id}")
        else:
            name = class_names[class_id]
        lines.append(
            f"{class_id:>2}  {name:<28.28} {item['support']:>9,}  {item['iou']:.4f}  "
            f"{item['recall']:.4f}  {item['precision']:.4f}  {str(item['supported']):>8}"
        )
    lines.append("=" * 82)
    return lines


def evaluate(args: argparse.Namespace) -> dict:
    from roadmc.data.dataloader import SyntheticPointCloudDataset
    from roadmc.models.model_pl import RoadMCSegModel

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_schema = require_observable_checkpoint_schema(
        checkpoint_payload,
        context="evaluation checkpoint",
    )
    checkpoint_point_count = _checkpoint_input_point_count(checkpoint_payload)
    if (
        checkpoint_point_count is not None
        and args.max_points is not None
        and checkpoint_point_count != args.max_points
        and not args.allow_input_point_mismatch
    ):
        raise ValueError(
            f"checkpoint was trained with max_points={checkpoint_point_count}, but evaluation requested "
            f"max_points={args.max_points}. Local geometric features depend on the sampled cloud; "
            "use the training point count or pass --allow-input-point-mismatch only for diagnostics."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.binary and args.label_stage not in ("auto", "binary"):
        raise ValueError("--binary cannot be combined with a non-binary --label-stage")
    checkpoint_num_classes = _checkpoint_num_classes(checkpoint, args.num_classes)
    if args.binary:
        label_stage = "binary"
    elif args.label_stage == "auto":
        label_stage = stage_for_num_classes(checkpoint_num_classes)
    else:
        label_stage = normalize_label_stage(args.label_stage)
    num_classes = num_classes_for_stage(label_stage)
    if checkpoint_num_classes != num_classes:
        raise ValueError(
            f"checkpoint has num_classes={checkpoint_num_classes}, but label_stage={label_stage} "
            f"requires {num_classes}"
        )

    model = RoadMCSegModel.load_from_checkpoint(checkpoint, map_location=device)
    if model.num_classes != num_classes:
        raise ValueError(
            f"checkpoint has num_classes={model.num_classes}, requested evaluation uses {num_classes}"
        )
    model = model.to(device).eval()

    dataset = SyntheticPointCloudDataset(
        args.data_dir,
        split=args.split,
        max_points=args.max_points,
        augment=False,
        label_stage=label_stage,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    scene_confusions: list[torch.Tensor] = []
    calibration = CalibrationAccumulator(n_bins=args.calibration_bins)
    binary_probabilities: list[np.ndarray] = []
    binary_targets: list[np.ndarray] = []
    threshold = args.binary_threshold
    calibration_scene_count = max(0, args.threshold_calibration_scenes)

    with torch.inference_mode():
        for scene_index, batch in enumerate(loader):
            coords = batch["coords"].unsqueeze(0).to(device, non_blocking=True)
            feats = batch["feats"].unsqueeze(0).to(device, non_blocking=True)
            targets = batch["labels"].long().to(device, non_blocking=True)
            logits = model(coords, feats)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

            if num_classes == 2:
                disease_probability = probabilities[:, 1]
                predictions = (disease_probability >= threshold).long()
                if args.scan_binary_thresholds:
                    binary_probabilities.append(disease_probability.detach().cpu().numpy())
                    binary_targets.append(targets.detach().cpu().numpy())
            else:
                predictions = probabilities.argmax(dim=-1)

            scene_confusion = confusion_matrix_from_predictions(predictions, targets, num_classes)
            confusion += scene_confusion.cpu()
            scene_confusions.append(scene_confusion.cpu())
            if not args.scan_binary_thresholds or scene_index >= calibration_scene_count:
                calibration.update(probabilities, targets)

    threshold_scan = []
    threshold_selection = None
    if args.scan_binary_thresholds:
        if num_classes != 2:
            raise ValueError("--scan-binary-thresholds requires a binary checkpoint")
        if calibration_scene_count >= len(binary_probabilities) and calibration_scene_count > 0:
            raise ValueError(
                "--threshold-calibration-scenes must leave at least one held-out evaluation scene"
            )
        selection_scene_count = calibration_scene_count or len(binary_probabilities)
        selection_probabilities = np.concatenate(binary_probabilities[:selection_scene_count])
        selection_targets = np.concatenate(binary_targets[:selection_scene_count])
        threshold_scan = scan_binary_thresholds(
            selection_probabilities,
            selection_targets,
            _threshold_values(args.threshold_start, args.threshold_end, args.threshold_step),
        )
        best = max(threshold_scan, key=lambda item: item["foreground_iou"])
        threshold = best["threshold"]
        evaluation_probabilities = binary_probabilities[selection_scene_count:] if calibration_scene_count else binary_probabilities
        evaluation_targets = binary_targets[selection_scene_count:] if calibration_scene_count else binary_targets
        confusion.zero_()
        scene_confusions = []
        for probabilities, targets_array in zip(evaluation_probabilities, evaluation_targets):
            prediction_tensor = torch.from_numpy((probabilities >= threshold).astype(np.int64))
            target_tensor = torch.from_numpy(targets_array.astype(np.int64))
            scene_confusion = confusion_matrix_from_predictions(
                prediction_tensor,
                target_tensor,
                num_classes,
            ).cpu()
            confusion += scene_confusion
            if calibration_scene_count:
                scene_confusions.append(scene_confusion)
        threshold_selection = {
            "selection_scene_count": selection_scene_count,
            "evaluation_scene_count": len(evaluation_probabilities),
            "independent_evaluation": calibration_scene_count > 0,
            "selected_threshold": threshold,
        }

    tail_labels = _parse_labels(args.tail_labels)
    metrics = metrics_from_confusion(
        confusion, min_support=args.min_support, tail_labels=tail_labels
    )
    calibration_summary = calibration.as_dict()
    bootstrap = bootstrap_scene_confidence_intervals(
        scene_confusions,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        confidence=args.bootstrap_confidence,
        min_support=args.min_support,
        tail_labels=tail_labels,
    )

    output = {
        "checkpoint": str(checkpoint),
        "data_dir": str(Path(args.data_dir)),
        "split": args.split,
        "num_classes": num_classes,
        "label_stage": label_stage,
        "feature_schema": feature_schema,
        "checkpoint_input_point_count": checkpoint_point_count,
        "evaluation_max_points": args.max_points,
        "binary_threshold": threshold if num_classes == 2 else None,
        "threshold_scan": threshold_scan,
        "threshold_selection": threshold_selection,
        "metrics": metrics,
        "calibration": calibration_summary,
        "scene_bootstrap": bootstrap,
    }
    print("\n".join(_report_lines(metrics, calibration_summary, bootstrap, label_stage)))
    if threshold_scan:
        best = max(threshold_scan, key=lambda item: item["foreground_iou"])
        if threshold_selection and threshold_selection["independent_evaluation"]:
            print(
                f"Best calibration threshold: {best['threshold']:.3f}, "
                f"calibration IoU={best['foreground_iou']:.4f} "
                f"(scenes={threshold_selection['selection_scene_count']})"
            )
        else:
            print(
                f"Best same-split scanned threshold: {best['threshold']:.3f}, "
                f"IoU={best['foreground_iou']:.4f}; no post-selection CI is reported"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"JSON report saved to: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="RoadMC global, calibrated evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data/synthetic_output")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument(
        "--allow-input-point-mismatch",
        action="store_true",
        help="Permit a diagnostic evaluation with a point count different from the checkpoint contract",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--label-stage",
        choices=["auto", "binary", "four", "eight", "full38"],
        default="auto",
        help="Curriculum label space; auto infers it from the checkpoint output count",
    )
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--scan-binary-thresholds", action="store_true")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--threshold-calibration-scenes",
        type=int,
        default=0,
        help="Use this many leading scenes only for threshold selection; remaining scenes form an independent report",
    )
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--tail-labels", type=str, default="")
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--output-json", type=str, default="")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
