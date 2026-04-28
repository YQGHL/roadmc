"""
RoadMC Evaluation Pipeline.

Loads a trained segmentation model checkpoint and computes
per-class metrics following JTG 5210—2018 standard.

Usage:
    python roadmc/evaluate.py --checkpoint ./lightning_logs/checkpoints/best.ckpt \
        --data_dir ./data/synthetic_output

Output:
    - JSON report to stdout
    - Detailed per-class tables grouped by asphalt/concrete
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

# JTG 5210-2018 label info
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

ASPHALT_LABELS = set(range(0, 21))
CONCRETE_LABELS = set(range(21, 38))


def compute_per_class_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 38,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class IoU, recall, precision for all classes.
    
    Args:
        preds: (N,) predicted labels.
        targets: (N,) ground truth labels.
        num_classes: Number of classes (default 38).
    
    Returns:
        dict mapping class_id → {iou, recall, precision}.
    """
    metrics = {}
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        tp = (pred_c & target_c).sum().float()
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()
        
        iou = (tp / (tp + fp + fn + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        precision = (tp / (tp + fp + 1e-8)).item()
        
        metrics[str(c)] = {"iou": iou, "recall": recall, "precision": precision}
    
    return metrics


def compute_grouped_metrics(
    per_class: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute grouped statistics for asphalt [0-20] and concrete [21-37].
    
    Args:
        per_class: Output from compute_per_class_metrics.
    
    Returns:
        dict with 'asphalt_mIoU', 'concrete_mIoU', 'overall_mIoU', etc.
    """
    ious = {int(k): v["iou"] for k, v in per_class.items()}
    
    asphalt_ious = [ious[c] for c in range(1, 21) if str(c) in per_class]
    concrete_ious = [ious[c] for c in range(21, 38) if str(c) in per_class]
    all_ious = [ious[c] for c in range(1, 38) if str(c) in per_class]
    
    result = {
        "asphalt_mIoU": float(np.mean(asphalt_ious)) if asphalt_ious else 0.0,
        "concrete_mIoU": float(np.mean(concrete_ious)) if concrete_ious else 0.0,
        "overall_mIoU": float(np.mean(all_ious)) if all_ious else 0.0,
        "num_asphalt_classes": len(asphalt_ious),
        "num_concrete_classes": len(concrete_ious),
    }
    return result


def format_report(
    per_class: Dict[str, Dict[str, float]],
    grouped: Dict[str, float],
) -> str:
    """Format evaluation results as a readable string."""
    lines = []
    lines.append("=" * 72)
    lines.append("  RoadMC Evaluation Report — JTG 5210-2018")
    lines.append("=" * 72)
    lines.append(f"\n  Overall mIoU:         {grouped['overall_mIoU']:.4f}")
    lines.append(f"  Asphalt mIoU (1-20):  {grouped['asphalt_mIoU']:.4f} ({grouped['num_asphalt_classes']} classes)")
    lines.append(f"  Concrete mIoU (21-37): {grouped['concrete_mIoU']:.4f} ({grouped['num_concrete_classes']} classes)")
    lines.append("\n" + "-" * 72)
    lines.append(f"  {'ID':<4} {'Name':<28} {'IoU':<8} {'Recall':<8} {'Precision':<8}")
    lines.append("  " + "-" * 60)
    
    for c in sorted(int(k) for k in per_class):
        m = per_class[str(c)]
        name = LABEL_NAMES.get(c, f"Class {c}")
        lines.append(f"  {c:<4} {name:<28} {m['iou']:<8.4f} {m['recall']:<8.4f} {m['precision']:<8.4f}")
    
    lines.append("=" * 72)
    return "\n".join(lines)


def evaluate(args):
    """Main evaluation flow."""
    from roadmc.models.model_pl import RoadMCSegModel
    from roadmc.data.dataloader import SyntheticPointCloudDataset
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model from checkpoint
    model = RoadMCSegModel.load_from_checkpoint(
        args.checkpoint,
        in_channels=3, num_classes=38,
    )
    model = model.to(device).eval()
    
    # Load validation data
    dataset = SyntheticPointCloudDataset(
        args.data_dir, split="val",
        max_points=args.max_points,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=lambda batch: batch[0],  # single sample
    )
    
    all_preds, all_targets = [], []
    for batch in loader:
        coords = batch["coords"].unsqueeze(0).to(device)
        feats = batch["feats"].unsqueeze(0).to(device)
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            logits = model(coords, feats)
        
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds.flatten())
        all_targets.append(labels.cpu().flatten())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    per_class = compute_per_class_metrics(preds, targets)
    grouped = compute_grouped_metrics(per_class)
    
    # Format and output
    report = format_report(per_class, grouped)
    print(report)
    
    # JSON output
    if args.output_json:
        output = {
            "per_class": {k: v for k, v in sorted(per_class.items())},
            "grouped": grouped,
            "total_points": len(targets),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  JSON report saved to: {args.output_json}")


def main():
    parser = argparse.ArgumentParser(description="RoadMC Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/synthetic_output",
                        help="Validation data directory")
    parser.add_argument("--max_points", type=int, default=65536)
    parser.add_argument("--output_json", type=str, default="",
                        help="Path to save JSON report (optional)")
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == "__main__":
    # If called without arguments, run self-test; otherwise run evaluation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    if len(sys.argv) <= 1 or not any(a.startswith("--") for a in sys.argv[1:]):
        from roadmc.models.model_pl import RoadMCSegModel
        from roadmc.data.dataloader import SyntheticPointCloudDataset
        
        preds = torch.randint(0, 38, (1000,))
        targets = torch.randint(0, 38, (1000,))
        metrics = compute_per_class_metrics(preds, targets)
        grouped = compute_grouped_metrics(metrics)
        print(f"[PASS] compute_per_class_metrics: {len(metrics)} classes")
        print(f"[PASS] grouped: asphalt_mIoU={grouped['asphalt_mIoU']:.4f}")
        
        report = format_report(metrics, grouped)
        print(report[:200] + "...")
        print("\n[PASS] Evaluation module ready.")
    else:
        main()
