"""Compute reproducible bounded effective-number class weights for RoadMC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.class_balance import (
    class_balance_summary,
    effective_number_class_weights,
    point_class_counts,
)
from roadmc.data.curriculum import normalize_label_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--label-stage",
        choices=("binary", "four", "eight", "full38"),
        default="full38",
    )
    parser.add_argument("--beta", type=float, default=0.999999)
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--output", required=True, help="Output .pt file")
    args = parser.parse_args()

    stage = normalize_label_stage(args.label_stage)
    counts = point_class_counts(args.data_dir, split=args.split, label_stage=stage)
    weights = effective_number_class_weights(counts, beta=args.beta, max_weight=args.max_weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(weights), output)

    report = {
        "data_dir": str(Path(args.data_dir)),
        "split": args.split,
        "label_stage": stage,
        "beta": args.beta,
        "max_weight": args.max_weight,
        **class_balance_summary(counts, weights),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved weights: {output}")
    print(f"Saved report: {report_path}")
    print(f"Counts: {report['counts']}")
    print(f"Weights: {report['weights']}")


if __name__ == "__main__":
    main()
