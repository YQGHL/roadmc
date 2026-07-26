"""Run the R2 ablation matrix under one frozen protocol.

统一协议（论文消融表的最低要求）：同一份冻结数据（credibility_v2）、
同一 split、同一超参、每格多个 seed；每格报告 test disease IoU +
scene-block bootstrap CI + 校准指标 + 显存/吞吐。阈值一律在 val 前缀
场景上单独扫描并冻结，再在 test 上单次评估——绝不用 test 选阈值。

消融维度（可用 --dims 子选）：
  mixing    none | dscm | hc2 | hc4     （xHC 语境下的必答对照）
  backbone  swin3d | pointmamba         （层级注意力 vs 门控 EMA）
  optimizer muon | adamw                （match_rms 路由修复后重跑）
  partition columnar | cubic            （2.5D 柱状分窗的正面消融）

用法::

    python roadmc/scripts/run_ablation_matrix.py \
        --data-dir ./data/credibility_v2 --out-dir ./output/ablation_r2 \
        --dims mixing --seeds 42 43 --max-epochs 15

结果汇总写入 ``<out-dir>/ablation_summary.json``；已完成的格子会被跳过
（断点续跑），因此可以分多次执行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# 分配器配置（2026-07-26 根因裁决的建议）：
# - 不用 expandable_segments —— Windows 上是彻底 no-op；
# - garbage_collection_threshold 让分配器在压力下主动回收；
# - max_split_size_mb 抑制大块被切碎。
# 注意力后端由 train.py/evaluate.py 内部固定为 math（见其注释）。
ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:512"

# 基准配置：与基线 v2 完全一致，消融只改一个维度。
BASE = {
    "backbone": "swin3d",
    "optimizer": "muon",
    "mixing": "dscm",
    "partition": "columnar",
    "embed_dim": 48,
    "depths": ["2", "2", "6", "2"],
    "num_heads": ["3", "3", "6", "6"],
    "window_size": 32,
    "batch_size": 8,
    "max_points": 2048,
    "lr": 1e-3,
    "drop_path_rate": 0.1,
}

DIM_VALUES = {
    "mixing": ["none", "dscm", "hc2", "hc4"],
    "backbone": ["swin3d", "pointmamba"],
    "optimizer": ["muon", "adamw"],
    "partition": ["columnar", "cubic"],
}


def cell_name(dim: str, value: str, seed: int) -> str:
    return f"{dim}={value}_seed{seed}"


def train_one(cell_dir: Path, cfg: Dict[str, Any], args) -> Path:
    """训练一格，返回最佳 checkpoint 路径。"""
    cmd = [
        PYTHON, str(REPO_ROOT / "roadmc" / "train.py"), "baseline",
        "--data_dir", args.data_dir,
        "--label_stage", "binary",
        "--backbone", cfg["backbone"],
        "--optimizer", cfg["optimizer"],
        "--mixing", cfg["mixing"],
        "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--max_points", str(cfg["max_points"]),
        "--embed_dim", str(cfg["embed_dim"]),
        "--depths", *cfg["depths"],
        "--num_heads", *cfg["num_heads"],
        "--window_size", str(cfg["window_size"]),
        "--max_epochs", str(args.max_epochs),
        "--num_workers", str(args.num_workers),
        "--precision", "16-mixed",
        "--auto_class_weights",
        "--metric_min_support", "500",
        "--drop_path_rate", str(cfg["drop_path_rate"]),
        "--seed", str(cfg["seed"]),
        "--run_dir", str(cell_dir),
    ]
    log_path = cell_dir / "train.log"
    cell_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True, env=env)

    ckpts = sorted((cell_dir / "lightning_logs").rglob("*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"no checkpoint produced in {cell_dir}")
    # ModelCheckpoint 文件名带 val_mIoU=，取最大者。
    def score(p: Path) -> float:
        stem = p.stem
        marker = "val_mIoU="
        return float(stem.split(marker)[-1]) if marker in stem else -1.0
    return max(ckpts, key=score)


def evaluate_one(ckpt: Path, cell_dir: Path, args) -> Dict[str, Any]:
    """val 扫阈值冻结 → test 单次评估。返回汇总指标。"""
    val_json = cell_dir / "eval_val.json"
    test_json = cell_dir / "eval_test.json"

    base_cmd = [
        PYTHON, str(REPO_ROOT / "roadmc" / "evaluate.py"),
        "--checkpoint", str(ckpt),
        "--data-dir", args.data_dir,
        "--label-stage", "binary",
        "--max-points", str(BASE["max_points"]),
        "--bootstrap-samples", str(args.bootstrap_samples),
    ]
    subprocess.run(
        base_cmd + ["--split", "val", "--scan-binary-thresholds",
                    "--threshold-calibration-scenes", str(args.calibration_scenes),
                    "--output-json", str(val_json)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    val_report = json.loads(val_json.read_text(encoding="utf-8"))
    threshold = _extract_threshold(val_report)

    subprocess.run(
        base_cmd + ["--split", "test", "--binary-threshold", str(threshold),
                    "--output-json", str(test_json)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    test_report = json.loads(test_json.read_text(encoding="utf-8"))
    return {
        "frozen_threshold": threshold,
        "val": _summarize(val_report),
        "test": _summarize(test_report),
    }


def _extract_threshold(report: Dict[str, Any]) -> float:
    """从 val 报告取在校准前缀场景上选出的阈值。

    优先 ``threshold_selection.selected_threshold``（前缀校准 + 独立
    评估的口径）；退回顶层 ``binary_threshold``。
    """
    selection = report.get("threshold_selection") or {}
    if selection.get("selected_threshold") is not None:
        return float(selection["selected_threshold"])
    if report.get("binary_threshold") is not None:
        return float(report["binary_threshold"])
    raise KeyError("no threshold found in val report; inspect the JSON schema")


def _summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics") or {}
    calibration = report.get("calibration") or {}
    bootstrap = report.get("scene_bootstrap") or {}
    return {
        "foreground_iou": metrics.get("foreground_iou"),
        "foreground_precision": metrics.get("foreground_precision"),
        "foreground_recall": metrics.get("foreground_recall"),
        "all_non_background_miou": metrics.get("all_non_background_miou"),
        "ece": calibration.get("ece"),
        "brier": calibration.get("brier"),
        "nll": calibration.get("nll"),
        "bootstrap": bootstrap.get("metrics"),
        "bootstrap_scenes": bootstrap.get("n_scenes"),
        "threshold": report.get("binary_threshold"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/credibility_v2")
    parser.add_argument("--out-dir", default="./output/ablation_r2")
    parser.add_argument("--dims", nargs="+", default=["mixing"],
                        choices=sorted(DIM_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--calibration-scenes", type=int, default=170)
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要执行的格子")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ablation_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    cells: List[Dict[str, Any]] = []
    for dim in args.dims:
        for value in DIM_VALUES[dim]:
            for seed in args.seeds:
                cfg = dict(BASE)
                cfg[dim] = value
                cfg["seed"] = seed
                cells.append({"dim": dim, "value": value, "seed": seed, "cfg": cfg})

    print(f"Ablation cells: {len(cells)} "
          f"(dims={args.dims}, seeds={args.seeds}, epochs={args.max_epochs})")
    for cell in cells:
        print(f"  - {cell_name(cell['dim'], cell['value'], cell['seed'])}")
    if args.dry_run:
        return

    for cell in cells:
        name = cell_name(cell["dim"], cell["value"], cell["seed"])
        if name in summary and summary[name].get("status") == "done":
            print(f"[skip] {name} (already complete)")
            continue
        cell_dir = out_dir / name
        print(f"[run ] {name}")
        started = time.time()
        try:
            ckpt = train_one(cell_dir, cell["cfg"], args)
            metrics = evaluate_one(ckpt, cell_dir, args)
            summary[name] = {
                "status": "done",
                "dim": cell["dim"], "value": cell["value"], "seed": cell["seed"],
                "config": {k: v for k, v in cell["cfg"].items() if k != "seed"},
                "checkpoint": str(ckpt),
                "elapsed_sec": time.time() - started,
                **metrics,
            }
        except subprocess.CalledProcessError as exc:
            summary[name] = {"status": "failed", "returncode": exc.returncode,
                             "log": str(cell_dir / "train.log")}
            print(f"[FAIL] {name}: see {cell_dir / 'train.log'}")
        except Exception as exc:  # pragma: no cover - operational
            summary[name] = {"status": "error", "error": repr(exc)}
            print(f"[ERR ] {name}: {exc!r}")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\nSummary: {summary_path}")
    done = [k for k, v in summary.items() if v.get("status") == "done"]
    print(f"Completed cells: {len(done)}/{len(cells)}")


if __name__ == "__main__":
    main()
