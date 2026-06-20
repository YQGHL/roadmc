"""Grid search for binary classification hyperparameters.

Runs baseline --binary with different lr, weight_decay, lambda combinations.
Saves results summary and identifies best config.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Grid definition ──────────────────────────────────────────────────────
GRID = [
    # (lr, weight_decay, lambda_focal, lambda_dice, lambda_edge, label)
    (1e-4, 0.05, 1.0, 1.0, 0.0, "lr1e-4_wd0.05_edge0"),
    (5e-4, 0.05, 1.0, 1.0, 0.0, "lr5e-4_wd0.05_edge0"),
    (1e-4, 0.01, 1.0, 1.0, 0.0, "lr1e-4_wd0.01_edge0"),
    (5e-4, 0.01, 1.0, 1.0, 0.0, "lr5e-4_wd0.01_edge0"),
    (1e-3, 0.05, 1.0, 1.0, 0.0, "lr1e-3_wd0.05_edge0"),
    (1e-4, 0.05, 2.0, 0.5, 0.0, "lr1e-4_wd0.05_focal2_dice05"),
]

DATA_DIR = "./data/synthetic_gpu_run"
MAX_EPOCHS = 8
BATCH_SIZE = 2
MAX_POINTS = 1024
EMBED_DIM = 48
DEPTHS = (2, 2, 3, 2)
NUM_HEADS = (2, 4, 6, 12)


def parse_best_metrics(log_dir: Path) -> dict | None:
    """Parse metrics.csv and return best epoch's metrics."""
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        return None

    best_miou = -1.0
    best_row = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("val_mIoU") and row["val_mIoU"].strip():
                miou = float(row["val_mIoU"])
                if miou > best_miou:
                    best_miou = miou
                    best_row = {
                        "best_epoch": int(float(row["epoch"])),
                        "best_step": int(float(row["step"])),
                        "val_mIoU": miou,
                        "val_precision": float(row.get("val_precision_1", 0) or 0),
                        "val_recall": float(row.get("val_recall_1", 0) or 0),
                        "val_loss": float(row["val_loss"]),
                    }

    return best_row


def find_latest_version() -> int:
    """Find the latest lightning_logs version number."""
    logs_dir = Path("lightning_logs")
    if not logs_dir.exists():
        return -1
    versions = [d for d in logs_dir.iterdir() if d.name.startswith("version_")]
    if not versions:
        return -1
    nums = [int(d.name.split("_")[1]) for d in versions]
    return max(nums)


def main():
    results = []

    train_script = Path(__file__).resolve().parents[1] / "train.py"

    for lr, wd, lf, ld, le, label in GRID:
        print("\n" + "=" * 70)
        print(f"[GRID] Running: {label}")
        print(f"       lr={lr}, wd={wd}, lambda_focal={lf}, lambda_dice={ld}, lambda_edge={le}")
        print("=" * 70)

        version_before = find_latest_version()

        cmd = [
            sys.executable, str(train_script),
            "baseline",
            "--data_dir", DATA_DIR,
            "--max_epochs", str(MAX_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--max_points", str(MAX_POINTS),
            "--embed_dim", str(EMBED_DIM),
            "--depths", *map(str, DEPTHS),
            "--num_heads", *map(str, NUM_HEADS),
            "--lr", str(lr),
            "--weight_decay", str(wd),
            "--lambda_focal", str(lf),
            "--lambda_dice", str(ld),
            "--lambda_edge", str(le),
            "--binary",
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        # Find which version was created
        version_after = find_latest_version()
        new_versions = [v for v in range(version_before + 1, version_after + 1)]

        print(f"       Completed in {elapsed:.0f}s")
        if result.returncode != 0:
            print(f"       FAILED (returncode={result.returncode})")
            stderr_last = result.stderr[-500:] if result.stderr else ""
            stdout_last = result.stdout[-500:] if result.stdout else ""
            print(f"       stderr tail: {stderr_last}")
            print(f"       stdout tail: {stdout_last}")
            results.append({
                "label": label,
                "lr": lr, "wd": wd,
                "lambda_focal": lf, "lambda_dice": ld, "lambda_edge": le,
                "status": "FAILED",
                "elapsed_s": elapsed,
            })
            continue

        # Collect results from newest version
        best_metrics = None
        for v in reversed(new_versions):
            log_dir = Path("lightning_logs") / f"version_{v}"
            metrics = parse_best_metrics(log_dir)
            if metrics is not None:
                best_metrics = metrics
                print(f"       Version {v}: best_epoch={metrics['best_epoch']}, "
                      f"mIoU={metrics['val_mIoU']:.4f}, "
                      f"prec={metrics['val_precision']:.4f}, "
                      f"recall={metrics['val_recall']:.4f}, "
                      f"loss={metrics['val_loss']:.4f}")
                break

        if best_metrics:
            results.append({
                "label": label,
                "lr": lr, "wd": wd,
                "lambda_focal": lf, "lambda_dice": ld, "lambda_edge": le,
                "status": "OK",
                "version": v,
                "best_epoch": best_metrics["best_epoch"],
                "best_mIoU": best_metrics["val_mIoU"],
                "best_precision": best_metrics["val_precision"],
                "best_recall": best_metrics["val_recall"],
                "best_val_loss": best_metrics["val_loss"],
                "elapsed_s": elapsed,
            })
        else:
            results.append({
                "label": label,
                "lr": lr, "wd": wd,
                "lambda_focal": lf, "lambda_dice": ld, "lambda_edge": le,
                "status": "NO_METRICS",
                "elapsed_s": elapsed,
            })

    # ── Summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    print(f"{'Label':<28} {'mIoU':<8} {'Prec':<8} {'Recall':<8} {'Loss':<8} {'Epoch':<6}")
    print("-" * 70)
    best = None
    for r in results:
        if r["status"] == "OK":
            miou = r["best_mIoU"]
            print(f"{r['label']:<28} {miou:<8.4f} {r['best_precision']:<8.4f} {r['best_recall']:<8.4f} {r['best_val_loss']:<8.4f} {r['best_epoch']:<6}")
            if best is None or miou > best["best_mIoU"]:
                best = r
        else:
            print(f"{r['label']:<28} {r['status']}")

    if best:
        print("-" * 70)
        print(f"\nBEST: {best['label']}")
        print(f"  mIoU={best['best_mIoU']:.4f}, Precision={best['best_precision']:.4f}, Recall={best['best_recall']:.4f}")
        print(f"  lr={best['lr']}, wd={best['wd']}, lambda_focal={best['lambda_focal']}, lambda_dice={best['lambda_dice']}, lambda_edge={best['lambda_edge']}")
        print(f"  version_{best['version']}, epoch={best['best_epoch']}")

    # Save summary to file
    summary_path = Path("lightning_logs") / "grid_search_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Grid Search Summary\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"{r}\n")
    print(f"\nSummary saved to {summary_path}")
    print("Grid search complete.")


if __name__ == "__main__":
    main()
