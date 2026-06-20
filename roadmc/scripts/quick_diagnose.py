"""Quick RoadMC training diagnostics.

Runs a short CPU-friendly training loop and reports loss, mIoU, and prediction
class distribution. This intentionally avoids Lightning so it can run in minimal
environments.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_lightning_stub() -> None:
    if "pytorch_lightning" in sys.modules:
        return
    pl = types.ModuleType("pytorch_lightning")

    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    class LightningDataModule:
        pass

    pl.LightningModule = LightningModule
    pl.LightningDataModule = LightningDataModule
    sys.modules["pytorch_lightning"] = pl


def _parse_binary_weights(value: str | None) -> torch.Tensor | None:
    if value is None:
        return None
    weights = [float(x) for x in value.split(",")]
    if len(weights) != 2:
        raise ValueError("--binary_class_weights must be formatted as 'bg,disease'")
    return torch.tensor(weights, dtype=torch.float32)


def _load_class_weights(path: str | None, expected_classes: int) -> torch.Tensor | None:
    if path is None:
        return None
    weights_path = Path(path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Class weights file not found: {weights_path}")
    if weights_path.suffix == ".npy":
        import numpy as np

        weights = torch.tensor(np.load(weights_path), dtype=torch.float32)
    else:
        weights = torch.load(weights_path, weights_only=True).float()
    if weights.numel() != expected_classes:
        raise ValueError(
            f"Class weights length {weights.numel()} does not match num_classes={expected_classes}"
        )
    return weights


def _load_binary_checkpoint_into_model(model, ckpt_path: str) -> None:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    skip_patterns = ("decode.cls_head", "focal_loss.alpha")
    stripped = {k: v for k, v in ckpt.items() if not any(p in k for p in skip_patterns)}
    msg = model.load_state_dict(stripped, strict=False)
    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    print(
        f"[PRETRAINED] loaded={len(stripped)} missing={len(missing)} unexpected={len(unexpected)} "
        f"from {path}"
    )
    if missing:
        print(f"[PRETRAINED] missing keys sample={missing[:8]}")
    if unexpected:
        print(f"[PRETRAINED] unexpected keys sample={unexpected[:8]}")


def _eval(model, loader, max_batches: int):
    model.eval()
    miou_values = []
    pred_counts = {}
    label_counts = {}
    mean_prob = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            logits = model(batch["coords"], batch["feats"])
            preds = logits.argmax(dim=-1)
            valid_mask = batch["valid_mask"]
            miou, *_ = model.compute_miou(preds, batch["labels"], model.num_classes, valid_mask)
            miou_values.append(float(miou))

            flat_preds = preds[valid_mask]
            flat_labels = batch["labels"][valid_mask]
            for cls, count in zip(*torch.unique(flat_preds, return_counts=True)):
                pred_counts[int(cls)] = pred_counts.get(int(cls), 0) + int(count)
            for cls, count in zip(*torch.unique(flat_labels, return_counts=True)):
                label_counts[int(cls)] = label_counts.get(int(cls), 0) + int(count)

            probs = torch.softmax(logits, dim=-1)[valid_mask]
            batch_mean = probs.mean(dim=0)
            mean_prob = batch_mean if mean_prob is None else mean_prob + batch_mean

    model.train()
    if mean_prob is not None:
        mean_prob = (mean_prob / max(1, len(miou_values))).tolist()
    return {
        "miou": sum(miou_values) / len(miou_values) if miou_values else float("nan"),
        "pred_counts": pred_counts,
        "label_counts": label_counts,
        "mean_prob": mean_prob,
    }


def main() -> None:
    _ensure_lightning_stub()

    from roadmc.data.dataloader import RoadMCDataModule
    from roadmc.models.model_pl import RoadMCSegModel

    parser = argparse.ArgumentParser(description="Quick RoadMC diagnostic run")
    parser.add_argument("--data_dir", default="data/synthetic_output")
    parser.add_argument("--backbone", default="pointmamba", choices=["swin3d", "pointmamba"])
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--binary_class_weights", default=None)
    parser.add_argument("--class_weights", default=None)
    parser.add_argument("--pretrained_binary", default=None)
    parser.add_argument("--save_checkpoint", default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_points", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mhc", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    num_classes = 2 if args.binary else 38
    if args.binary:
        class_weights = _parse_binary_weights(args.binary_class_weights)
    else:
        class_weights = _load_class_weights(args.class_weights, num_classes)

    dm = RoadMCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_points=args.max_points,
        num_workers=0,
        binary=args.binary,
    )
    dm.setup("fit")

    model = RoadMCSegModel(
        backbone_name=args.backbone,
        use_mhc=not args.no_mhc,
        optimizer_name="muon",
        in_channels=3,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depths=(1, 1, 1, 1),
        num_heads=(2, 2, 4, 4),
        window_size=16,
        lr=args.lr,
        weight_decay=0.05,
        use_checkpoint=False,
        t_max=args.t_max or args.steps,
        class_weights=class_weights,
    )
    if args.pretrained_binary:
        _load_binary_checkpoint_into_model(model, args.pretrained_binary)
    if args.freeze_backbone:
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if "backbone.decode.cls_head" in name:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        print(f"[FREEZE] backbone frozen params={frozen} trainable_params={trainable}")
    model.train()
    optimizer, scheduler = model.build_optimizer_and_scheduler("muon")

    losses = []
    start = time.time()
    for step, batch in enumerate(itertools.cycle(dm.train_dataloader())):
        if step >= args.steps:
            break
        optimizer.zero_grad()
        loss = model.training_step(batch, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))

        if step in {0, args.steps // 2, args.steps - 1}:
            metrics = _eval(model, dm.val_dataloader(), args.eval_batches)
            print(
                f"step={step} loss={losses[-1]:.4f} val_mIoU={metrics['miou']:.4f} "
                f"pred={metrics['pred_counts']} labels={metrics['label_counts']}"
            )

    metrics = _eval(model, dm.val_dataloader(), args.eval_batches)
    elapsed = time.time() - start
    print(
        f"done backbone={args.backbone} binary={args.binary} steps={len(losses)} "
        f"avg_loss={sum(losses)/len(losses):.4f} "
        f"first25={sum(losses[:25])/min(25, len(losses)):.4f} "
        f"last25={sum(losses[-25:])/min(25, len(losses)):.4f} "
        f"val_mIoU={metrics['miou']:.4f} elapsed={elapsed:.2f}s"
    )
    print(f"pred_counts={metrics['pred_counts']}")
    print(f"label_counts={metrics['label_counts']}")
    print(f"mean_prob={metrics['mean_prob']}")
    if args.save_checkpoint:
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, ckpt_path)
        print(f"saved_checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
