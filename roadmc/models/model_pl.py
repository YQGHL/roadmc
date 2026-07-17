"""PyTorch Lightning wrapper for RoadMC segmentation.

Combines FocalLoss + DiceLoss + EdgeLoss for training
Swin3D backbone with macro mIoU evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple

from roadmc.metrics import (
    CalibrationAccumulator,
    bootstrap_scene_confidence_intervals,
    confusion_matrix_from_predictions,
    metrics_from_confusion,
)
from roadmc.data.features import OBSERVABLE_FEATURE_SCHEMA, require_observable_checkpoint_schema


class HybridMuonAdamW(torch.optim.Optimizer):
    """Wrap Muon and AdamW so matrix params can use Muon while 1D params use AdamW."""

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer):
        dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        super().__init__([dummy], {})
        self.muon = muon
        self.adamw = adamw
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


class FocalLoss(nn.Module):
    """Focal Loss: -α(1-p_t)^γ log(p_t)

    Used for handling class imbalance (some disease classes are rare).
    γ=2 focuses training on hard, misclassified examples.
    α = class_weights (inverse frequency) to balance class distribution.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute focal loss with optional valid_mask masking. -1 target values are ignored.
        """
        if valid_mask is not None:
            logits = logits[valid_mask]
            targets = targets[valid_mask]
            if targets.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        probs = F.softmax(logits, dim=-1)  # (BxN, C) after masking

        # gather p_t = p[target_class] for each point
        # targets: (B, N) → (B, N, 1) for gather
        targets_expanded = targets.unsqueeze(-1)  # (B, N, 1)
        p_t = probs.gather(dim=-1, index=targets_expanded).squeeze(-1)  # (B, N)

        focal_weight = (1.0 - p_t) ** self.gamma  # (B, N)

        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)  # (B, N)

        if self.alpha is not None:
            alpha_weights = self.alpha.to(logits.device)[targets]  # (B, N)
            focal_loss = alpha_weights * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Multi-class Dice Loss: 1 - 2*|P∩T| / (|P|+|T|)

    Smooth variant with ε=1 to prevent division by zero.
    Computed per-class then averaged.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute dice loss (vectorized, L2 fix) with optional valid_mask masking. -1 target values are ignored.
        """
        if valid_mask is not None:
            logits = logits[valid_mask]
            targets = targets[valid_mask]
            if targets.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)  # (M, C) after masking

        targets_one_hot = F.one_hot(targets, num_classes).float()  # (M, C)

        # L2 fix: vectorized per-class dice (no Python loop)
        # intersection: (C,) sum over B,N
        intersection = (probs * targets_one_hot).sum(dim=0)  # (C,)
        # union: (C,)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)  # (C,)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (C,)
        dice_loss = 1.0 - dice  # (C,)

        return dice_loss.mean()


class EdgeLoss(nn.Module):
    """Supervised BEV boundary loss without label-derived model inputs.

    Predictions and ground truth are rasterized from all valid points.  The
    target labels are used only as the ordinary training target, while a
    dilated target-edge map increases the weight around road-damage boundaries.
    This remains available for every curriculum stage and for every damage
    family; no crack-only privileged distance is involved.
    """

    def __init__(
        self,
        grid_size: int = 200,
        boundary_dilation: int = 2,
        boundary_weight: float = 2.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.boundary_dilation = boundary_dilation
        self.boundary_weight = boundary_weight

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _scatter_to_bev(
        self, values: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """M3+M4: Scatter per-point values to 2D BEV grid with mean aggregation + aspect ratio preservation.

        Uses ``scatter_add_`` + count tracking for per-pixel mean (not "last write wins").
        Preserves physical aspect ratio by computing grid dimensions from coordinate ranges.
        """
        gs = self.grid_size
        x_phys = coords[:, 0]
        y_phys = coords[:, 1]

        x_min, x_max = x_phys.min(), x_phys.max()
        y_min, y_max = y_phys.min(), y_phys.max()

        if (x_max - x_min) < 1e-6:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if (y_max - y_min) < 1e-6:
            y_min, y_max = y_min - 0.5, y_max + 0.5

        # M4: preserve aspect ratio — scale x,y to same physical resolution
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range > y_range:
            gs_y = max(1, int(gs * y_range / x_range))
            gs_x = gs
        else:
            gs_x = max(1, int(gs * x_range / y_range))
            gs_y = gs

        xi = ((x_phys - x_min) / (x_max - x_min) * (gs_x - 1)).long().clamp(0, gs_x - 1)
        yi = ((y_phys - y_min) / (y_max - y_min) * (gs_y - 1)).long().clamp(0, gs_y - 1)

        # M3: mean aggregation via scatter_add_ + count tracking
        grid = torch.zeros(gs_y, gs_x, device=values.device, dtype=torch.float64)
        cnt = torch.zeros(gs_y, gs_x, device=values.device, dtype=torch.int64)
        grid.index_put_((yi, xi), values.double(), accumulate=True)
        cnt.index_put_((yi, xi), torch.ones_like(values, dtype=torch.int64), accumulate=True)
        cnt = cnt.clamp(min=1)
        grid = (grid / cnt.float()).float()

        # Resize to (gs, gs) for Sobel conv
        if gs_y != gs or gs_x != gs:
            grid = grid.unsqueeze(0).unsqueeze(0)
            grid = F.interpolate(grid, size=(gs, gs), mode='bilinear', align_corners=False)
            grid = grid.squeeze(0).squeeze(0)

        return grid

    def _sobel_edge(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filter to a 2D grid.
        """
        grid_4d = grid.unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(grid_4d, self.sobel_x, padding=1)
        gy = F.conv2d(grid_4d, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        return edge.squeeze(0).squeeze(0)  # (H, W)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a differentiable damage-boundary loss for valid point sets."""
        B = logits.shape[0]

        if valid_mask is not None:
            logits_list = [logits[b][valid_mask[b]] for b in range(B)]
            targets_list = [targets[b][valid_mask[b]] for b in range(B)]
            coords_list = [coords[b][valid_mask[b]] for b in range(B)]
        else:
            logits_list = [logits[b] for b in range(B)]
            targets_list = [targets[b] for b in range(B)]
            coords_list = [coords[b] for b in range(B)]

        losses = []

        for b in range(B):
            if len(coords_list[b]) < 4:
                continue

            probabilities = F.softmax(logits_list[b], dim=-1)
            damage_probability = (1.0 - probabilities[..., 0]).float()
            damage_target = (targets_list[b] != 0).float()
            pred_grid = self._scatter_to_bev(damage_probability, coords_list[b])
            target_grid = self._scatter_to_bev(damage_target, coords_list[b])

            pred_edge = self._sobel_edge(pred_grid)
            target_edge = self._sobel_edge(target_grid)
            boundary = (target_edge > 1e-3).float().unsqueeze(0).unsqueeze(0)
            if self.boundary_dilation > 0:
                kernel = 2 * self.boundary_dilation + 1
                boundary = F.max_pool2d(boundary, kernel_size=kernel, stride=1, padding=self.boundary_dilation)
            weights = 1.0 + self.boundary_weight * boundary.squeeze(0).squeeze(0)
            losses.append(((pred_edge - target_edge).abs() * weights).sum() / weights.sum().clamp_min(1.0))

        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()


class RoadMCSegModel(pl.LightningModule):
    """RoadMC segmentation model with PyTorch Lightning wrapper.

    Combines FocalLoss + DiceLoss + EdgeLoss for training
    Swin3D backbone with macro mIoU evaluation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 38,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        num_heads: tuple = (3, 6, 12, 24),
        window_size: int = 64,
        mlp_ratio: float = 4.0,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        optimizer_name: str = "muon",
        backbone_name: str = "swin3d",
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_edge: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        crack_dist_feat_idx: Optional[int] = None,
        feature_schema: str = OBSERVABLE_FEATURE_SCHEMA,
        input_point_count: Optional[int] = None,
        t_max: int = 50,  # Cosine annealing period
        use_checkpoint: bool = False,  # gradient checkpointing to reduce VRAM
        use_mhc: bool = True,
        binary_threshold: float = 0.5,
        metric_min_support: int = 1,
        validation_bootstrap_samples: int = 0,
        validation_bootstrap_seed: int = 42,
    ):
        super().__init__()
        if feature_schema != OBSERVABLE_FEATURE_SCHEMA:
            raise ValueError(
                f"RoadMCSegModel requires {OBSERVABLE_FEATURE_SCHEMA}, got {feature_schema!r}"
            )
        self.save_hyperparameters()

        backbone_name = backbone_name.lower()
        if backbone_name == "swin3d":
            from roadmc.models.backbone.swin3d import Swin3D
            backbone_cls = Swin3D
        elif backbone_name == "pointmamba":
            from roadmc.models.backbone.pointmamba import PointMambaBackbone
            backbone_cls = PointMambaBackbone
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone = backbone_cls(
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint,
            use_mhc=use_mhc,
        )

        self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()

        self.num_classes = num_classes
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_edge = lambda_edge
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.backbone_name = backbone_name
        # Retain this legacy argument solely so checkpoints created before the
        # observable-feature migration can still be loaded.  It is ignored.
        self.crack_dist_feat_idx = crack_dist_feat_idx
        self.feature_schema = feature_schema
        self.input_point_count = input_point_count
        self.t_max = t_max  # L2: now settable via constructor
        self.binary_threshold = binary_threshold
        self.metric_min_support = metric_min_support
        self.validation_bootstrap_samples = validation_bootstrap_samples
        self.validation_bootstrap_seed = validation_bootstrap_seed
        self.register_buffer(
            "_val_confusion", torch.zeros(num_classes, num_classes, dtype=torch.long),
            persistent=False,
        )
        self._val_scene_confusions = []
        self._val_calibration = CalibrationAccumulator()
        self.last_validation_summary = {}

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.
        """
        return self.backbone(coords, feats)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist the input contract outside hyperparameters for independent audit."""

        checkpoint["roadmc_feature_schema"] = self.feature_schema

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Prevent accidental reuse of a checkpoint trained with privileged inputs."""

        require_observable_checkpoint_schema(checkpoint, context="checkpoint")

    def training_step(self, batch, batch_idx):
        """Training step. Accepts both dict batches (DataLoader) and tuple batches (self-test).
        """
        if isinstance(batch, dict):
            coords, feats, labels = batch["coords"], batch["feats"], batch["labels"]
            valid_mask = batch.get("valid_mask")
        else:
            coords, feats, labels = batch
            valid_mask = None

        logits = self(coords, feats)

        fl = self.focal_loss(logits, labels, valid_mask)
        dl = self.dice_loss(logits, labels, valid_mask)
        el = self.edge_loss(logits, labels, coords, valid_mask)

        loss = self.lambda_focal * fl + self.lambda_dice * dl + self.lambda_edge * el

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_focal", fl, prog_bar=False)
        self.log("train_dice", dl, prog_bar=False)
        self.log("train_edge", el, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with mIoU computation. Accepts both dict and tuple batches.
        """
        if isinstance(batch, dict):
            coords, feats, labels = batch["coords"], batch["feats"], batch["labels"]
            valid_mask = batch.get("valid_mask")
        else:
            coords, feats, labels = batch
            valid_mask = None

        logits = self(coords, feats)

        fl = self.focal_loss(logits, labels, valid_mask)
        dl = self.dice_loss(logits, labels, valid_mask)
        el = self.edge_loss(logits, labels, coords, valid_mask)

        loss = self.lambda_focal * fl + self.lambda_dice * dl + self.lambda_edge * el

        preds = self._prediction_from_logits(logits)
        batch_confusion = confusion_matrix_from_predictions(
            preds, labels, self.num_classes, valid_mask
        )
        self._val_confusion += batch_confusion.to(self._val_confusion.device)

        probabilities = torch.softmax(logits, dim=-1)
        for scene_index in range(preds.shape[0]):
            scene_mask = valid_mask[scene_index] if valid_mask is not None else None
            scene_confusion = confusion_matrix_from_predictions(
                preds[scene_index], labels[scene_index], self.num_classes, scene_mask
            )
            self._val_scene_confusions.append(scene_confusion.detach().cpu())
            self._val_calibration.update(
                probabilities[scene_index], labels[scene_index], scene_mask
            )

        # Retain the batch figure only as a diagnostic.  Checkpoints use the
        # epoch-global val_mIoU logged in on_validation_epoch_end.
        batch_metrics = metrics_from_confusion(
            batch_confusion, min_support=self.metric_min_support
        )
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_batch_mIoU", batch_metrics["supported_miou"], prog_bar=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_confusion.zero_()
        self._val_scene_confusions = []
        self._val_calibration = CalibrationAccumulator()

    def on_validation_epoch_end(self) -> None:
        """Log one statistically valid metric from the complete validation set."""
        summary = metrics_from_confusion(
            self._val_confusion, min_support=self.metric_min_support
        )
        calibration = self._val_calibration.as_dict()
        self.last_validation_summary = {
            "metrics": summary,
            "calibration": calibration,
        }
        self.log("val_mIoU", summary["supported_miou"], prog_bar=True, sync_dist=True)
        self.log("val_foreground_iou", summary["foreground_iou"], prog_bar=True, sync_dist=True)
        self.log("val_ece", calibration["ece"], prog_bar=False, sync_dist=True)
        self.log("val_brier", calibration["brier"], prog_bar=False, sync_dist=True)

        for item in summary["per_class"]:
            class_id = item["class_id"]
            if class_id == 0 or not item["supported"]:
                continue
            self.log(f"val_IoU_{class_id}", item["iou"], prog_bar=False, sync_dist=True)
            self.log(f"val_recall_{class_id}", item["recall"], prog_bar=False, sync_dist=True)
            self.log(
                f"val_precision_{class_id}", item["precision"], prog_bar=False, sync_dist=True
            )

        if self.validation_bootstrap_samples > 0:
            bootstrap = bootstrap_scene_confidence_intervals(
                self._val_scene_confusions,
                n_bootstrap=self.validation_bootstrap_samples,
                seed=self.validation_bootstrap_seed,
                min_support=self.metric_min_support,
            )
            self.last_validation_summary["bootstrap"] = bootstrap

    def _prediction_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply calibrated binary threshold when requested, otherwise argmax."""
        if self.num_classes == 2 and self.binary_threshold != 0.5:
            disease_prob = torch.softmax(logits, dim=-1)[..., 1]
            return (disease_prob >= self.binary_threshold).long()
        return logits.argmax(dim=-1)

    def configure_optimizers(self):
        """Muon optimizer with grouped learning rates and cosine annealing."""
        optimizer, scheduler = self.build_optimizer_and_scheduler()
        return [optimizer], [scheduler]

    def build_optimizer_and_scheduler(self, optimizer_name: Optional[str] = None):
        """Build the optimizer and scheduler used by both Lightning and manual training."""
        optimizer_name = (optimizer_name or self.optimizer_name).lower()
        matrix_params = []
        head_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "decode.cls_head" in name and param.ndim == 2:
                head_params.append(param)
            elif param.ndim == 2:
                matrix_params.append(param)
            else:
                adamw_params.append(param)

        if optimizer_name == "muon":
            muon_groups = []
            if matrix_params:
                muon_groups.append(
                    {"params": matrix_params, "lr": self.lr, "weight_decay": self.weight_decay}
                )
            if head_params:
                muon_groups.append(
                    {"params": head_params, "lr": self.lr * 3.0, "weight_decay": self.weight_decay * 0.5}
                )

            adamw_groups = []
            if adamw_params:
                adamw_groups.append(
                    {"params": adamw_params, "lr": self.lr, "weight_decay": 0.0}
                )

            muon = torch.optim.Muon(
                muon_groups,
                momentum=0.95,
                nesterov=True,
            )
            if adamw_groups:
                adamw = torch.optim.AdamW(
                    adamw_groups,
                    lr=self.lr,
                    weight_decay=0.0,
                )
                optimizer = HybridMuonAdamW(muon, adamw)
            else:
                optimizer = muon
        elif optimizer_name == "adamw":
            param_groups = []
            if matrix_params:
                param_groups.append(
                    {"params": matrix_params, "lr": self.lr, "weight_decay": self.weight_decay}
                )
            if head_params:
                param_groups.append(
                    {"params": head_params, "lr": self.lr * 3.0, "weight_decay": self.weight_decay * 0.5}
                )
            if adamw_params:
                param_groups.append(
                    {"params": adamw_params, "lr": self.lr, "weight_decay": 0.0}
                )
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
        # T_max can be set via init param or defaults to 50
        t_max = getattr(self, 't_max', 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return optimizer, scheduler

    @staticmethod
    def compute_miou(
        preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute macro mean IoU + per-class IoU, recall, precision. -1 target entries ignored.
        """
        if valid_mask is not None:
            preds_flat = preds[valid_mask]
            targets_flat = targets[valid_mask]
        else:
            # Also filter -1 labels even without valid_mask
            valid_entries = targets >= 0
            preds_flat = preds[valid_entries]
            targets_flat = targets[valid_entries]

        if targets_flat.numel() == 0:
            return (
                torch.tensor(0.0, device=preds.device),
                torch.zeros(num_classes, device=preds.device),
                torch.zeros(num_classes, device=preds.device),
                torch.zeros(num_classes, device=preds.device),
            )

        ious = []
        recalls = []
        precisions = []

        for c in range(num_classes):
            pred_c = preds_flat == c
            target_c = targets_flat == c
            tp = (pred_c & target_c).sum().float()
            fp = (pred_c & ~target_c).sum().float()
            fn = (~pred_c & target_c).sum().float()

            denom_iou = tp + fp + fn + 1e-8
            iou_c = tp / denom_iou

            denom_recall = tp + fn + 1e-8
            recall_c = tp / denom_recall

            denom_precision = tp + fp + 1e-8
            precision_c = tp / denom_precision

            ious.append(iou_c)
            recalls.append(recall_c)
            precisions.append(precision_c)

        per_class_iou = torch.stack(ious)
        per_class_recall = torch.stack(recalls)
        per_class_precision = torch.stack(precisions)

        # macro mIoU: skip background to match evaluation/reporting convention
        if num_classes > 1:
            miou = per_class_iou[1:].mean()
        else:
            miou = per_class_iou.mean()

        return miou, per_class_iou, per_class_recall, per_class_precision


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    torch.manual_seed(42)
    B, N = 2, 256

    coords = torch.rand(B, N, 3)
    feats = torch.rand(B, N, 3)
    labels = torch.randint(0, 38, (B, N))

    # Test FocalLoss
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(B, N, 38)
    loss_f = fl(logits, labels)
    assert torch.isfinite(loss_f), "FocalLoss should be finite"
    print(f"  FocalLoss: {loss_f.item():.4f}")

    # Test DiceLoss
    dl = DiceLoss()
    loss_d = dl(logits, labels)
    assert torch.isfinite(loss_d), "DiceLoss should be finite"
    print(f"  DiceLoss: {loss_d.item():.4f}")

    # Test EdgeLoss (when no cracks, should be 0)
    el = EdgeLoss()
    loss_e = el(logits, labels, coords)
    assert torch.isfinite(loss_e), "EdgeLoss should be finite"
    print(f"  EdgeLoss: {loss_e.item():.4f}")

    # Test mIoU + per-class metrics
    preds = logits.argmax(dim=-1)
    miou, _, _, _ = RoadMCSegModel.compute_miou(preds, labels, 38)
    assert 0 <= miou <= 1, f"mIoU out of range: {miou}"
    print(f"  mIoU: {miou:.4f}, per-class metrics implemented")

    # Test LightningModule (small config)
    model = RoadMCSegModel(
        in_channels=3,
        num_classes=38,
        embed_dim=48,
        depths=(1, 1, 2, 1),
        num_heads=(2, 4, 8, 16),
    )
    batch = (coords, feats, labels)

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), f"Training loss NaN: {loss}"
    print(f"  Training step: loss={loss.item():.4f}")

    model.validation_step(batch, 0)
    print("  Validation step OK")

    # Test configure_optimizers
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) >= 1
    assert len(schedulers) >= 1
    print(f"  Optimizer configured: {type(optimizers[0]).__name__}")

    print(
        f"\nAll tests passed. Parameter count: {sum(p.numel() for p in model.parameters())}"
    )
