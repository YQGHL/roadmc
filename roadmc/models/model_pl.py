"""PyTorch Lightning wrapper for RoadMC segmentation.

Combines FocalLoss + DiceLoss + EdgeLoss for training
Swin3D backbone with macro mIoU evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
#  Loss Functions
# ═══════════════════════════════════════════════════════════════════════


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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, N, C) raw scores.
            targets: (B, N) integer labels in [0, C-1].

        Returns:
            Scalar focal loss.
        """
        # softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, N, C)

        # gather p_t = p[target_class] for each point
        # targets: (B, N) → (B, N, 1) for gather
        targets_expanded = targets.unsqueeze(-1)  # (B, N, 1)
        p_t = probs.gather(dim=-1, index=targets_expanded).squeeze(-1)  # (B, N)

        # focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma  # (B, N)

        # cross-entropy: -log(p_t)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)  # (B, N)

        # apply alpha weights per class
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss (vectorized, L2 fix).

        Args:
            logits: (B, N, C) raw scores.
            targets: (B, N) integer labels in [0, C-1].

        Returns:
            Scalar dice loss.
        """
        num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)  # (B, N, C)

        # one-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).float()  # (B, N, C)

        # L2 fix: vectorized per-class dice (no Python loop)
        # intersection: (C,) sum over B,N
        intersection = (probs * targets_one_hot).sum(dim=(0, 1))  # (C,)
        # union: (C,)
        union = probs.sum(dim=(0, 1)) + targets_one_hot.sum(dim=(0, 1))  # (C,)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (C,)
        dice_loss = 1.0 - dice  # (C,)

        return dice_loss.mean()


class EdgeLoss(nn.Module):
    """Edge-aware loss: Sobel edge detection on predicted vs target labels.

    Converts per-point labels to a 2D BEV (bird's eye view) grid,
    applies Sobel 3×3 kernel to detect edges, then computes L1 on edge magnitudes.
    Only computed where crack_boundary_dist < threshold in feats.

    This enforces sharper crack boundary predictions.
    """

    def __init__(self, grid_size: int = 200, sigma: float = 2.0):
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        self.threshold = sigma  # crack_boundary_dist threshold

        # Sobel kernels (3×3)
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

        Args:
            values: (N,) per-point scalar values.
            coords: (N, 3) point coordinates (x, y, z).

        Returns:
            (grid_size, grid_size) BEV grid.
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

        Args:
            grid: (H, W) 2D grid.

        Returns:
            (H, W) edge magnitude map.
        """
        # (1, 1, H, W) for conv2d
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
        crack_boundary_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute edge loss.

        Args:
            logits: (B, N, C) raw scores.
            targets: (B, N) integer labels.
            coords: (B, N, 3) point coordinates.
            crack_boundary_dist: optional (B, N) or (N,) from feats[:,:,2].

        Returns:
            Scalar edge loss (0 if no crack data).
        """
        B = logits.shape[0]
        preds = logits.argmax(dim=-1)  # (B, N)

        total_loss = 0.0
        count = 0

        for b in range(B):
            # Determine which points are near cracks
            if crack_boundary_dist is not None:
                cbd = crack_boundary_dist[b]  # (N,)
                near_crack = cbd < self.threshold
                if not near_crack.any():
                    continue
                mask_coords = coords[b, near_crack]  # (K, 3)
                mask_preds = preds[b, near_crack].float()  # (K,)
                mask_targets = targets[b, near_crack].float()  # (K,)
            else:
                # No crack info — skip this batch item
                continue

            # Scatter to BEV grids
            pred_grid = self._scatter_to_bev(mask_preds, mask_coords)
            target_grid = self._scatter_to_bev(mask_targets, mask_coords)

            # Sobel edge detection
            pred_edge = self._sobel_edge(pred_grid)
            target_edge = self._sobel_edge(target_grid)

            # L1 loss on edge magnitudes
            total_loss = total_loss + F.l1_loss(pred_edge, target_edge)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        return total_loss / count


# ═══════════════════════════════════════════════════════════════════════
#  Lightning Module
# ═══════════════════════════════════════════════════════════════════════


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
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_edge: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        crack_dist_feat_idx: int = 2,  # L4: index of crack_boundary_dist in feats
        t_max: int = 50,  # Cosine annealing period
    ):
        super().__init__()
        self.save_hyperparameters()

        from roadmc.models.backbone.swin3d import Swin3D

        self.backbone = Swin3D(
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
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
        self.crack_dist_feat_idx = crack_dist_feat_idx  # L4
        self.t_max = t_max  # L2: now settable via constructor

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.

        Args:
            coords: (B, N, 3) point coordinates.
            feats: (B, N, in_channels) input features.

        Returns:
            (B, N, num_classes) per-point logits.
        """
        return self.backbone(coords, feats)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: tuple of (coords, feats, labels).
            batch_idx: int.

        Returns:
            Loss tensor.
        """
        coords, feats, labels = batch
        logits = self(coords, feats)

        fl = self.focal_loss(logits, labels)
        dl = self.dice_loss(logits, labels)
        el = self.edge_loss(logits, labels, coords, feats[:, :, self.crack_dist_feat_idx])

        loss = self.lambda_focal * fl + self.lambda_dice * dl + self.lambda_edge * el

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_focal", fl, prog_bar=False)
        self.log("train_dice", dl, prog_bar=False)
        self.log("train_edge", el, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with mIoU computation."""
        coords, feats, labels = batch
        logits = self(coords, feats)

        fl = self.focal_loss(logits, labels)
        dl = self.dice_loss(logits, labels)
        el = self.edge_loss(logits, labels, coords, feats[:, :, self.crack_dist_feat_idx])

        loss = self.lambda_focal * fl + self.lambda_dice * dl + self.lambda_edge * el

        preds = logits.argmax(dim=-1)
        miou, per_class_iou, per_class_recall, per_class_precision = self.compute_miou(
            preds, labels, self.num_classes
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mIoU", miou, prog_bar=True)
        # M5: log per-class metrics (only non-background classes)
        for c in range(1, min(self.num_classes, 10)):  # log first 9 non-bg for readability
            if per_class_iou[c] > 0 or per_class_recall[c] > 0:
                self.log(f"val_IoU_{c}", per_class_iou[c])
                self.log(f"val_recall_{c}", per_class_recall[c])
                self.log(f"val_precision_{c}", per_class_precision[c])
        return loss

    def configure_optimizers(self):
        """AdamW optimizer with cosine annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # T_max can be set via init param or defaults to 50
        t_max = getattr(self, 't_max', 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return [optimizer], [scheduler]

    @staticmethod
    def compute_miou(
        preds: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute macro mean IoU + per-class IoU, recall, precision.

        Args:
            preds: (B, N) predicted labels.
            targets: (B, N) ground truth labels.
            num_classes: C.

        Returns:
            Tuple of (mIoU, per_class_IoU, per_class_recall, per_class_precision).
        """
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)

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

        # macro mIoU: mean over all classes (or skip background if min_class=1)
        miou = per_class_iou.mean()

        return miou, per_class_iou, per_class_recall, per_class_precision


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════
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
    print(f"[PASS] FocalLoss: {loss_f.item():.4f}")

    # Test DiceLoss
    dl = DiceLoss()
    loss_d = dl(logits, labels)
    assert torch.isfinite(loss_d), "DiceLoss should be finite"
    print(f"[PASS] DiceLoss: {loss_d.item():.4f}")

    # Test EdgeLoss (when no cracks, should be 0)
    el = EdgeLoss()
    loss_e = el(logits, labels, coords)
    assert torch.isfinite(loss_e), "EdgeLoss should be finite"
    print(f"[PASS] EdgeLoss: {loss_e.item():.4f}")

    # Test mIoU + per-class metrics
    preds = logits.argmax(dim=-1)
    miou, _, _, _ = RoadMCSegModel.compute_miou(preds, labels, 38)
    assert 0 <= miou <= 1, f"mIoU out of range: {miou}"
    print(f"[PASS] mIoU: {miou:.4f}, per-class metrics implemented")

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
    print(f"[PASS] Training step: loss={loss.item():.4f}")

    model.validation_step(batch, 0)
    print("[PASS] Validation step OK")

    # Test configure_optimizers
    optimizers = model.configure_optimizers()
    assert len(optimizers) >= 1
    print(f"[PASS] Optimizer configured: {type(optimizers[0]).__name__}")

    print(
        f"\nAll tests passed. Parameter count: {sum(p.numel() for p in model.parameters())}"
    )