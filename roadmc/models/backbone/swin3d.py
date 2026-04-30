"""Swin3D point cloud backbone for road pavement disease segmentation.

4-stage hierarchical Transformer with 3D window attention.
No spatial coarsening — channel projection only between stages.
FCN decoder with skip connections for per-point segmentation.

Input:  coords (B, N, 3), feats (B, N, in_channels)
Output: (B, N, num_classes) per-point logits
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from roadmc.models.attention.window_attention import ShiftedWindowTransformerBlock


class Stage(nn.Module):
    """One Swin3D stage: N shifted-window Transformer blocks + optional downsampling.

    Even-indexed blocks use shifted-window attention (shift=True);
    odd-indexed blocks use regular window attention (shift=False).

    The skip output is captured BEFORE the downsample projection so
    the segmentation head receives features at the original channel
    dimension of each stage.
    """

    def __init__(
        self,
        blocks: List[ShiftedWindowTransformerBlock],
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample if downsample is not None else nn.Identity()

    def forward(
        self, coords: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            coords: (B, N, 3) point coordinates.
            x: (B, N, C_in) input features.

        Returns:
            x:   (B, N, C_out) features after blocks + downsample.
            skip: (B, N, C_in)  features after blocks, before downsample.
        """
        for block in self.blocks:
            x = block(coords, x)
        skip = x
        x = self.downsample(x)
        return x, skip


class SegmentationHead(nn.Module):
    """FCN decoder with skip connections for per-point segmentation.

    Fuses multi-scale features from each stage via channel projection
    and concatenation.  Since all stages operate on the same N points
    (no spatial coarsening), we only need channel manipulation — no
    upsampling layers.

    Decode chain (channels [C0, C1, C2, C3])::

        f3 (C3) → Linear(C3→C2) → merge with f2 (C2) → 2×C2
        → Linear(2×C2→C1) → merge with f1 (C1) → 2×C1
        → Linear(2×C1→C0) → merge with f0 (C0) → 2×C0
        → Linear(2×C0→C0) → Linear(C0→num_classes)
    """

    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.merge3 = nn.Sequential(
            nn.Linear(c3, c2),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Linear(c2 * 2, c1),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Linear(c1 * 2, c0),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Linear(c0 * 2, c0),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(c0, num_classes)

    def forward(
        self, features: List[torch.Tensor], coords: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: [stage0, stage1, stage2, stage3] features
                      at channel dims [C0, C1, C2, C3].
            coords:   (B, N, 3) point coordinates (reserved for future
                      position-aware decoding; currently unused).

        Returns:
            (B, N, num_classes) per-point logits.
        """
        f0, f1, f2, f3 = features

        # Stage 3 → Stage 2
        x = self.merge3(f3)              # C3 → C2
        x = torch.cat([x, f2], dim=-1)   # 2×C2
        x = self.fuse3(x)                # 2×C2 → C1

        # Stage 1
        x = torch.cat([x, f1], dim=-1)   # 2×C1
        x = self.fuse2(x)                # 2×C1 → C0

        # Stage 0
        x = torch.cat([x, f0], dim=-1)   # 2×C0
        x = self.fuse1(x)                # 2×C0 → C0

        # Classification
        x = self.cls_head(x)             # C0 → num_classes

        return x


class Swin3D(nn.Module):
    """Swin3D point cloud backbone for road pavement disease segmentation.

    4-stage hierarchical Transformer for 3D point clouds:

    1. **PatchEmbedding** — concat(coords, feats) → Linear → LayerNorm
    2. **Stages 0–3** — shifted-window Transformer blocks → channel projection
    3. **SegmentationHead** — FCN decoder with skip connections → per-point logits

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (default 3: intensity, curvature,
        crack-boundary distance).
    num_classes : int
        Number of JTG pavement disease classes (default 38).
    embed_dim : int
        Base embedding dimension for stage 0.  Doubles at each stage.
    depths : tuple of int
        Number of ``ShiftedWindowTransformerBlock`` blocks per stage.
        Default (2, 2, 6, 2) matches Swin-Tiny.
    num_heads : tuple of int
        Number of attention heads per stage.
    window_size : int
        Target number of points per 3D window.
    mlp_ratio : float
        Expansion ratio for FFN hidden dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 38,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # Channels per stage: [C0, C1, C2, C3]
        channels = [embed_dim * (2 ** i) for i in range(4)]

        # ── Patch Embedding ──────────────────────────────────────────
        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels + 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Stages ───────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks: List[ShiftedWindowTransformerBlock] = []
            for j in range(depths[i]):
                shift = (j % 2 == 0)  # even → shifted window, odd → regular
                blocks.append(
                    ShiftedWindowTransformerBlock(
                        dim=channels[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        shift=shift,
                    )
                )

            # Channel projection (Identity for the last stage)
            downsample: Optional[nn.Module]
            if i < 3:
                downsample = nn.Linear(channels[i], channels[i + 1])
            else:
                downsample = None

            self.stages.append(Stage(blocks, downsample))

        # ── Segmentation Head ────────────────────────────────────────
        self.decode = SegmentationHead(channels, num_classes)

    def forward(
        self, coords: torch.Tensor, feats: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: (B, N, 3) point coordinates.
            feats:  (B, N, in_channels) input features
                    (e.g. [intensity, curvature, crack_boundary_dist]).

        Returns:
            (B, N, num_classes) per-point logits.
        """
        x = torch.cat([coords, feats], dim=-1)   # (B, N, 3 + in_channels)
        x = self.patch_embed(x)                   # (B, N, C0)

        skip_features: List[torch.Tensor] = []
        for stage in self.stages:
            x, skip = stage(coords, x)
            skip_features.append(skip)

        logits = self.decode(skip_features, coords)

        return logits


# Self-test
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")
    B, N = 2, 512

    coords = torch.rand(B, N, 3, device=device)
    feats = torch.rand(B, N, 3, device=device)  # [intensity, curvature, crack_boundary_dist]

    # Tiny config for fast CPU test
    model = Swin3D(
        in_channels=3,
        num_classes=38,
        embed_dim=48,
        depths=(1, 1, 2, 1),
        num_heads=(2, 4, 8, 16),
        window_size=64,
    )

    logits = model(coords, feats)

    # Test 1: output shape
    assert logits.shape == (B, N, 38), (
        f"Expected ({B}, {N}, 38), got {logits.shape}"
    )

    # Test 2: no NaN
    assert not torch.isnan(logits).any(), "NaN in output"

    # Test 3: finite gradients (norm2 is now used in updated ShiftedWindowTransformerBlock)
    loss = logits.sum()
    loss.backward()
    params_no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_no_grad, (
        f"Parameters without gradient: {params_no_grad}"
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[PASS] Swin3D: output={logits.shape}, params={param_count}")

    # Test 4: forward pass consistency (deterministic)
    logits2 = model(coords, feats)
    assert torch.allclose(logits, logits2, atol=1e-5), "Non-deterministic output"
    print("[PASS] Swin3D: deterministic forward pass")
