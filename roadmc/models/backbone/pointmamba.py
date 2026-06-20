"""PointMamba-inspired backbone for road pavement disease segmentation."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from roadmc.models.attention.window_attention import MHCConnection
from roadmc.models.backbone.swin3d import SegmentationHead


def _expand_bits(v: torch.Tensor) -> torch.Tensor:
    """Expand 10-bit integers for Morton code construction."""
    v = v & 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _morton_permutation(coords: torch.Tensor, levels: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort points by Morton (z-order) code."""
    coords_min = coords.amin(dim=1, keepdim=True)
    coords_max = coords.amax(dim=1, keepdim=True)
    coords_range = torch.where(
        (coords_max - coords_min) < 1e-6,
        torch.ones_like(coords_max - coords_min),
        coords_max - coords_min,
    )
    coords_norm = (coords - coords_min) / coords_range
    bins = (coords_norm * ((1 << levels) - 1)).long().clamp(0, (1 << levels) - 1)

    x = _expand_bits(bins[..., 0])
    y = _expand_bits(bins[..., 1])
    z = _expand_bits(bins[..., 2])
    morton = x | (y << 1) | (z << 2)

    perm = morton.argsort(dim=1, stable=True)
    inv_perm = perm.argsort(dim=1)
    return perm, inv_perm


def _gather_along_points(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    idx = perm.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return x.gather(1, idx)


class PointMambaBlock(nn.Module):
    """State-space style mixer with causal scan and local convolution."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, use_mhc: bool = True):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.decay = nn.Parameter(torch.zeros(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.mhc = MHCConnection(dim) if use_mhc else None

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        alpha = 0.05 + 0.9 * torch.sigmoid(self.decay).view(1, 1, -1)
        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            state = x[:, t, :] + alpha.squeeze(1) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del coords
        residual = x
        x = self.norm1(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = self._scan(u)
        u = self.out_proj(F.silu(u) * torch.sigmoid(gate))
        x_scan = residual + u
        x = x_scan + self.ffn(self.norm2(x_scan))
        if self.mhc is not None:
            B, N, C = x.shape
            x = self.mhc(x.reshape(-1, C), x_scan.reshape(-1, C)).reshape(B, N, C)
        return x


class PointMambaStage(nn.Module):
    """One point sequence stage with Morton ordering."""

    def __init__(
        self,
        blocks: List[PointMambaBlock],
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _run_block(self, block: PointMambaBlock, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, coords, x, use_reentrant=False)
        return block(coords, x)

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        perm, inv_perm = _morton_permutation(coords)
        coords_ord = _gather_along_points(coords, perm)
        x_ord = _gather_along_points(x, perm)

        for block in self.blocks:
            x_ord = self._run_block(block, coords_ord, x_ord)

        skip = _gather_along_points(x_ord, inv_perm)
        x = self.downsample(skip)
        return x, skip


class PointMambaBackbone(nn.Module):
    """PointMamba-inspired backbone with Morton ordering and scan mixers."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 38,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 64,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_mhc: bool = True,
    ):
        super().__init__()
        del num_heads, window_size

        channels = [embed_dim * (2 ** i) for i in range(4)]
        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels + 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks: List[PointMambaBlock] = []
            for _ in range(depths[i]):
                blocks.append(PointMambaBlock(channels[i], mlp_ratio=mlp_ratio, use_mhc=use_mhc))

            downsample: Optional[nn.Module]
            if i < 3:
                downsample = nn.Linear(channels[i], channels[i + 1])
            else:
                downsample = None

            self.stages.append(PointMambaStage(blocks, downsample, use_checkpoint=use_checkpoint))

        self.decode = SegmentationHead(channels, num_classes)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([coords, feats], dim=-1)
        x = self.patch_embed(x)

        skip_features: List[torch.Tensor] = []
        for stage in self.stages:
            x, skip = stage(coords, x)
            skip_features.append(skip)

        return self.decode(skip_features, coords)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N = 2, 256
    coords = torch.rand(B, N, 3)
    feats = torch.rand(B, N, 3)

    model = PointMambaBackbone(in_channels=3, num_classes=38, embed_dim=48, depths=(1, 1, 2, 1))
    logits = model(coords, feats)
    assert logits.shape == (B, N, 38)
    assert torch.isfinite(logits).all()
    print(f"PointMambaBackbone: output={logits.shape}, params={sum(p.numel() for p in model.parameters())}")
