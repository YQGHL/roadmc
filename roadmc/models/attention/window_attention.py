"""3D window attention for point cloud transformers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from roadmc.models.mhc.mhc import MHCConnection


def _window_partition(
    coords: torch.Tensor,
    window_size: int,
    shift: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Assign points to 3D windows based on normalized coordinates."""
    _, N, _ = coords.shape

    coords_min = coords.amin(dim=1, keepdim=True)
    coords_max = coords.amax(dim=1, keepdim=True)
    coords_range = coords_max - coords_min
    coords_range = torch.where(coords_range < 1e-6, torch.ones_like(coords_range), coords_range)
    coords_norm = (coords - coords_min) / coords_range

    grid_res = max(1, round((N / window_size) ** (1.0 / 3.0)))
    if shift:
        coords_norm = (coords_norm + 0.5 / grid_res) % 1.0

    bin_idx = (coords_norm * grid_res).long().clamp(0, grid_res - 1)
    window_id = (
        bin_idx[..., 0] * (grid_res * grid_res)
        + bin_idx[..., 1] * grid_res
        + bin_idx[..., 2]
    )
    return window_id, grid_res ** 3


def _window_attention_blockwise(
    coords: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_id: torch.Tensor,
    pos_mlp: nn.Module,
    softmax: nn.Softmax,
) -> torch.Tensor:
    """Compute attention independently per window to avoid global N x N tensors."""
    B, H, _, D = q.shape
    out = torch.zeros_like(q)
    scale = D ** -0.5

    for b in range(B):
        ids = window_id[b]
        for wid in torch.unique(ids):
            idx = torch.nonzero(ids == wid, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            qb = q[b, :, idx, :]
            kb = k[b, :, idx, :]
            vb = v[b, :, idx, :]
            local_coords = coords[b, idx]
            offsets = local_coords.unsqueeze(1) - local_coords.unsqueeze(0)
            rel_pos_bias = pos_mlp(offsets).permute(2, 0, 1)  # (H, M, M)

            attn = (qb @ kb.transpose(-2, -1)) * scale
            attn = softmax(attn + rel_pos_bias.unsqueeze(0))
            out[b, :, idx, :] = attn @ vb

    return out


class WindowAttention3D(nn.Module):
    """3D window attention with MLP-learned relative position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        pos_hidden = int(dim // mlp_ratio)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, num_heads),
        )

    def forward(
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        shift: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        window_id, _ = _window_partition(coords, self.window_size, shift=shift)
        out = _window_attention_blockwise(
            coords=coords,
            q=q,
            k=k,
            v=v,
            window_id=window_id,
            pos_mlp=self.pos_mlp,
            softmax=self.softmax,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DeformableWindowAttention3D(nn.Module):
    """Deformable 3D attention using learned offsets and nearest-key lookup."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        num_sample_points: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        offset_init_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_sample_points = num_sample_points
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, f"dim {dim} not divisible by {num_heads}"

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.offset_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3 * num_sample_points),
        )
        self.offset_scale = offset_init_scale
        pos_hidden = int(dim // mlp_ratio)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, num_heads),
        )

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D, K = self.num_heads, self.head_dim, self.num_sample_points

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        offsets = self.offset_net(x).view(B, N, K, 3)
        ref_points = coords.unsqueeze(2)
        sample_points = ref_points + offsets * self.offset_scale

        sampled_k = []
        sampled_v = []
        for b in range(B):
            sp = sample_points[b]
            kc = coords[b]
            dist = torch.cdist(sp.reshape(-1, 3), kc)
            _, nn_idx = dist.min(dim=-1)
            nn_idx = nn_idx.view(N, K)
            sampled_k.append(k[b][:, nn_idx])
            sampled_v.append(v[b][:, nn_idx])

        sampled_k = torch.stack(sampled_k)
        sampled_v = torch.stack(sampled_v)

        scale = D ** -0.5
        attn = (q.unsqueeze(3) * sampled_k).sum(dim=-1) * scale
        rel_pos_bias = self.pos_mlp(offsets).permute(0, 3, 1, 2)
        attn = self.softmax(attn + rel_pos_bias)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ShiftedWindowTransformerBlock(nn.Module):
    """Transformer block with window attention and MHC connection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        shift: bool = False,
        use_mhc: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.shift = shift
        self.use_mhc = use_mhc

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size, mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.mhc = MHCConnection(dim) if use_mhc else None

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_attn = self.attn(coords, self.norm1(x), shift=self.shift) + x
        x_ffn = self.ffn(self.norm2(x_attn)) + x_attn
        if self.mhc is not None and self.use_mhc:
            return self.mhc(x_ffn.reshape(-1, C), x_attn.reshape(-1, C)).reshape(B, N, C)
        return x_ffn


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, C = 2, 512, 96
    coords = torch.rand(B, N, 3) * 100
    feats = torch.rand(B, N, C)

    attn = WindowAttention3D(dim=C, num_heads=4, window_size=50)
    out = attn(coords, feats)
    assert out.shape == (B, N, C)
    assert not torch.isnan(out).any()

    def_attn = DeformableWindowAttention3D(dim=C, num_heads=4, num_sample_points=8)
    out_def = def_attn(coords, feats)
    assert out_def.shape == (B, N, C)
    assert not torch.isnan(out_def).any()

    block = ShiftedWindowTransformerBlock(dim=C, num_heads=4, window_size=50, use_mhc=True)
    out2 = block(coords, feats)
    assert out2.shape == (B, N, C)
    assert not torch.isnan(out2).any()

    block2 = ShiftedWindowTransformerBlock(dim=C, num_heads=4, window_size=50, use_mhc=False)
    out3 = block2(coords, feats)
    assert out3.shape == (B, N, C)
    print("window_attention self-test passed")
