"""Gated-EMA point sequence mixer (Mamba-inspired) for pavement segmentation.

诚实命名口径（2026-07 审计）：本骨干**不是** PointMamba (Liang et al.
2024) 复现，也不是选择性 SSM (S6)——扫描核心是每通道门控指数滑动
平均（对角 LTI 递推，状态维数 1），论文表述应为 *"a Mamba-inspired
linear-memory sequence mixer: points are serialized by Morton order and
processed by a gated, per-channel exponential moving average"*。被实测
支持的声明只有线性显存扩展性（峰值显存约为窗口注意力的 1/2.4-1/2.7）。

审计修复：
- Morton 量化改为**各轴等比**（共享 range）：逐轴独立归一化把 5mm
  的 z 拉伸 ~2000 倍，序列相邻点平均 xy 跳距 0.567m（等比后 0.142m）
  ——空间局部性被摧毁大半。
- **双向扫描**：y = scan(x) + flip(scan(flip(x)))，消解序列化引入的
  人为方向性（PointMamba/PCM/Mamba3D 的共识设计）。
- **单位增益 damped EMA**：s_t = (1-α)x_t + α s_{t-1}（旧形式直流
  增益 1/(1-α) 最高 20 倍）；α 按通道对数间隔初始化覆盖多时间尺度
  （S4D 惯例），τ ∈ [~10, ~1000] token。
- **分块并行扫描**：块内下三角幂矩阵一次 einsum、块间 carry 串行
  （串行步数 N/64），替代逐 token Python 循环（实测 ~80×）。
- **层级化**：stage 间序列化网格池化（÷4，与 swin3d 共用实现与
  解码头），序列长度逐级缩短。
- DSCM 乘法接线（见 mhc.py）。
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from roadmc.models.attention.window_attention import DropPath, MHCConnection
from roadmc.models.backbone.swin3d import SegmentationHead, SerializedGridPool


def _expand_bits(v: torch.Tensor) -> torch.Tensor:
    """Expand 10-bit integers for Morton code construction."""
    v = v & 0x3FF
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def _morton_permutation(
    coords: torch.Tensor, levels: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort points by Morton code with **isotropic** quantization.

    量化尺度各轴共享（取最大 extent）：Z-order 曲线的局部性界只在
    各向同性立方体的均匀量化下成立；逐轴独立归一化等价于对空间做
    各向异性仿射拉伸再编码，近平面路面云的局部性保证随之失效。
    """
    coords_min = coords.amin(dim=1, keepdim=True)
    extent = (coords.amax(dim=1, keepdim=True) - coords_min).amax(dim=-1, keepdim=True)
    extent = torch.clamp(extent, min=1e-6)
    coords_norm = ((coords - coords_min) / extent).clamp(0.0, 1.0)
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


def _chunked_ema_scan(x: torch.Tensor, alpha: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Causal damped EMA s_t = (1-α)x_t + α s_{t-1} via blockwise parallel scan.

    块内用下三角幂矩阵一次张量积完成，块间只串行传递 carry 状态——
    串行步数从 N 降到 N/chunk。数学与逐 token 递推严格一致（回归
    测试断言逐点相等）。不使用 α^t·cumsum(x/α^t) 的朴素技巧：α^N
    在长序列下溢为 0。
    """
    B, N, C = x.shape
    L = chunk
    pad = (-N) % L
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    K = x.shape[1] // L
    xk = x.reshape(B, K, L, C)

    t_idx = torch.arange(L, device=x.device)
    delta = (t_idx[:, None] - t_idx[None, :]).clamp(min=0)          # (L, L)
    mask = (t_idx[:, None] >= t_idx[None, :])
    # T[t, j, c] = α_c^{t-j}·(1-α_c) for t ≥ j
    T = alpha.view(1, 1, C) ** delta.unsqueeze(-1) * mask.unsqueeze(-1)
    T = T * (1.0 - alpha).view(1, 1, C)

    # 块内（零初始状态）：y0[b,k,t,c] = Σ_j T[t,j,c]·x[b,k,j,c]
    y = torch.einsum("tjc,bkjc->bktc", T, xk)

    # 块间 carry：y[t] += α^{t+1}·s_prev；s_end = y[L-1]
    powvec = alpha.view(1, C) ** (t_idx + 1).unsqueeze(-1)          # (L, C)
    s = x.new_zeros(B, C)
    out_chunks = []
    for k in range(K):
        yk = y[:, k] + powvec.unsqueeze(0) * s.unsqueeze(1)
        out_chunks.append(yk)
        s = yk[:, -1]
    out = torch.cat(out_chunks, dim=1)
    return out[:, :N]


class PointMambaBlock(nn.Module):
    """Bidirectional gated-EMA mixer block（非选择性 SSM，见模块 docstring）。"""

    def __init__(
        self, dim: int, mlp_ratio: float = 4.0, use_mhc: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # α 按通道对数间隔初始化：α ∈ [0.9, 0.999]，时间常数 τ =
        # -1/ln α ∈ [9.5, 1000] token，覆盖多尺度（S4D 惯例）。前向
        # 与反向扫描各有独立衰减参数。
        alpha0 = 1.0 - torch.logspace(-1, -3, dim)
        decay_init = torch.log(alpha0 / (1.0 - alpha0))
        self.decay_fwd = nn.Parameter(decay_init.clone())
        self.decay_bwd = nn.Parameter(decay_init.clone())
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.mhc = MHCConnection(dim) if use_mhc else None
        self.drop_path = DropPath(drop_path)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """双向 damped EMA：每点聚合完整上下文，无人为方向性。"""
        alpha_f = torch.sigmoid(self.decay_fwd)
        alpha_b = torch.sigmoid(self.decay_bwd)
        fwd = _chunked_ema_scan(x, alpha_f)
        bwd = _chunked_ema_scan(x.flip(1), alpha_b).flip(1)
        return fwd + bwd

    def forward(self, coords: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del coords
        residual = x
        x = self.norm1(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = self._scan(u)
        u = self.out_proj(F.silu(u) * torch.sigmoid(gate))
        x_scan = residual + self.drop_path(u)
        ffn_out = self.drop_path(self.ffn(self.norm2(x_scan)))
        if self.mhc is not None:
            B, N, C = x_scan.shape
            mixed = self.mhc(x_scan.reshape(-1, C)).reshape(B, N, C)
            return mixed + ffn_out
        return x_scan + ffn_out


class PointMambaStage(nn.Module):
    """One sequence stage: Morton 排序 → blocks → 序列化网格池化。"""

    def __init__(
        self,
        blocks: List[PointMambaBlock],
        downsample: Optional[SerializedGridPool] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

    def _run_block(
        self, block: PointMambaBlock, coords: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, coords, x, use_reentrant=False)
        return block(coords, x)

    def forward(
        self, coords: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        perm, inv_perm = _morton_permutation(coords)
        coords_ord = _gather_along_points(coords, perm)
        x_ord = _gather_along_points(x, perm)

        for block in self.blocks:
            x_ord = self._run_block(block, coords_ord, x_ord)

        skip = _gather_along_points(x_ord, inv_perm)
        if self.downsample is None:
            return coords, skip, skip, None
        coords_out, x_out, mapping = self.downsample(coords, skip)
        return coords_out, x_out, skip, mapping


class PointMambaBackbone(nn.Module):
    """Gated-EMA sequence-mixer backbone（类名保留仅为 CLI 兼容）。"""

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
        drop_path_rate: float = 0.0,
        mixing: Optional[str] = None,
    ):
        super().__init__()
        # 本分支只支持 none / dscm；n 流 HC 的消融在 swin3d 主线上做
        # （EMA mixer 是效率基线而非贡献候选）。
        if mixing is not None:
            m = mixing.lower()
            if m.startswith("hc"):
                raise ValueError(
                    "PointMambaBackbone does not implement n-stream hyper-connections; "
                    "run the HC ablation on the swin3d backbone."
                )
            use_mhc = m not in ("none", "off")
        if tuple(num_heads) != (3, 6, 12, 24) or window_size != 64:
            warnings.warn(
                "PointMambaBackbone ignores num_heads/window_size; "
                "these flags have no effect on this branch.",
                stacklevel=2,
            )
        del num_heads, window_size

        channels = [embed_dim * (2 ** i) for i in range(4)]
        self.patch_embed = nn.Sequential(
            nn.Linear(in_channels + 3, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_idx = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks: List[PointMambaBlock] = []
            for _ in range(depths[i]):
                blocks.append(
                    PointMambaBlock(
                        channels[i], mlp_ratio=mlp_ratio, use_mhc=use_mhc,
                        drop_path=dpr[block_idx],
                    )
                )
                block_idx += 1

            downsample: Optional[SerializedGridPool]
            if i < 3:
                downsample = SerializedGridPool(channels[i], channels[i + 1])
            else:
                downsample = None

            self.stages.append(
                PointMambaStage(blocks, downsample, use_checkpoint=use_checkpoint)
            )

        self.decode = SegmentationHead(channels, num_classes)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([coords, feats], dim=-1)
        x = self.patch_embed(x)

        skip_features: List[torch.Tensor] = []
        mappings: List[Optional[torch.Tensor]] = []
        cur_coords = coords
        for stage in self.stages:
            cur_coords, x, skip, mapping = stage(cur_coords, x)
            skip_features.append(skip)
            if mapping is not None:
                mappings.append(mapping)

        return self.decode(skip_features, mappings)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N = 2, 256
    coords = torch.rand(B, N, 3)
    coords[..., 2] *= 0.005
    feats = torch.rand(B, N, 3)

    model = PointMambaBackbone(in_channels=3, num_classes=38, embed_dim=48, depths=(1, 1, 2, 1))
    model.eval()
    logits = model(coords, feats)
    assert logits.shape == (B, N, 38)
    assert torch.isfinite(logits).all()

    # 分块并行扫描 vs 逐 token 递推：逐点一致
    dim = 8
    x = torch.randn(1, 200, dim)
    alpha = torch.sigmoid(torch.randn(dim))
    ref_state = torch.zeros(1, dim)
    ref = []
    for t in range(200):
        ref_state = (1 - alpha) * x[:, t] + alpha * ref_state
        ref.append(ref_state)
    ref = torch.stack(ref, dim=1)
    fast = _chunked_ema_scan(x, alpha, chunk=64)
    assert torch.allclose(ref, fast, atol=1e-5), (ref - fast).abs().max()

    # 单位直流增益：常数输入的稳态输出 = 输入
    const = torch.ones(1, 512, dim)
    y = _chunked_ema_scan(const, torch.full((dim,), 0.95))
    assert abs(float(y[0, -1, 0]) - 1.0) < 1e-3, float(y[0, -1, 0])

    print(f"PointMambaBackbone (gated-EMA mixer): output={logits.shape}, "
          f"params={sum(p.numel() for p in model.parameters())}")
