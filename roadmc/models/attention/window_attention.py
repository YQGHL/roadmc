"""3D window attention for point cloud transformers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from roadmc.models.mhc.mhc import (  # noqa: E402 — sys.path bootstrap above
    HyperConnection,
    MHCConnection,
)

# 窗口注意力偏置的显存上限（超出即判定为退化窗口占用而非正常配置）。
_MAX_BIAS_BYTES: int = 3 * (1024 ** 3)

# C1 每窗口占用上限乘数：window_size 是目标占用，退化场景（稠密病害
# 集群、collate padding 塌缩）可把数百点塞进一个窗口。上限 = window_size
# × 此乘数，超限点按序重切成新窗口。实测占用：v2 ≤71(ws=32)/98(ws=64)、
# v3 ≤65/103，4× 上限不触发 → 结果 bit-identical；raw 稠密场景 1032 点
# 被压到上限内，偏置内存从 22.7 GiB 降到 O(几 MB)。
MAX_OCCUPANCY_MULTIPLIER: int = 4


def parse_mixing(mixing: str) -> tuple[str, int]:
    """解析混合模式字符串 → (kind, n_streams)。

    - ``none``：标准残差（无混合）
    - ``dscm``：通道维双随机混合残差（本项目模块，n=1）
    - ``hc2`` / ``hc4``：文献口径 n 流 Hyper-Connections（mHC 对照）
    """
    m = mixing.lower()
    if m in ("none", "off"):
        return "none", 1
    if m in ("dscm", "mhc"):  # 'mhc' 为旧 CLI 别名，语义 = DSCM
        return "dscm", 1
    if m.startswith("hc"):
        n = int(m[2:]) if len(m) > 2 else 2
        if n < 1:
            raise ValueError(f"hc stream count must be >= 1, got {n}")
        return "hc", n
    raise ValueError(f"unknown mixing mode: {mixing!r} (expected none|dscm|hc2|hc4)")


def _cap_window_occupancy(
    window_id: torch.Tensor,
    max_occupancy: int,
    num_windows: int,
) -> torch.Tensor:
    """Re-chunk overflow points of crowded windows into fresh bounded windows.

    ``window_size`` is a *target* occupancy, not an upper bound: the columnar
    grid side ``g`` is derived from ``round(sqrt(N/window_size))``, so a
    spatially concentrated cluster (e.g. a dense pothole interior, or all the
    padding points of a short scene) can put hundreds of points into one bin.
    ``_window_attention_sdpa`` then pads every window to the batch-global max
    occupancy ``M``, and the relative-position-bias footprint ``W*M^2`` blows
    up (measured 22.7 GiB for M=1056).

    This cap keeps the *first* ``max_occupancy`` points of each crowded window
    in place and re-chunks the rest into fresh windows of at most
    ``max_occupancy`` points (consecutive overflow points stay adjacent, so the
    new windows retain spatial locality).  No point is dropped and memory stays
    bounded: both ``M`` and ``W`` grow only O(overflow / max_occupancy).  For
    windows that never exceed the cap (all well-behaved scenes) the return is
    bit-identical to the input, so existing checkpoints trained without the
    cap are unaffected.

    Fresh ids are assigned **per batch element** starting at ``num_windows``.
    ``_window_attention_sdpa`` separates batch elements by adding
    ``arange(B) * max_wid`` (max_wid = global max + 1), so an element's ids
    must lie in ``[0, max_wid)``; per-element numbering keeps them disjoint
    across elements regardless of differing overflow counts.
    """
    B, N = window_id.shape

    def _rechunk(wid: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(wid, minlength=num_windows)
        overflow = torch.nonzero(counts > max_occupancy, as_tuple=False).squeeze(-1)
        if overflow.numel() == 0:
            return wid
        new = wid.clone()
        next_id = num_windows
        for w in overflow.tolist():
            positions = torch.nonzero(wid == w, as_tuple=False).squeeze(-1)
            extra = positions[max_occupancy:]
            n_extra = extra.numel()
            n_chunks = (n_extra + max_occupancy - 1) // max_occupancy
            for c in range(n_chunks):
                lo = c * max_occupancy
                hi = min(lo + max_occupancy, n_extra)
                new[extra[lo:hi]] = next_id
                next_id += 1
        return new

    if B == 1:
        return _rechunk(window_id[0]).unsqueeze(0)
    return torch.stack([_rechunk(window_id[b]) for b in range(B)], dim=0)


def _window_partition(
    coords: torch.Tensor,
    window_size: int,
    shift: bool = False,
    mode: str = "columnar",
    valid_mask: torch.Tensor | None = None,
    max_occupancy: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Assign points to attention windows.

    两种分窗模式：

    - ``columnar``（默认，2.5D 柱状）：窗口只由 (x, y) 决定，z 不参与
      分箱。路面点云是近平面的（x/y 米级、z 毫米级）：逐轴归一化的
      立方分窗会把毫米级高度残差拉伸成整轴，导致坑底与坑沿被切进
      不同窗口——病害点被系统性隔离出其物理邻域。柱状窗口保证
      病害全深度与其周边路表同窗。
    - ``cubic``：三轴分箱（各向同性点云用）。

    Shift 通过**平移分箱原点**实现（bin = floor(u·g + 0.5)，每轴产生
    g+1 个箱，边缘窗口自然变小）——不做 mod-1 环绕。旧实现的
    ``% 1.0`` 环绕复刻了 Swin 的 cyclic shift 却没有配套 attention
    mask，使场景两端相距整条路面的点落入同一窗口互相 attend
    （shifted block 中约 58% 的窗口跨越 >90% 场景范围）。平移分箱
    无环绕即无需掩码（Stratified Transformer 的做法）。

    Returns:
        window_id: (B, N) 整型窗口编号。
        num_windows: 编号上界（含空窗）。
    """
    B, N, _ = coords.shape

    coords_min = coords.amin(dim=1, keepdim=True)
    coords_max = coords.amax(dim=1, keepdim=True)
    coords_range = coords_max - coords_min
    coords_range = torch.where(coords_range < 1e-6, torch.ones_like(coords_range), coords_range)
    coords_norm = (coords - coords_min) / coords_range

    n_windows = max(1, round(N / window_size))
    if mode == "columnar":
        g = max(1, round(n_windows ** 0.5))
        u = coords_norm[..., :2]  # z 不分箱
        dims = 2
    else:
        g = max(1, round(n_windows ** (1.0 / 3.0)))
        u = coords_norm
        dims = 3

    if shift:
        # 平移半个窗口的分箱原点：bin ∈ [0, g]，共 g+1 个箱/轴。
        bin_idx = torch.floor(u * g + 0.5).long().clamp(0, g)
        base = g + 1
    else:
        bin_idx = torch.floor(u * g).long().clamp(0, g - 1)
        base = g

    if dims == 2:
        window_id = bin_idx[..., 0] * base + bin_idx[..., 1]
        num_windows = base ** 2
    else:
        window_id = (
            bin_idx[..., 0] * (base * base)
            + bin_idx[..., 1] * base
            + bin_idx[..., 2]
        )
        num_windows = base ** 3

    # C1: 每窗口占用上限。退化场景（稠密病害集群 / collate padding 点
    # 塌缩）可把数百点塞进一个窗口，把全局 M 撑爆（实测 1056 点 →
    # 偏置 22.7 GiB）。超限点的多余部分按序重切为有界新窗口——不丢点、
    # 内存有界。窗口不超限时返回 bit-identical 结果，已训练的 checkpoint
    # 不受影响。
    if valid_mask is not None and not bool(valid_mask.all()):
        # D1: 无效点（collate padding）不参与分箱，统一并入独立窗口，
        # 避免 (0,0,0) 塌缩进真实窗口。之后照常过占用上限。
        invalid_mask = ~valid_mask.reshape(-1)
        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            window_id = window_id.reshape(-1).clone()
            window_id[invalid_mask] = (
                num_windows + torch.arange(n_invalid, device=coords.device)
            ).to(window_id.dtype)
            window_id = window_id.reshape(B, N)
            num_windows += n_invalid
    if max_occupancy is not None and max_occupancy > 0:
        window_id = _cap_window_occupancy(window_id, max_occupancy, num_windows)

    return window_id, num_windows


def _window_attention_sdpa(
    coords: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_id: torch.Tensor,
    pos_mlp: nn.Module,
) -> torch.Tensor:
    """Batched window attention via sort + pad + scaled_dot_product_attention.

    与逐窗 Python 循环在数学上严格等价（同一 softmax(QKᵀ/√d + B)V），
    但把每场景数百次小 kernel 调用合并为一次 SDPA——旧实现单层前向
    39-254 ms（GPU 利用率损失 >600 倍），且 `for wid in torch.unique`
    在 CUDA tensor 上迭代每窗触发一次 device-host 同步。
    """
    B, H, N, D = q.shape
    device = q.device

    # 全批展平：不同 batch 元素的窗口互不相通。
    max_wid = int(window_id.max().item()) + 1
    flat_wid = (window_id + torch.arange(B, device=device)[:, None] * max_wid).reshape(-1)
    order = torch.argsort(flat_wid, stable=True)
    sorted_wid = flat_wid[order]
    _, counts = torch.unique_consecutive(sorted_wid, return_counts=True)
    W = counts.shape[0]
    M = int(counts.max().item())

    # 每个排序位置的 (窗口序号, 窗口内槽位)
    w_of = torch.repeat_interleave(torch.arange(W, device=device), counts)
    offsets = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(B * N, device=device) - offsets[w_of]

    # M 桶化到 32 的倍数：把逐步各异的 (W,H,M,M) 形状收敛到少数几档
    # （实测 distinct M 从 21 降到 2），使缓存分配器的块可复用，并让
    # mask 的最后一维恒为 32 的倍数（fused attention bias 的对齐前提）。
    # 数值惰性已在 fp64 下对前向与反向验证（bucket ∈ {1,8,32,64,128}
    # 前向 maxabsdiff ≤ 6.7e-16，反向 ≤ 1.6e-14）。
    M = ((M + 31) // 32) * 32

    # 偏置显存的悬崖预检：M 是**全批次全局**最大窗口占用，单个拥挤
    # 窗口（点共面/重合、远离群点）会把全部 W 个窗口一起撑大——
    # pos_mlp 在 (W,M,M) 上稠密求值，实测退化几何可要 6.9-19.2 GiB。
    # 与其在算子内 OOM，不如给出可诊断的报错。
    pos_hidden = getattr(pos_mlp[0], "out_features", H) if len(pos_mlp) else H
    bias_bytes = W * M * M * (3 + 2 * pos_hidden + H) * q.element_size()
    if bias_bytes > _MAX_BIAS_BYTES:
        raise RuntimeError(
            f"window attention bias would need {bias_bytes / 2**30:.1f} GiB "
            f"(W={W}, M={M}, pos_hidden={pos_hidden}); this indicates degenerate "
            "window occupancy (coincident/coplanar points or a far outlier), not "
            "a normal configuration. Reduce window_size or inspect the cloud."
        )

    def pad(t: torch.Tensor) -> torch.Tensor:
        # t: (B, H, N, D) -> (W, H, M, D)，padding 为 0
        t_flat = t.permute(0, 2, 1, 3).reshape(B * N, H, -1)[order]
        buf = t.new_zeros(W, M, H, t_flat.shape[-1])
        buf[w_of, slot] = t_flat
        return buf.permute(0, 2, 1, 3)

    q_pad, k_pad, v_pad = pad(q), pad(k), pad(v)

    coords_flat = coords.reshape(B * N, 3)[order]
    coords_pad = coords.new_zeros(W, M, 3)
    coords_pad[w_of, slot] = coords_flat
    rel = coords_pad.unsqueeze(2) - coords_pad.unsqueeze(1)      # (W, M, M, 3)
    bias = pos_mlp(rel).permute(0, 3, 1, 2)                       # (W, H, M, M)

    valid = torch.zeros(W, M, dtype=torch.bool, device=device)
    valid[w_of, slot] = True
    # 用有限大负值而非 -inf：fp16/低精度 SDPA 内核对 -inf mask 的
    # 处理在部分后端（尤其新架构 GPU 的 cuDNN 路径）不稳定；
    # softmax(-0.5·finfo.min) 数值上同样为零权重。
    neg_fill = torch.finfo(bias.dtype).min / 2
    attn_mask = bias.masked_fill(~valid[:, None, None, :], neg_fill)

    out_pad = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)

    out_flat = out_pad.permute(0, 2, 1, 3)[w_of, slot]            # (B·N, H, D)
    # argsort 而非 empty+scatter：后者的正确性依赖"order 必为完整置换"
    # 这一上游性质，若被破坏就会读到未初始化内存（真实越界）；argsort
    # 在同等语义下把最坏情形降级为"结果错但不越界"。
    inv = torch.argsort(order)
    out = out_flat[inv].reshape(B, N, H, D).permute(0, 2, 1, 3)
    return out


def _window_attention_blockwise(
    coords: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_id: torch.Tensor,
    pos_mlp: nn.Module,
    softmax: nn.Softmax,
) -> torch.Tensor:
    """Reference per-window loop (kept for equivalence tests only)."""
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


class DropPath(nn.Module):
    """Stochastic depth (Huang et al. 2016): x + b/(1-p)·F(x), b~Bernoulli."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class WindowAttention3D(nn.Module):
    """Windowed point attention with MLP-learned relative position bias.

    分窗默认 2.5D 柱状（z 不分箱），shift 用平移分箱原点（无环绕）；
    注意力经 sort+pad+SDPA 批量执行，与逐窗参考实现数学等价。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        partition_mode: str = "columnar",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.partition_mode = partition_mode

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
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        window_id, _ = _window_partition(
            coords, self.window_size, shift=shift, mode=self.partition_mode,
            valid_mask=valid_mask,
            max_occupancy=self.window_size * MAX_OCCUPANCY_MULTIPLIER,
        )
        out = _window_attention_sdpa(
            coords=coords,
            q=q,
            k=k,
            v=v,
            window_id=window_id,
            pos_mlp=self.pos_mlp,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DeformableWindowAttention3D(nn.Module):
    """EXPERIMENTAL — not wired into any backbone; do not cite as a method component.

    已知缺陷（审计 F0.4，未修复）：argmin 最近邻采样不可微（offset
    学习只剩偏置旁路）；torch.cdist 距离阵 O(N²K) 在 N=8192 时约 4GB。
    如需启用须改为 kNN 软插值 + 局部候选检索并跑消融。
    """

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
    """Transformer block with window attention and optional DSCM mixing.

    DSCM（双随机通道混合）以**乘法/凸组合**语义作用在被携带的残差流
    上：``x_out = H·x_attn + FFN(norm(x_attn))``。双随机矩阵满足
    σ_max(H)=1 且积仍双随机，该接线的深层复合范数有界；旧的加法
    接线 ``y = x + H·x`` 的块雅可比含 (I+H)，沿全 1 方向特征值为 2，
    同维块堆叠特征范数按 2^L 指数增长（审计实测 12 块 ×428）。
    H 恒等初始化，训练起点严格等价于标准残差。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        window_size: int = 32,
        mlp_ratio: float = 4.0,
        shift: bool = False,
        use_mhc: bool = True,
        drop_path: float = 0.0,
        mixing: str | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.shift = shift
        # mixing 优先；use_mhc 保留为旧 CLI 的兼容入口。
        if mixing is None:
            mixing = "dscm" if use_mhc else "none"
        self.mixing_kind, self.n_streams = parse_mixing(mixing)
        self.use_mhc = self.mixing_kind != "none"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size, mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        if self.mixing_kind == "dscm":
            self.mhc = MHCConnection(dim)
            self.hc_attn = None
            self.hc_ffn = None
        elif self.mixing_kind == "hc":
            # HC 口径：每个子层（注意力 / FFN）各自是一"层"，各带一组
            # 读出/混合/写回参数。
            self.mhc = None
            self.hc_attn = HyperConnection(dim, n_streams=self.n_streams)
            self.hc_ffn = HyperConnection(dim, n_streams=self.n_streams)
        else:
            self.mhc = None
            self.hc_attn = None
            self.hc_ffn = None
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mixing_kind == "hc":
            # x: (B, N, n, C) —— 每个子层读出 → 分支 → 双随机混合写回
            a_in = self.hc_attn.read(x)
            branch = self.drop_path(
                self.attn(coords, self.norm1(a_in), shift=self.shift, valid_mask=valid_mask)
            )
            x = self.hc_attn.write(x, branch)
            f_in = self.hc_ffn.read(x)
            branch_ffn = self.drop_path(self.ffn(self.norm2(f_in)))
            return self.hc_ffn.write(x, branch_ffn)

        B, N, C = x.shape
        x_attn = x + self.drop_path(
            self.attn(coords, self.norm1(x), shift=self.shift, valid_mask=valid_mask)
        )
        ffn_out = self.drop_path(self.ffn(self.norm2(x_attn)))
        if self.mhc is not None:
            mixed = self.mhc(x_attn.reshape(-1, C)).reshape(B, N, C)
            return mixed + ffn_out
        return x_attn + ffn_out


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
