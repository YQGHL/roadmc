"""Hierarchical shifted-window point transformer for pavement segmentation.

真层级结构（2026-07 审计修复）：stage 之间做序列化网格池化
（serialized grid pooling，点数 ÷4、通道 ×2、窗口物理尺寸随之扩大），
解码端按记录的池化映射逐级上采样并融合 skip——旧实现 stage 间仅
nn.Linear 通道加宽、点数与窗口划分全程不变，"4-stage hierarchical"
名不副实，且 768 通道在全部 N 点上运行造成显存天花板。

命名口径：本骨干是 *hierarchical shifted-window point transformer
(Swin3D-style blocks)*，非 Yang et al. Swin3D 的复现（无 cRSE）。

Input:  coords (B, N, 3), feats (B, N, in_channels)
Output: (B, N, num_classes) per-point logits
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple

from roadmc.models.attention.window_attention import (
    ShiftedWindowTransformerBlock,
    parse_mixing,
)
from roadmc.models.mhc.mhc import HyperConnection


def _serial_order(coords: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """xy Morton 序（等比量化）用于序列化池化分组。

    量化使用**各轴共享**的尺度（等比）：路面点云 x/y 米级、z 毫米级，
    z 不参与排序键（2.5D 口径，与柱状分窗一致）。

    Args:
        coords: (B, N, 3)。
        bits: 每轴量化位数。

    Returns:
        order: (B, N) 排序索引。
    """
    B, N, _ = coords.shape
    xy = coords[..., :2]
    mins = xy.amin(dim=1, keepdim=True)
    ranges = (xy.amax(dim=1, keepdim=True) - mins).amax(dim=-1, keepdim=True)
    ranges = torch.clamp(ranges, min=1e-6)
    q = ((xy - mins) / ranges * (2**bits - 1)).long().clamp(0, 2**bits - 1)

    # 逐位交织（清晰优先；bits=10 → 循环 10 次，纯张量位运算）
    code = torch.zeros(B, N, dtype=torch.long, device=coords.device)
    for i in range(bits):
        code |= ((q[..., 0] >> i) & 1) << (2 * i + 1)
        code |= ((q[..., 1] >> i) & 1) << (2 * i)
    return code.argsort(dim=1, stable=True)


class SerializedGridPool(nn.Module):
    """Serialized grid pooling: 按 xy Morton 序每 4 个连续点并为 1 个。

    PTv3 风格的序列化池化：排序后分组取均值（坐标与特征），通道经
    Linear 投影加宽。返回池化映射（细点 → 粗点索引，均在各自的
    原始顺序下）供解码端 O(N) gather 上采样——不做 O(N·M) 距离阵。
    """

    STRIDE = 4

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self, coords: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (coords_pooled, x_pooled, fine_to_coarse) with M = ceil(N/4).

        ``x`` 可以是 (B, N, C) 或带流维的 (B, N, n, C)——池化只作用在点维，
        投影/归一只作用在最后的通道维，中间维原样保留。
        """
        B, N = x.shape[0], x.shape[1]
        trailing = tuple(x.shape[2:])          # (C,) 或 (n, C)
        x_flat = x.reshape(B, N, -1)
        Cflat = x_flat.shape[-1]
        s = self.STRIDE
        order = _serial_order(coords)

        pad = (-N) % s
        if pad:
            # 补齐到 4 的倍数：重复排序后的最后一个点。
            tail = order[:, -1:].expand(B, pad)
            order_p = torch.cat([order, tail], dim=1)
        else:
            order_p = order
        Np = order_p.shape[1]
        M = Np // s

        gather_c = torch.gather(coords, 1, order_p.unsqueeze(-1).expand(B, Np, 3))
        gather_x = torch.gather(x_flat, 1, order_p.unsqueeze(-1).expand(B, Np, Cflat))
        coords_pooled = gather_c.reshape(B, M, s, 3).mean(dim=2)
        x_pooled = gather_x.reshape(B, M, s, Cflat).mean(dim=2).reshape(B, M, *trailing)
        x_pooled = self.norm(self.proj(x_pooled))

        # fine_to_coarse[b, i] = 原始顺序下第 i 个细点所属的粗点索引。
        group_of_sorted = (
            torch.arange(Np, device=x.device).div(s, rounding_mode="floor")
            .unsqueeze(0).expand(B, Np)
        )
        fine_to_coarse = torch.empty(B, N, dtype=torch.long, device=x.device)
        fine_to_coarse.scatter_(1, order_p[:, :N], group_of_sorted[:, :N])

        return coords_pooled, x_pooled, fine_to_coarse


class Stage(nn.Module):
    """One stage: N shifted-window Transformer blocks + optional grid pooling.

    Swin 约定：块序为 W-MSA → SW-MSA（偶数块不 shift、奇数块 shift）。
    skip 在池化前捕获，供解码端在该分辨率融合。
    """

    def __init__(
        self,
        blocks: List[ShiftedWindowTransformerBlock],
        downsample: Optional[SerializedGridPool] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample
        self.use_checkpoint = use_checkpoint

    def _run_block(
        self,
        block: ShiftedWindowTransformerBlock,
        coords: torch.Tensor,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, coords, x, valid_mask, use_reentrant=False)
        return block(coords, x, valid_mask=valid_mask)

    def forward(
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """blocks → pool. Returns (coords_out, x_out, skip, fine_to_coarse)."""
        for block in self.blocks:
            x = self._run_block(block, coords, x, valid_mask)
        skip = x
        if self.downsample is None:
            return coords, x, skip, None
        coords_out, x_out, mapping = self.downsample(coords, x)
        return coords_out, x_out, skip, mapping


class SegmentationHead(nn.Module):
    """Decoder: 按池化映射逐级上采样（gather 广播）+ skip 融合。

    Decode chain (channels [C0, C1, C2, C3], 点数 [N, N/4, N/16, N/64])::

        f3 (N/64, C3) → Linear(C3→C2) → unpool(map2) → cat f2 → fuse (C1)
        → unpool(map1) → cat f1 → fuse (C0)
        → unpool(map0) → cat f0 → fuse (C0) → cls (num_classes)
    """

    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.merge3 = nn.Sequential(nn.Linear(c3, c2), nn.GELU())
        self.fuse3 = nn.Sequential(nn.Linear(c2 * 2, c1), nn.GELU())
        self.fuse2 = nn.Sequential(nn.Linear(c1 * 2, c0), nn.GELU())
        self.fuse1 = nn.Sequential(nn.Linear(c0 * 2, c0), nn.GELU())
        self.cls_head = nn.Linear(c0, num_classes)

    @staticmethod
    def _unpool(x: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
        """Broadcast coarse features back to fine points via the pool mapping."""
        B, N = mapping.shape
        C = x.shape[-1]
        return torch.gather(x, 1, mapping.unsqueeze(-1).expand(B, N, C))

    def forward(
        self,
        features: List[torch.Tensor],
        mappings: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        f0, f1, f2, f3 = features
        map0, map1, map2 = mappings[0], mappings[1], mappings[2]

        x = self.merge3(f3)
        x = self._unpool(x, map2)
        x = self.fuse3(torch.cat([x, f2], dim=-1))

        x = self._unpool(x, map1)
        x = self.fuse2(torch.cat([x, f1], dim=-1))

        x = self._unpool(x, map0)
        x = self.fuse1(torch.cat([x, f0], dim=-1))

        return self.cls_head(x)


class Swin3D(nn.Module):
    """Hierarchical shifted-window point transformer (Swin3D-style blocks).

    1. **PatchEmbedding** — concat(coords, feats) → Linear → LayerNorm
    2. **Stages 0–3** — window blocks → serialized grid pooling (÷4, 通道 ×2)
    3. **SegmentationHead** — 映射上采样 + skip 融合 → per-point logits

    Parameters
    ----------
    in_channels : int
        输入特征通道数（默认 3：强度 / PCA 曲率 / 局部高度残差）。
    num_classes : int
        JTG 类别数（默认 38）。
    embed_dim : int
        stage 0 基础通道数，逐 stage ×2。
    depths, num_heads : tuple
        每 stage 的块数与注意力头数。
    window_size : int
        每窗口目标点数。
    drop_path_rate : float
        stochastic depth 上限，按块全局序号线性递增（Swin 惯例）。
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
        use_checkpoint: bool = False,
        use_mhc: bool = True,
        drop_path_rate: float = 0.0,
        mixing: Optional[str] = None,
    ):
        super().__init__()

        if mixing is None:
            mixing = "dscm" if use_mhc else "none"
        self.mixing = mixing
        self.mixing_kind, self.n_streams = parse_mixing(mixing)

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
            blocks: List[ShiftedWindowTransformerBlock] = []
            for j in range(depths[i]):
                # Swin 约定 W-MSA → SW-MSA：偶数块不 shift。
                shift = (j % 2 == 1)
                blocks.append(
                    ShiftedWindowTransformerBlock(
                        dim=channels[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        shift=shift,
                        use_mhc=use_mhc,
                        drop_path=dpr[block_idx],
                        mixing=mixing,
                    )
                )
                block_idx += 1

            downsample: Optional[SerializedGridPool]
            if i < 3:
                downsample = SerializedGridPool(channels[i], channels[i + 1])
            else:
                downsample = None

            self.stages.append(Stage(blocks, downsample, use_checkpoint=use_checkpoint))

        self.decode = SegmentationHead(channels, num_classes)

    def forward(
        self,
        coords: torch.Tensor,
        feats: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patch embed → 4 stages (skip + 池化映射) → decode.

        HC 模式下残差被扩展为 n 条流 (B,N,n,C) 贯穿骨干，skip 在进入
        解码器前按流维取均值收缩——恒等初始化时该 expand/contract 对
        使整网 bit-exact 等价于标准残差网络。
        """
        x = torch.cat([coords, feats], dim=-1)
        x = self.patch_embed(x)
        if self.mixing_kind == "hc":
            x = HyperConnection.expand(x, self.n_streams)

        skip_features: List[torch.Tensor] = []
        mappings: List[Optional[torch.Tensor]] = []
        cur_coords = coords
        for stage in self.stages:
            cur_coords, x, skip, mapping = stage(cur_coords, x, valid_mask)
            if self.mixing_kind == "hc":
                skip = HyperConnection.contract(skip)
            skip_features.append(skip)
            if mapping is not None:
                mappings.append(mapping)

        return self.decode(skip_features, mappings)


# Self-test
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")
    B, N = 2, 512

    coords = torch.rand(B, N, 3, device=device)
    coords[..., 2] *= 0.005  # 路面型 z 量级
    feats = torch.rand(B, N, 3, device=device)

    model = Swin3D(
        in_channels=3,
        num_classes=38,
        embed_dim=48,
        depths=(1, 1, 2, 1),
        num_heads=(2, 4, 8, 16),
        window_size=64,
        drop_path_rate=0.1,
    )
    model.eval()

    logits = model(coords, feats)
    assert logits.shape == (B, N, 38), f"Expected ({B}, {N}, 38), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in output"

    loss = logits.sum()
    loss.backward()
    params_no_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_no_grad, f"Parameters without gradient: {params_no_grad}"

    # 层级断言：各 stage 点数 ÷4
    with torch.no_grad():
        x = model.patch_embed(torch.cat([coords, feats], dim=-1))
        cur = coords
        sizes = []
        for stage in model.stages:
            cur, x, skip, mapping = stage(cur, x)
            sizes.append(skip.shape[1])
    assert sizes == [512, 128, 32, 8], f"hierarchy sizes wrong: {sizes}"

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Swin3D: output={logits.shape}, params={param_count}, stage sizes={sizes}")

    logits2 = model(coords, feats)
    assert torch.allclose(logits, logits2, atol=1e-5), "Non-deterministic output"
    print("Swin3D: deterministic forward pass (eval mode)")
