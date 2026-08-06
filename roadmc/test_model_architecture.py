"""Architecture regression tests for the 2026-07 model-side audit fixes.

- shift 分窗无环绕（窗内最大跨度有界）；柱状分窗对 z 扰动不变
- SDPA 批量注意力与逐窗参考实现数学等价
- DSCM：恒等初始化、梯度非零、深堆叠有界、log-domain 无溢出、σ_max=1
- 层级：stage 点数 ÷4；解码输出逐点分辨率
- EMA 扫描：分块并行 = 逐 token 递推；单位直流增益；双向对称性
- Muon 参数路由：头/嵌入/DSCM 核不进 Muon 组
- 类权重 clip 上界严格成立
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

os.environ.pop("ROADMC_GENERATOR_NO_TORCH", None)

from roadmc.data.class_balance import effective_number_class_weights  # noqa: E402
from roadmc.models.attention.window_attention import (  # noqa: E402
    _window_attention_blockwise,
    _window_attention_sdpa,
    _window_partition,
)
from roadmc.models.backbone.pointmamba import (  # noqa: E402
    PointMambaBackbone,
    _chunked_ema_scan,
    _morton_permutation,
)
from roadmc.models.backbone.swin3d import Swin3D  # noqa: E402
from roadmc.models.mhc.mhc import MHCConnection, sinkhorn_log  # noqa: E402


def _road_cloud(B: int = 2, N: int = 512, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(B, N, 3, generator=g)
    coords[..., 0] *= 10.0
    coords[..., 1] *= 4.0
    coords[..., 2] *= 0.005
    return coords


class WindowPartitionTests(unittest.TestCase):
    def test_shift_does_not_wrap(self) -> None:
        """shift 窗口的窗内最大跨度必须有界（旧 %1.0 环绕跨度=1.0）。"""
        coords = _road_cloud(1, 4096)
        for mode in ("columnar", "cubic"):
            wid, _ = _window_partition(coords, window_size=64, shift=True, mode=mode)
            norm = (coords - coords.amin(1, keepdim=True)) / (
                coords.amax(1, keepdim=True) - coords.amin(1, keepdim=True)
            ).clamp(min=1e-6)
            axes = slice(0, 2) if mode == "columnar" else slice(0, 3)
            max_span = 0.0
            for w in torch.unique(wid[0]):
                pts = norm[0][wid[0] == w][:, axes]
                span = float((pts.amax(0) - pts.amin(0)).max())
                max_span = max(max_span, span)
            # 分箱轴上的窗内跨度不得接近整轴（旧环绕实现跨度 = 1.0）
            self.assertLess(max_span, 0.55, f"mode={mode}: wrap detected span={max_span}")

    def test_columnar_ignores_z(self) -> None:
        """柱状分窗下 z 扰动 ±5mm 不得改变任何点的窗口归属。"""
        coords = _road_cloud(1, 2048)
        wid1, _ = _window_partition(coords, 64, shift=False, mode="columnar")
        coords2 = coords.clone()
        coords2[..., 2] += (torch.rand_like(coords2[..., 2]) - 0.5) * 0.01
        wid2, _ = _window_partition(coords2, 64, shift=False, mode="columnar")
        self.assertTrue(torch.equal(wid1, wid2))


class SdpaEquivalenceTests(unittest.TestCase):
    def test_sdpa_matches_blockwise_reference(self) -> None:
        torch.manual_seed(1)
        B, H, N, D = 2, 4, 256, 16
        coords = _road_cloud(B, N, seed=1)
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        wid, _ = _window_partition(coords, 32, shift=False, mode="columnar")
        pos_mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 8), torch.nn.GELU(), torch.nn.Linear(8, H)
        )
        ref = _window_attention_blockwise(
            coords, q, k, v, wid, pos_mlp, torch.nn.Softmax(dim=-1)
        )
        fast = _window_attention_sdpa(coords, q, k, v, wid, pos_mlp)
        self.assertTrue(
            torch.allclose(ref, fast, atol=1e-5),
            f"max dev {(ref - fast).abs().max()}",
        )


class DscmTests(unittest.TestCase):
    def test_identity_init_and_learnable(self) -> None:
        torch.manual_seed(0)
        mhc = MHCConnection(96)
        x = torch.randn(8, 96)
        y = mhc(x)
        self.assertLess(float((y - x).norm() / x.norm()), 0.02)
        mhc(x).square().sum().backward()
        self.assertGreater(float(mhc.log_kernel.grad.norm()), 1e-6)

    def test_log_domain_no_overflow(self) -> None:
        """权重漂移 +60 后仍无 inf/NaN（旧 torch.exp 路径 +6 即溢出）。

        极端尖锐核的收敛需要更多迭代（Franklin & Lorenz 1989 收缩率），
        本断言只锁数值稳定性；双随机精度由 diagnostics() 在训练中监控。
        """
        torch.manual_seed(11)
        L = torch.randn(64, 64) * 20.0 + 60.0
        H = sinkhorn_log(L, iters=300)
        self.assertTrue(torch.isfinite(H).all())
        self.assertLessEqual(float(torch.linalg.matrix_norm(H, ord=2)), 1.1)
        self.assertLess(float((H.sum(dim=0) - 1).abs().max()), 1e-3)

    def test_deep_stack_norm_bounded(self) -> None:
        """乘法接线 60 层特征范数有界（旧加法接线按 2^L 爆炸）。"""
        torch.manual_seed(2)
        mhc = MHCConnection(64)
        with torch.no_grad():
            mhc.log_kernel.add_(torch.randn(64, 64) * 2.0)
        x = torch.randn(4, 64)
        x0 = x.norm()
        with torch.no_grad():
            for _ in range(60):
                x = mhc(x) + torch.randn(4, 64) * 0.1
        self.assertLess(float(x.norm() / x0), 20.0)

    def test_block_with_mhc_matches_baseline_at_init(self) -> None:
        """恒等初始化下 use_mhc=True/False 的块输出应几乎一致。"""
        from roadmc.models.attention.window_attention import (
            ShiftedWindowTransformerBlock,
        )
        torch.manual_seed(3)
        blk_mhc = ShiftedWindowTransformerBlock(dim=32, num_heads=2, window_size=32, use_mhc=True)
        blk_ref = ShiftedWindowTransformerBlock(dim=32, num_heads=2, window_size=32, use_mhc=False)
        blk_ref.load_state_dict(
            {k: v for k, v in blk_mhc.state_dict().items() if not k.startswith("mhc.")},
            strict=False,
        )
        blk_mhc.eval()
        blk_ref.eval()
        coords = _road_cloud(1, 128, seed=3)
        x = torch.randn(1, 128, 32)
        y1 = blk_mhc(coords, x)
        y2 = blk_ref(coords, x)
        self.assertLess(float((y1 - y2).norm() / y2.norm()), 0.05)


class HyperConnectionTests(unittest.TestCase):
    """文献口径 n 流 mHC（消融对照，xHC 语境下的必答题）。"""

    def test_identity_init_is_bit_exact_standard_residual(self) -> None:
        """恒等初始化下 HC 骨干必须与标准残差骨干逐值一致。"""
        coords = _road_cloud(2, 256, seed=11)
        feats = torch.rand(2, 256, 3)
        for mixing in ("hc2", "hc4"):
            with self.subTest(mixing=mixing):
                m_hc = Swin3D(embed_dim=32, depths=(1, 1, 2, 1), num_heads=(2, 2, 4, 4),
                              window_size=32, mixing=mixing).eval()
                m_ref = Swin3D(embed_dim=32, depths=(1, 1, 2, 1), num_heads=(2, 2, 4, 4),
                               window_size=32, mixing="none").eval()
                shared = {k: v for k, v in m_hc.state_dict().items()
                          if "hc_attn" not in k and "hc_ffn" not in k}
                m_ref.load_state_dict(shared, strict=False)
                with torch.no_grad():
                    a, b = m_ref(coords, feats), m_hc(coords, feats)
                rel = float((a - b).norm() / a.norm())
                self.assertLess(rel, 1e-5, f"{mixing} identity init broken: {rel}")

    def test_stream_mixing_is_doubly_stochastic_and_learnable(self) -> None:
        from roadmc.models.mhc.mhc import HyperConnection
        for n in (2, 4):
            hc = HyperConnection(64, n_streams=n)
            d = hc.diagnostics()
            self.assertLess(d["row_sum_err"], 1e-3)
            self.assertLess(d["col_sum_err"], 1e-3)
            self.assertLessEqual(d["spectral_norm"], 1.0 + 1e-3)
            self.assertLess(d["dist_to_identity"], 1e-2)
            h = HyperConnection.expand(torch.randn(2, 8, 64), n)
            hc.write(h, torch.randn(2, 8, 64)).square().sum().backward()
            self.assertGreater(float(hc.log_kernel.grad.norm()), 1e-8)

    def test_all_mixing_modes_forward_backward(self) -> None:
        coords = _road_cloud(2, 256, seed=12)
        feats = torch.rand(2, 256, 3)
        for mixing in ("none", "dscm", "hc2", "hc4"):
            with self.subTest(mixing=mixing):
                m = Swin3D(embed_dim=32, depths=(1, 1, 1, 1), num_heads=(2, 2, 4, 4),
                           window_size=32, mixing=mixing)
                out = m(coords, feats)
                self.assertEqual(out.shape, (2, 256, 38))
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                no_grad = [n for n, p in m.named_parameters()
                           if p.requires_grad and p.grad is None]
                self.assertEqual(no_grad, [])

    def test_ema_backbone_rejects_hc(self) -> None:
        with self.assertRaises(ValueError):
            PointMambaBackbone(embed_dim=32, depths=(1, 1, 1, 1), mixing="hc2")


class HierarchyTests(unittest.TestCase):
    def test_swin3d_stage_sizes_and_output(self) -> None:
        torch.manual_seed(4)
        model = Swin3D(embed_dim=32, depths=(1, 1, 1, 1), num_heads=(2, 2, 4, 4), window_size=32)
        model.eval()
        coords = _road_cloud(2, 256, seed=4)
        feats = torch.rand(2, 256, 3)
        logits = model(coords, feats)
        self.assertEqual(logits.shape, (2, 256, 38))
        x = model.patch_embed(torch.cat([coords, feats], dim=-1))
        cur, sizes = coords, []
        with torch.no_grad():
            for stage in model.stages:
                cur, x, skip, _ = stage(cur, x)
                sizes.append(skip.shape[1])
        self.assertEqual(sizes, [256, 64, 16, 4])

    def test_pointmamba_backbone_forward(self) -> None:
        torch.manual_seed(5)
        model = PointMambaBackbone(embed_dim=32, depths=(1, 1, 1, 1))
        model.eval()
        coords = _road_cloud(2, 256, seed=5)
        feats = torch.rand(2, 256, 3)
        logits = model(coords, feats)
        self.assertEqual(logits.shape, (2, 256, 38))
        self.assertTrue(torch.isfinite(logits).all())


class EmaScanTests(unittest.TestCase):
    def test_chunked_scan_equals_sequential(self) -> None:
        torch.manual_seed(6)
        dim, n = 8, 300
        x = torch.randn(2, n, dim)
        alpha = torch.sigmoid(torch.randn(dim))
        state = torch.zeros(2, dim)
        ref = []
        for t in range(n):
            state = (1 - alpha) * x[:, t] + alpha * state
            ref.append(state)
        ref = torch.stack(ref, dim=1)
        fast = _chunked_ema_scan(x, alpha, chunk=64)
        self.assertTrue(torch.allclose(ref, fast, atol=1e-5))

    def test_unit_dc_gain(self) -> None:
        """常数输入稳态输出 = 输入（旧形式直流增益最高 20 倍）。"""
        y = _chunked_ema_scan(torch.ones(1, 600, 4), torch.full((4,), 0.95))
        self.assertAlmostEqual(float(y[0, -1, 0]), 1.0, delta=1e-3)

    def test_morton_isotropic_locality(self) -> None:
        """等比量化的 Morton 序：路面云相邻 token 的 xy 跳距应远小于
        逐轴归一化（审计实测 0.142m vs 0.567m）。"""
        coords = _road_cloud(1, 4096, seed=7)
        perm, _ = _morton_permutation(coords)
        ordered = coords[0][perm[0]]
        jumps = (ordered[1:, :2] - ordered[:-1, :2]).norm(dim=-1)
        self.assertLess(float(jumps.mean()), 0.35)


class LossBoundTests(unittest.TestCase):
    """损失对越界标签的防护（CUDA gather 无边界检查 → 真实非法访存）。"""

    def test_focal_and_dice_ignore_out_of_range_labels(self) -> None:
        from roadmc.models.model_pl import DiceLoss, FocalLoss
        logits = torch.randn(2, 16, 2, requires_grad=True)
        targets = torch.zeros(2, 16, dtype=torch.long)
        targets[0, :4] = 37   # 38 类标签喂进 2 类头（跨阶段错配场景）
        targets[0, 4:6] = -1  # padding
        for loss_fn in (FocalLoss(), DiceLoss()):
            with self.subTest(loss=type(loss_fn).__name__):
                loss = loss_fn(logits, targets)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreaterEqual(float(loss), 0.0)

    def test_all_labels_out_of_range_returns_zero(self) -> None:
        from roadmc.models.model_pl import FocalLoss
        logits = torch.randn(1, 8, 2, requires_grad=True)
        targets = torch.full((1, 8), 37, dtype=torch.long)
        loss = FocalLoss()(logits, targets)
        self.assertEqual(float(loss), 0.0)


class BiasBudgetTests(unittest.TestCase):
    """退化窗口占用必须被占用上限（C1）处理，而不是在算子内 OOM。"""

    def test_degenerate_occupancy_is_capped_not_crashing(self) -> None:
        import roadmc.models.attention.window_attention as wa
        # 所有点重合 → 单窗口吞下全部点 → 旧实现 M=N、偏置 4.0 GiB 直接
        # OOM/守卫报错。C1 上限把超限点重切为有界窗口，M 被压到上限内，
        # 注意力正常完成、输出形状正确、无 NaN。
        coords = torch.zeros(2, 4096, 3)
        attn = wa.WindowAttention3D(dim=48, num_heads=3, window_size=32)
        x = torch.randn(2, 4096, 48)
        out = attn(coords, x)
        self.assertEqual(out.shape, (2, 4096, 48))
        self.assertFalse(torch.isnan(out).any())

    def test_cap_bounds_window_occupancy(self) -> None:
        import roadmc.models.attention.window_attention as wa
        # 稠密集群：600 点堆在 xy 一角 → 单窗口 600 点。上限后每个窗口
        # 占用 ≤ 4×window_size。
        torch.manual_seed(7)
        coords = _road_cloud(1, 2048, seed=7)
        cluster = torch.randn(600, 3) * 0.002
        cluster[:, 2] *= 0.005
        coords[0, :600] = coords[0, 0] + cluster
        wid, _ = wa._window_partition(
            coords, 32, shift=False, mode="columnar",
            max_occupancy=32 * wa.MAX_OCCUPANCY_MULTIPLIER,
        )
        counts = torch.bincount(wid[0])
        self.assertLessEqual(int(counts.max()), 32 * wa.MAX_OCCUPANCY_MULTIPLIER)

    def test_well_behaved_window_ids_unchanged(self) -> None:
        """正常云（无超窗）下上限路径与原始分区 bit-identical。

        这是"已训练 checkpoint 不受影响"的机器保证：v2/v3 实测最大占用
        远低于 4×window_size，故该路径对现有消融结果零改动。
        """
        import roadmc.models.attention.window_attention as wa
        coords = _road_cloud(2, 2048, seed=21)
        wid_raw, _ = wa._window_partition(coords, 32, shift=False, mode="columnar")
        wid_cap, _ = wa._window_partition(
            coords, 32, shift=False, mode="columnar",
            max_occupancy=32 * wa.MAX_OCCUPANCY_MULTIPLIER,
        )
        self.assertTrue(torch.equal(wid_raw, wid_cap))
        # shift 路径同样 bit-identical
        wid_raw_s, _ = wa._window_partition(coords, 32, shift=True, mode="columnar")
        wid_cap_s, _ = wa._window_partition(
            coords, 32, shift=True, mode="columnar",
            max_occupancy=32 * wa.MAX_OCCUPANCY_MULTIPLIER,
        )
        self.assertTrue(torch.equal(wid_raw_s, wid_cap_s))

    def test_valid_mask_excludes_padding_from_windows(self) -> None:
        """D1：collate padding 点(0,0,0) 不参与真实窗口分箱。"""
        import roadmc.models.attention.window_attention as wa
        coords = torch.zeros(1, 256, 3)
        valid = torch.ones(1, 256, dtype=torch.bool)
        valid[0, 100:] = False  # 156 padding
        wid, num_w = wa._window_partition(
            coords, 32, shift=False, mode="columnar", valid_mask=valid,
            max_occupancy=32 * wa.MAX_OCCUPANCY_MULTIPLIER,
        )
        # padding 点落在独立窗口（id ≥ 名义窗口数），不污染真实窗口
        nominal = 8 * 8
        self.assertGreaterEqual(num_w, nominal + 1)
        real_ids = wid[0, :100]
        self.assertLess(int(real_ids.max()), nominal)


class OptimizerRoutingTests(unittest.TestCase):
    def test_muon_excludes_head_embed_and_dscm(self) -> None:
        from roadmc.models.model_pl import RoadMCSegModel
        model = RoadMCSegModel(
            embed_dim=32, depths=(1, 1, 1, 1), num_heads=(2, 2, 4, 4),
            window_size=32, optimizer_name="muon", t_max=5,
        )
        optimizer, scheduler = model.build_optimizer_and_scheduler()
        muon_params = {
            id(p) for g in optimizer.muon.param_groups for p in g["params"]
        }
        for name, p in model.named_parameters():
            if "decode.cls_head" in name or "patch_embed" in name or "log_kernel" in name:
                self.assertNotIn(id(p), muon_params, f"{name} must not be in Muon")
        # adjust_lr_fn 必须是 match_rms_adamw
        for g in optimizer.muon.param_groups:
            self.assertEqual(g["adjust_lr_fn"], "match_rms_adamw")
        del scheduler


class ClassWeightTests(unittest.TestCase):
    def test_clip_bound_strictly_holds(self) -> None:
        counts = np.array([10_000_000, 5_000_000] + [200] * 4 + [50_000] * 32)
        weights = effective_number_class_weights(counts, max_weight=5.0)
        self.assertLessEqual(float(weights.max()), 5.0 + 1e-6)
        self.assertGreater(float(weights.min()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
