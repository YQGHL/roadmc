"""JTG-criterion regression tests for damage-primitive geometry.

每条断言对应 2026-07 学术审计的一项修复：
- 线状裂缝严重度由缝宽（3 mm 判据）决定，measure-then-label
- 错台为板级位移：整个剖面只有一个台阶（无虚假反向台阶）
- 坑槽半径与严重度解耦（消除平面尺寸捷径）
- 波浪拥包空间包络（不再全场贴标）
- 车辙标签边界几何可观测（绝对阈值 ≥2 mm）
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.synthetic.config import (  # noqa: E402
    DiseaseConfig,
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    PotholeConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset  # noqa: E402
from roadmc.data.synthetic.primitives import (  # noqa: E402
    JTG_CRACK_WIDTH_THRESHOLD_M,
    add_concrete_damage,
    add_corrugation,
    add_crack,
    add_pothole,
    add_rutting,
)


def _flat_grid(extent_x: float, extent_y: float, res: float) -> np.ndarray:
    xs = np.arange(0.0, extent_x, res)
    ys = np.arange(0.0, extent_y, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)


class CrackWidthCriterionTests(unittest.TestCase):
    """裂缝轻/重 = 缝宽 3 mm 判据（JTG 5210-2018），非缝深。"""

    def _labeled_widths(self, severity: str, seed: int) -> tuple[np.ndarray, set]:
        """1 mm 网格上的横向裂缝：按 y 列量测标注带宽。"""
        pts = _flat_grid(0.6, 0.3, 0.001)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_crack(
            pts, labels, crack_type="transverse", severity=severity,
            params={"label_width_floor": 0.0}, seed=seed,
        )
        del out_pts
        widths = []
        xs = np.round(pts[:, 0], 6)
        for x_val in np.unique(xs)[::60]:
            col = xs == x_val
            labeled_y = pts[col & (out_lbl > 0), 1]
            if len(labeled_y) > 0:
                widths.append(labeled_y.max() - labeled_y.min() + 0.001)
        return np.asarray(widths), set(np.unique(out_lbl[out_lbl > 0]).tolist())

    def test_light_crack_width_at_most_3mm(self) -> None:
        widths, label_set = self._labeled_widths("light", seed=11)
        self.assertGreater(len(widths), 3)
        self.assertLessEqual(
            float(np.median(widths)), JTG_CRACK_WIDTH_THRESHOLD_M + 0.0015
        )
        self.assertEqual(label_set, {7})  # 轻横向裂缝

    def test_severe_crack_width_above_3mm(self) -> None:
        widths, label_set = self._labeled_widths("severe", seed=11)
        self.assertGreater(len(widths), 3)
        self.assertGreater(
            float(np.median(widths)), JTG_CRACK_WIDTH_THRESHOLD_M
        )
        self.assertEqual(label_set, {8})  # 重横向裂缝

    def test_severity_no_longer_separable_by_depth_alone(self) -> None:
        """轻/重的定义性差异是宽度侧别；深度只是弱相关参数。"""
        pts = _flat_grid(0.6, 0.3, 0.001)
        labels = np.zeros(len(pts), dtype=np.int64)
        _, lbl_light = add_crack(
            pts, labels, crack_type="transverse", severity="light",
            params={"label_width_floor": 0.0, "d_max": 0.030}, seed=5,
        )
        # d_max 被显式设为"重度深度"，标签仍必须按宽度判轻。
        self.assertEqual(set(np.unique(lbl_light[lbl_light > 0]).tolist()), {7})


class FaultingSlabTests(unittest.TestCase):
    """错台 = 整板一次跳变，无 ±40mm 处的虚假反向台阶。"""

    def test_single_step_profile(self) -> None:
        pts = _flat_grid(4.0, 6.0, 0.01)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_concrete_damage(
            pts, labels, damage_type="faulting", severity="severe",
            params={"slab_length": 3.0, "slab_width": 4.0, "d_max": 0.015},
            seed=3,
        )
        self.assertGreater(int((out_lbl > 0).sum()), 0)
        # 按 y 行取中位高程剖面，统计 |Δz| > 3mm 的跳变次数
        ys = np.round(pts[:, 1], 6)
        y_vals = np.unique(ys)
        profile = np.array([np.median(out_pts[ys == yv, 2]) for yv in y_vals])
        steps = np.abs(np.diff(profile))
        n_steps = int((steps > 0.003).sum())
        self.assertEqual(n_steps, 1)
        # 台阶量 = 完整 fault_offset
        self.assertAlmostEqual(float(steps.max()), 0.015, delta=0.002)


class PotholeDecoupleTests(unittest.TestCase):
    """坑槽半径与严重度解耦：轻/重的半径分布必须重叠。"""

    def test_radius_ranges_overlap(self) -> None:
        config = GeneratorConfig(
            road=RoadSurfaceConfig(
                width=3.0, length=3.0, grid_res=0.03, roughness_class="A",
                crossfall=0.0,
            ),
            micro_texture=MicroTextureConfig(amplitude=0.0),
            disease=DiseaseConfig(max_diseases_per_scene=1),
            lidar_noise=LidarNoiseConfig(
                distance_noise_std=0.0, dropout_rate=0.0,
                angular_jitter_deg=0.0, enable_edge_mixing=False,
            ),
            seed=77,
            num_points=8192,
        )
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        area_per_scene_pt = None

        def est_radius(scene, label):
            n = int((scene["labels"] == label).sum())
            frac = n / len(scene["labels"])
            return float(np.sqrt(frac * 9.0 / np.pi))  # 3×3 m 场景

        light_r = []
        severe_r = []
        for i in range(10):
            light_r.append(est_radius(dataset.generate_scene(i, target_label=9), 9))
            severe_r.append(est_radius(dataset.generate_scene(100 + i, target_label=10), 10))
        del area_per_scene_pt
        # 旧实现：轻 ≤0.15m、重 ≥0.15m 完全不相交；解耦后两组范围重叠。
        self.assertGreater(max(light_r), min(severe_r))


class CorrugationEnvelopeTests(unittest.TestCase):
    """波浪拥包是局部病害：标注比例必须远低于旧实现的 ~94%。"""

    def test_labeled_fraction_is_local(self) -> None:
        pts = _flat_grid(7.0, 5.0, 0.05)
        labels = np.zeros(len(pts), dtype=np.int64)
        fracs = []
        for seed in (1, 2, 3):
            _, lbl = add_corrugation(
                pts, labels, direction="transverse", wavelength=0.5,
                amplitude=0.02, severity="light", seed=seed,
            )
            fracs.append(float((lbl > 0).mean()))
        self.assertGreater(min(fracs), 0.001)
        self.assertLess(max(fracs), 0.5)


class CornerBreakChordTests(unittest.TestCase):
    """板角断裂 = 弦线切割的三角块，裂缝不通过角点。"""

    def test_labeled_region_is_corner_triangle(self) -> None:
        pts = _flat_grid(4.0, 6.0, 0.02)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_concrete_damage(
            pts, labels, damage_type="corner_break", severity="severe",
            params={"slab_length": 3.0, "slab_width": 4.0,
                    "chord_frac_x": 0.4, "chord_frac_y": 0.4, "d_max": 0.015},
            seed=7,
        )
        labeled = out_lbl > 0
        self.assertGreater(int(labeled.sum()), 20)
        # 标注区应能被某个板角的 0.4/0.4 弦线三角形（含槽带边距）覆盖
        lx, ly = pts[labeled, 0], pts[labeled, 1]
        corners = [(cx, cy) for cx in (0.0, 4.0) for cy in (0.0, 3.0, 6.0)]
        margin = 0.08
        best_cover = 0.0
        for cx, cy in corners:
            sx = 1.0 if cx == 0.0 else -1.0
            sy = 1.0 if cy in (0.0, 3.0) else -1.0
            # 三角形: |x-cx|/(0.4·4) + |y-cy|/(0.4·3) ≤ 1（加边距）
            u = np.abs(lx - cx) / (0.4 * 4.0 + margin)
            v = np.abs(ly - cy) / (0.4 * 3.0 + margin)
            inside = (u + v <= 1.0 + margin) & (sx * (lx - cx) >= -margin) & (
                sy * (ly - cy) >= -margin
            )
            best_cover = max(best_cover, float(inside.mean()))
        self.assertGreater(best_cover, 0.9)
        # 三角块整体沉降：标注点位移非零且量级 = d_max 的一半以上
        dz = pts[labeled, 2] - out_pts[labeled, 2]
        self.assertGreater(float(np.median(dz)), 0.005)
        del sx, sy


class AlligatorLocalizationTests(unittest.TestCase):
    """龟裂是局部 patch，不再撒满全场。"""

    def test_alligator_confined_to_region(self) -> None:
        pts = _flat_grid(7.0, 5.0, 0.03)
        labels = np.zeros(len(pts), dtype=np.int64)
        _, lbl = add_crack(
            pts, labels, crack_type="alligator", severity="severe",
            params={"label_width_floor": 0.03, "region_center": (2.0, 2.0),
                    "region_width": 1.0, "region_length": 3.0},
            seed=5,
        )
        labeled = lbl > 0
        self.assertGreater(int(labeled.sum()), 10)
        lx, ly = pts[labeled, 0], pts[labeled, 1]
        self.assertGreater(float(lx.min()), 2.0 - 0.5 - 0.15)
        self.assertLess(float(lx.max()), 2.0 + 0.5 + 0.15)
        self.assertGreater(float(ly.min()), 2.0 - 1.5 - 0.15)
        self.assertLess(float(ly.max()), 2.0 + 1.5 + 0.15)
        self.assertLess(float(labeled.mean()), 0.25)


class WheelPathPriorTests(unittest.TestCase):
    def test_samples_concentrate_on_wheel_paths(self) -> None:
        from roadmc.data.synthetic.generator import SyntheticRoadDataset
        rng = np.random.default_rng(0)
        xs = np.array([
            SyntheticRoadDataset._sample_wheelpath_x(rng, 7.0) for _ in range(500)
        ])
        dist_to_path = np.minimum(np.abs(xs - (3.5 - 0.9)), np.abs(xs - (3.5 + 0.9)))
        self.assertLess(float(np.median(dist_to_path)), 0.3)
        self.assertGreater(float((dist_to_path < 0.6).mean()), 0.85)


class GeometricOcclusionTests(unittest.TestCase):
    """深窄裂缝内部点被遮挡，宽坑槽内部可见。"""

    def test_narrow_trench_occluded_wide_bowl_visible(self) -> None:
        from roadmc.data.synthetic.primitives import (
            geometric_occlusion_keep_mask,
            local_depression_depth,
        )
        pts = _flat_grid(5.0, 5.0, 0.01)
        # 窄槽：x=2.50 单列下凹 30mm（开口 ~1cm）
        trench = np.abs(pts[:, 0] - 2.50) < 0.004
        pts[trench, 2] = -0.030
        # 宽碗：以 (1.0, 1.0) 为心、半径 0.3 的盘下凹 30mm
        bowl = np.hypot(pts[:, 0] - 1.0, pts[:, 1] - 1.0) < 0.3
        pts[bowl, 2] = -0.030

        depth = local_depression_depth(pts)
        rng = np.random.default_rng(3)
        keep = geometric_occlusion_keep_mask(
            pts, np.array([2.5, -1.0, 2.0]), depth, rng
        )
        trench_keep = float(keep[trench].mean())
        bowl_core = bowl & (np.hypot(pts[:, 0] - 1.0, pts[:, 1] - 1.0) < 0.2)
        bowl_keep = float(keep[bowl_core].mean())
        self.assertLess(trench_keep, 0.5)
        self.assertGreater(bowl_keep, 0.85)


class RuttingObservabilityTests(unittest.TestCase):
    """车辙标签边界必须几何可观测（≥2 mm），带宽受物理约束。"""

    def test_label_boundary_deformation_observable(self) -> None:
        pts = _flat_grid(7.0, 5.0, 0.01)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, out_lbl = add_rutting(
            pts, labels, center_line=3.5, wheel_separation=1.8,
            depth=0.012, width=0.6, severity="light", seed=1,
        )
        labeled = out_lbl > 0
        self.assertGreater(int(labeled.sum()), 0)
        dz = pts[:, 2] - out_pts[:, 2]
        self.assertGreaterEqual(float(dz[labeled].min()), 0.002 - 1e-9)
        # 单轮迹标注带宽：解析上 2σ√(2ln(depth/2mm)) ≈ 0.76 m @ 12mm
        left_track = labeled & (pts[:, 0] < 3.5)
        band = pts[left_track, 0]
        self.assertLess(float(band.max() - band.min()), 0.85)


class PotholeBetaRangeTests(unittest.TestCase):
    """坑槽超椭圆指数 β 由 beta_range 配置驱动（G06-D1/D2 修复）。

    轻度恒为 β=2（椭球）；重度按 beta_range 采样（β>2 平底）。
    β 越大，r=0.5R 处相对深度 |z|/d 越大（底部更平）。
    """

    def _depth_ratio(self, severity: str, beta_range: tuple[float, float] | None,
                     radius: float = 0.2, depth: float = 0.05) -> float:
        pts = _flat_grid(1.0, 1.0, 0.005)
        labels = np.zeros(len(pts), dtype=np.int64)
        out_pts, _ = add_pothole(
            pts, labels, center=(0.5, 0.5), radius=radius, depth=depth,
            edge_quality=1.0, severity=severity, seed=0, beta_range=beta_range,
        )
        r = np.hypot(out_pts[:, 0] - 0.5, out_pts[:, 1] - 0.5)
        ring = (r > 0.099) & (r < 0.101)  # r/R ≈ 0.5
        return float(np.median(out_pts[ring, 2]) / depth)

    def test_light_pothole_is_ellipsoid_beta2(self) -> None:
        # 椭球剖面 z(r=0.5R) = -d·(1-0.5²)^0.5 = -0.866d
        self.assertAlmostEqual(self._depth_ratio("light", None), -0.866, delta=0.04)

    def test_beta_range_governs_severe_profile(self) -> None:
        # β 越大底部越平 → 0.5R 处更负
        d3 = self._depth_ratio("severe", (3.0, 3.0))
        d5 = self._depth_ratio("severe", (5.0, 5.0))
        self.assertLess(d5, d3)
        # 理论：β=3 → -(1-0.5³)^(1/3) = -0.956；β=5 → -(1-0.5⁵)^0.2 = -0.994
        self.assertAlmostEqual(d3, -0.956, delta=0.04)
        self.assertAlmostEqual(d5, -0.994, delta=0.04)

    def test_invalid_beta_range_rejected(self) -> None:
        # 直调 primitive 也防御：lo<=0 或 lo>hi 直接拒绝，避免 β∈(0,1)
        # 产生坑心 cusp 或反序区间静默出错。
        pts = _flat_grid(0.4, 0.4, 0.01)
        labels = np.zeros(len(pts), dtype=np.int64)
        for bad in ((0.0, 3.0), (4.0, 3.0)):
            with self.assertRaises(ValueError):
                add_pothole(
                    pts, labels, center=(0.2, 0.2), radius=0.1, depth=0.03,
                    edge_quality=1.0, severity="severe", seed=0,
                    beta_range=bad,
                )

    def test_config_beta_range_validated(self) -> None:
        # 死配置修复：beta_range 成为有效参数，非法区间在构造时拒绝。
        with self.assertRaises(ValueError):
            PotholeConfig(beta_range=(5.0, 3.0))
        self.assertEqual(PotholeConfig().beta_range, (3.0, 5.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
