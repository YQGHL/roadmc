"""Physical-correctness regression tests for the LiDAR observation model.

每条断言对应 2026-07 学术审计的一项修复：
- 统一车载传感器位姿：测距噪声必须产生毫米级垂直分量（旧实现
  以路面角点为球心，垂直分量仅 ~7 µm）
- 辐射强度模型：几何依赖、类间分布重叠（消除标签查表泄漏）、
  泛油变暗、8-bit 量化
- 逐点 σ_r 异方差与辐射耦合丢点
- 扫描线距随距离二次增长（消费 vertical_fov_deg）
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("ROADMC_GENERATOR_NO_TORCH", "1")

from roadmc.data.synthetic.config import (  # noqa: E402
    GeneratorConfig,
    LidarNoiseConfig,
    MicroTextureConfig,
    RoadSurfaceConfig,
)
from roadmc.data.synthetic.generator import SyntheticRoadDataset  # noqa: E402
from roadmc.data.synthetic.primitives import (  # noqa: E402
    resample_to_lidar_pattern,
    simulate_lidar_noise,
)


def _flat_plane(n_side: int = 60, extent: float = 5.0) -> np.ndarray:
    xs, ys = np.meshgrid(
        np.linspace(0.0, extent, n_side), np.linspace(0.0, extent, n_side), indexing="ij"
    )
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n_side * n_side)])


def _bhattacharyya(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.sum(np.sqrt(pa * pb)))


class RangeNoiseGeometryTests(unittest.TestCase):
    """测距噪声必须沿真实波束方向投影。"""

    def test_vertical_noise_component_is_millimetre_scale(self) -> None:
        """车载位姿 (h=2m) 下 5mm 测距噪声的垂直分量应为毫米级。

        旧实现（传感器位于路面角点 z=0）实测垂直分量仅 ~7µm。
        """
        pts = _flat_plane()
        origin = np.array([2.5, -1.0, 2.0])
        noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.005, dropout_rate=0.0,
            angular_jitter_deg=0.0, seed=1, enable_edge_mixing=False,
            sensor_origin=origin,
        )
        dz_std_mm = float((noisy[:, 2] - pts[:, 2]).std()) * 1e3
        # E[|d_z/R|] 在该几何下约 0.3-0.6 → 期望 1.5-3 mm
        self.assertGreater(dz_std_mm, 0.5)
        self.assertLess(dz_std_mm, 5.0)

    def test_legacy_origin_still_produces_negligible_vertical_noise(self) -> None:
        """回归对照：旧原点几何的垂直分量确实近零（记录审计事实）。"""
        pts = _flat_plane()
        noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.005, dropout_rate=0.0,
            angular_jitter_deg=0.0, seed=1, enable_edge_mixing=False,
            sensor_origin=None,
        )
        dz_std_mm = float((noisy[:, 2] - pts[:, 2]).std()) * 1e3
        self.assertLess(dz_std_mm, 0.1)

    def test_heteroscedastic_sigma_r(self) -> None:
        """逐点 σ_r：两组点的位移标准差之比应跟随 σ 之比。"""
        pts = _flat_plane(40)
        n = len(pts)
        sigma = np.full(n, 0.001)
        sigma[n // 2:] = 0.01
        origin = np.array([2.5, -1.0, 2.0])
        noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.0, dropout_rate=0.0,
            angular_jitter_deg=0.0, seed=2, enable_edge_mixing=False,
            sensor_origin=origin, sigma_r=sigma,
        )
        disp = np.linalg.norm(noisy - pts, axis=1)
        ratio = disp[n // 2:].std() / max(disp[: n // 2].std(), 1e-12)
        self.assertGreater(ratio, 5.0)
        self.assertLess(ratio, 20.0)

    def test_dropout_couples_to_received_power(self) -> None:
        """低功率点必须丢得更多（低强度 ↔ 高丢点的真实耦合）。"""
        pts = _flat_plane(60)
        n = len(pts)
        weight = np.ones(n)
        weight[n // 2:] = 0.2  # 后一半是"暗"点
        origin = np.array([2.5, -1.0, 2.0])
        noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.0, dropout_rate=0.10,
            angular_jitter_deg=0.0, seed=3, enable_edge_mixing=False,
            sensor_origin=origin, drop_weight=weight,
        )
        # 用 y 坐标恢复归属（平面点，位移为零）
        survivors_bright = np.sum(np.isin(
            np.round(noisy[:, 0] * 1e6), np.round(pts[: n // 2, 0] * 1e6)
        ))
        # 更稳妥：按原始索引近邻匹配
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, idx = tree.query(noisy, k=1)
        kept = np.zeros(n, dtype=bool)
        kept[idx] = True
        rate_bright = 1.0 - kept[: n // 2].mean()
        rate_dim = 1.0 - kept[n // 2:].mean()
        self.assertGreater(rate_dim, rate_bright + 0.1)
        del survivors_bright


class RadiometricIntensityTests(unittest.TestCase):
    """强度 = ρ·shading·cosθ/R² 的可观测后果。"""

    def _scene(self, target_label=None, seed=123):
        config = GeneratorConfig(
            road=RoadSurfaceConfig(
                width=3.0, length=3.0, grid_res=0.02, roughness_class="A",
                crossfall=0.0,
            ),
            micro_texture=MicroTextureConfig(amplitude=0.0),
            lidar_noise=LidarNoiseConfig(
                distance_noise_std=0.0, dropout_rate=0.0,
                angular_jitter_deg=0.0, enable_edge_mixing=False,
            ),
            seed=seed,
            num_points=8192,
        )
        dataset = SyntheticRoadDataset(config=config, dataset_size=0)
        return dataset.generate_scene(0, target_label=target_label)

    def test_intensity_depends_on_geometry(self) -> None:
        """背景点强度必须随 cosθ/R² 单调（旧实现与几何完全无关）。"""
        scene = self._scene()
        # 反归一化坐标到米制
        pts = scene["points"] * scene["coordinate_scale"] + scene["coordinate_center"]
        bg = scene["labels"] == 0
        origin = np.array(
            scene["resolution_metadata"]["sensor_output"]["sensor_pose_m"]
        )
        d = pts[bg] - origin
        r2 = np.sum(d * d, axis=1)
        geom = 1.0 / r2  # 平坦场景 cosθ 变化远小于 R²
        intensity = scene["feats"][:, 0][bg]
        rho = np.corrcoef(
            np.argsort(np.argsort(geom)), np.argsort(np.argsort(intensity))
        )[0, 1]
        self.assertGreater(float(rho), 0.3)

    def test_crack_background_distributions_overlap(self) -> None:
        """裂缝与背景的强度直方图必须高度重叠（消除标签查表泄漏）。"""
        scene = self._scene(target_label=5)
        intensity = scene["feats"][:, 0]
        crack = intensity[(scene["labels"] >= 1) & (scene["labels"] <= 8)]
        background = intensity[scene["labels"] == 0]
        self.assertGreater(len(crack), 10)
        overlap = _bhattacharyya(crack, background)
        # 旧实现固定偏移 -0.05/σ0.02 → d'=2.5，重叠系数 ~0.45；
        # 辐射模型下类别只通过几何遮蔽间接进入，应高度重叠。
        self.assertGreater(overlap, 0.6)

    def test_bleeding_is_darker_than_background(self) -> None:
        """泛油方向修正：NIR 下油膜变暗（旧实现 +0.20 提亮为全场最亮）。"""
        scene = self._scene(target_label=19, seed=321)
        intensity = scene["feats"][:, 0]
        bleed = intensity[scene["labels"] == 19]
        background = intensity[scene["labels"] == 0]
        self.assertGreater(len(bleed), 10)
        self.assertLess(float(bleed.mean()), float(background.mean()))

    def test_intensity_is_quantized_8bit(self) -> None:
        scene = self._scene()
        levels = scene["feats"][:, 0] * 255.0
        np.testing.assert_allclose(levels, np.round(levels), atol=1e-3)

    def test_metadata_records_pose_and_model(self) -> None:
        scene = self._scene()
        sensor_output = scene["resolution_metadata"]["sensor_output"]
        self.assertEqual(len(sensor_output["sensor_pose_m"]), 3)
        self.assertGreater(sensor_output["sensor_pose_m"][2], 0.0)
        self.assertIn("radiometric.v2", sensor_output["intensity_model"])


class ScanPatternTests(unittest.TestCase):
    """扫描线密度必须消费 vertical_fov_deg 并随距离二次展宽。"""

    def test_scan_line_spacing_grows_with_range(self) -> None:
        pts = _flat_plane(n_side=200, extent=8.0)
        rng = np.random.default_rng(0)
        idx = resample_to_lidar_pattern(
            pts, scan_lines=24, vertical_fov_deg=50.0, scan_pattern="rotating",
            range_decay=0.0, incidence_angle_drop=0.0, rng=rng,
            sensor_pose=np.array([4.0, -1.0, 2.0]),
        )
        kept_y = pts[idx, 1]
        # 近端线距 ~0.1 m（近乎连续覆盖），远端线距 >1 m（大片真空）。
        # 用 2 cm 分箱的占用率对比检验各向异性随距离增长。
        def occupancy(lo: float, hi: float) -> float:
            hist, _ = np.histogram(kept_y, bins=int((hi - lo) / 0.02), range=(lo, hi))
            return float((hist > 0).mean())

        occ_near = occupancy(0.0, 2.0)
        occ_far = occupancy(6.0, 8.0)
        self.assertGreater(occ_near, occ_far + 0.3)
        # 远端必须存在线间真空（连续 >0.5 m 无点的间隙）
        far_y = np.sort(kept_y[kept_y > 5.0])
        self.assertGreater(len(far_y), 5)
        self.assertGreater(float(np.diff(far_y).max()), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
