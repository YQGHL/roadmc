"""
RoadMC 合成数据集生成器 —— SyntheticRoadDataset.
==================================================

组合 config.py 的配置参数和 primitives.py 的基元函数，
生成符合 JTG 5210-2018 标准的合成道路点云数据集。

数据流 (Data Pipeline)：
  1. 确定路面类型 (asphalt / concrete)
  2. ISO 8608 PSD 路面宏观轮廓生成
  3. fBm 微观纹理叠加
  4. 病害随机选择与组合应用
  5. 特征计算 (强度反射率 + 曲率)
  6. LiDAR 噪声仿真
  7. 标签/特征最近邻传递
  8. 定点数重采样 & 归一化
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from .config import (
        NUM_CLASSES,
        GeneratorConfig,
    )
    from .primitives import (
        generate_road_surface,
        add_micro_texture,
        add_crack,
        add_pothole,
        add_raveling,
        add_depression,
        add_rutting,
        add_corrugation,
        add_bleeding,
        add_concrete_damage,
        simulate_lidar_noise,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        NUM_CLASSES,
        GeneratorConfig,
    )
    from primitives import (  # type: ignore[no-redef]
        generate_road_surface,
        add_micro_texture,
        add_crack,
        add_pothole,
        add_raveling,
        add_depression,
        add_rutting,
        add_corrugation,
        add_bleeding,
        add_concrete_damage,
        simulate_lidar_noise,
    )

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CONCRETE_DAMAGE_TYPES: List[str] = [
    "slab_shatter",
    "slab_crack",
    "corner_break",
    "faulting",
    "pumping",
    "edge_spall",
    "joint_damage",
    "pitting",
    "blowup",
    "exposed_aggregate",
]

# 无严重程度区分的混凝土损坏类型（标签直接固定）
CONCRETE_NO_SEVERITY: set = {"pumping", "pitting", "blowup", "exposed_aggregate"}

# 沥青路面可用的病害 key（对应 disease_probs 字典）
ASPHALT_DISEASE_KEYS: List[str] = [
    "crack",
    "pothole",
    "raveling",
    "depression",
    "rutting",
    "corrugation",
    "bleeding",
]

# 病害应用的自然顺序（大尺度 → 小尺度）
DISEASE_APPLY_ORDER: Dict[str, int] = {
    "corrugation": 0,
    "rutting": 1,
    "depression": 2,
    "pothole": 3,
    "crack": 4,
    "bleeding": 5,
    "raveling": 6,
    "concrete_damage": 0,
}

# ===========================================================================
# SyntheticRoadDataset
# ===========================================================================


class SyntheticRoadDataset(torch.utils.data.Dataset):
    """JTG 5210-2018 合成道路点云数据集。

    使用 :class:`GeneratorConfig` 作为唯一配置入口，组合 config.py 的
    参数和 primitives.py 的物理基元，生成带有病害标注的道路点云场景。

    ``__getitem__`` 返回的字典包含:

    - **points**: ``(N, 3)`` torch.float32 — 归一化后的点云坐标
    - **labels**: ``(N,)`` torch.int64 — JTG 5210-2018 标签 (0-37)
    - **feats**: ``(N, 2)`` torch.float32 — [强度反射率, 局部曲率]
    - **normals**: ``(N, 3)`` torch.float32 — 表面单位法向量
    - **pavement_type**: str — 'asphalt' | 'concrete'

    Args:
        config: 顶层配置实例。
        dataset_size: 数据集总样本数，默认 2000。
    """

    def __init__(
        self,
        config: GeneratorConfig,
        dataset_size: int = 2000,
    ) -> None:
        self.config = config
        self.dataset_size = dataset_size

    # ------------------------------------------------------------------
    # 标准 Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """生成第 ``idx`` 个场景并返回 torch 张量字典。"""
        scene = self.generate_scene(idx)
        return {
            "points": torch.from_numpy(scene["points"]).float(),
            "labels": torch.from_numpy(scene["labels"]).long(),
            "feats": torch.from_numpy(scene["feats"]).float(),
            "normals": torch.from_numpy(scene["normals"]).float(),
            "pavement_type": scene["pavement_type"],
        }

    # ------------------------------------------------------------------
    # 核心场景生成
    # ------------------------------------------------------------------

    def generate_scene(self, idx: int) -> Dict:
        """生成一个完整的合成道路场景。

        内部流程按以下顺序执行：

        1. 选择路面类型 (asphalt / concrete)
        2. ISO 8608 PSD 路面轮廓 + fBm 微观纹理
        3. 病害组合随机选择 & 应用（按 DISEASE_APPLY_ORDER 排序）
        4. 网格曲率计算（LiDAR 噪声前，利用网格结构）
        5. 强度反射率计算（基于 pavement_type + 病害标签）
        6. 松散 (raveling) 产生 NaN 点 → 过滤
        7. LiDAR 噪声仿真（球坐标噪声 + dropout + edge mixing）
        8. KDTree 最近邻传递标签/特征/法向量
        9. 重采样至 ``num_points`` 点
        10. 可选归一化到单位球

        Args:
            idx: 场景索引，用于种子生成 (config.seed + idx)。

        Returns:
            包含 points, labels, feats, normals, pavement_type 的字典。
        """
        # --- 种子管理 ----------------------------------------------------
        if self.config.seed is not None:
            scene_seed = self.config.seed + idx
        else:
            scene_seed = None
        rng = np.random.default_rng(scene_seed)

        grid_res = self.config.road.grid_res
        width = self.config.road.width
        length = self.config.road.length

        # 网格尺寸 — 必须与 generate_road_surface 内部一致
        # np.arange(0, width, grid_res) 可能截断不同，所以从实际点数推断
        x_tmp = np.arange(0, width, grid_res)
        y_tmp = np.arange(0, length, grid_res)
        nx = len(x_tmp)
        ny = len(y_tmp)

        # ================================================================
        # 1. 路面类型选择
        # ================================================================
        pavement_type = self._select_pavement_type(rng)

        # ================================================================
        # 2. 路面宏观轮廓 + 微观纹理
        # ================================================================
        points, normals = generate_road_surface(
            width=width,
            length=length,
            grid_res=grid_res,
            pavement_type=pavement_type,
            roughness_class=self.config.road.roughness_class,
            seed=int(rng.integers(0, 2 ** 31)),
        )

        if self.config.micro_texture.amplitude > 0.0:
            points, normals = add_micro_texture(
                points,
                normals,
                amplitude=self.config.micro_texture.amplitude,
                hurst=self.config.micro_texture.hurst,
                octaves=self.config.micro_texture.octaves,
                seed=int(rng.integers(0, 2 ** 31)),
            )

        # ================================================================
        # 3. 标签初始化 + 病害选择 & 应用
        # ================================================================
        labels = np.zeros(points.shape[0], dtype=np.int64)

        diseases = self._select_diseases(rng, pavement_type)
        # 按自然顺序排序（大尺度 → 小尺度）
        diseases.sort(key=lambda d: DISEASE_APPLY_ORDER.get(d[0], 99))

        # 分离 raveling（最后处理，不参与曲率计算）
        raveling_entry = None
        non_raveling_diseases = []
        for d in diseases:
            if d[0] == "raveling":
                raveling_entry = d
            else:
                non_raveling_diseases.append(d)

        for disease_type, severity in non_raveling_diseases:
            seed = int(rng.integers(0, 2 ** 31))

            if disease_type == "crack":
                crack_type = str(rng.choice(self.config.crack.crack_types))
                params: dict = {
                    "d_max": 0.010 if severity == "light" else 0.030,
                }
                points, labels = add_crack(
                    points, labels,
                    crack_type=crack_type, severity=severity,
                    params=params, seed=seed,
                )

            elif disease_type == "pothole":
                cx = float(rng.uniform(0.5, width - 0.5))
                cy = float(rng.uniform(0.5, length - 0.5))
                if severity == "light":
                    radius = float(rng.uniform(0.05, self.config.pothole.max_radius_light))
                    depth = float(rng.uniform(0.005, self.config.pothole.max_depth_light))
                else:
                    radius = float(rng.uniform(
                        self.config.pothole.max_radius_light,
                        self.config.pothole.max_radius_severe,
                    ))
                    depth = float(rng.uniform(
                        self.config.pothole.max_depth_light * 1.1,
                        self.config.pothole.max_depth_severe,
                    ))
                edge_quality = float(rng.uniform(0.5, 1.0))
                points, labels = add_pothole(
                    points, labels,
                    center=(cx, cy), radius=radius, depth=depth,
                    edge_quality=edge_quality, severity=severity,
                    seed=seed,
                )

            elif disease_type == "depression":
                cx = float(rng.uniform(0.5, width - 0.5))
                cy = float(rng.uniform(0.5, length - 0.5))
                radius = float(rng.uniform(0.5, self.config.depression.max_radius))
                if severity == "light":
                    depth = float(rng.uniform(*self.config.depression.depth_light))
                else:
                    depth = float(rng.uniform(*self.config.depression.depth_severe))
                points, labels = add_depression(
                    points, labels,
                    center=(cx, cy), radius=radius,
                    depth=depth, severity=severity, seed=seed,
                )

            elif disease_type == "rutting":
                center_line = width / 2.0
                wheel_sep = self.config.rutting.wheel_separation
                if severity == "light":
                    depth = float(rng.uniform(0.005, self.config.rutting.max_depth_light))
                else:
                    depth = float(rng.uniform(
                        self.config.rutting.max_depth_light * 1.1,
                        self.config.rutting.max_depth_severe,
                    ))
                rut_width = self.config.rutting.rut_width
                points, labels = add_rutting(
                    points, labels,
                    center_line=center_line,
                    wheel_separation=wheel_sep,
                    depth=depth, width=rut_width,
                    severity=severity, seed=seed,
                )

            elif disease_type == "corrugation":
                direction = str(rng.choice(["longitudinal", "transverse"]))
                wavelength = float(rng.uniform(*self.config.corrugation.wavelength_range))
                if severity == "light":
                    amplitude = float(rng.uniform(*self.config.corrugation.amplitude_light))
                else:
                    amplitude = float(rng.uniform(*self.config.corrugation.amplitude_severe))
                points, labels = add_corrugation(
                    points, labels,
                    direction=direction, wavelength=wavelength,
                    amplitude=amplitude, severity=severity, seed=seed,
                )

            elif disease_type == "bleeding":
                lane_center = rng.uniform(0.3, 0.7) * width
                region_mask = (
                    np.abs(points[:, 0] - lane_center)
                    < self.config.bleeding.region_width * width / 2.0
                )
                labels = add_bleeding(
                    points, labels,
                    region_mask=region_mask, seed=seed,
                )

            elif disease_type == "concrete_damage":
                damage_type = str(rng.choice(CONCRETE_DAMAGE_TYPES))
                if damage_type in CONCRETE_NO_SEVERITY:
                    sev = "-"
                else:
                    sev = severity
                params = {
                    "slab_length": self.config.concrete_damage.slab_length,
                    "slab_width": self.config.concrete_damage.slab_width,
                    "joint_width": self.config.concrete_damage.joint_width,
                }
                points, labels = add_concrete_damage(
                    points, labels,
                    damage_type=damage_type, severity=sev,
                    params=params, seed=seed,
                )

        # ================================================================
        # 4. 网格曲率计算（利用网格结构，在 NaN / LiDAR 噪声之前）
        # ================================================================
        z_grid = points[:, 2].reshape(nx, ny).copy()
        z_grid = np.nan_to_num(z_grid, nan=0.0)
        curvature_grid = _compute_grid_curvature(z_grid, grid_res)
        curvature = curvature_grid.ravel().astype(np.float32)

        # ================================================================
        # 5. 强度反射率计算
        # ================================================================
        intensity = self._compute_intensity(labels, pavement_type, rng)

        # ================================================================
        # 6. 应用松散 (raveling) — 产生 NaN 点
        # ================================================================
        if raveling_entry is not None:
            disease_type, severity = raveling_entry
            seed = int(rng.integers(0, 2 ** 31))
            # 生成随机矩形区域掩码
            x_frac_min = float(rng.uniform(0.05, 0.35))
            x_frac_max = x_frac_min + float(rng.uniform(0.2, 0.5))
            y_frac_min = float(rng.uniform(0.05, 0.35))
            y_frac_max = y_frac_min + float(rng.uniform(0.2, 0.5))
            x_min_r = np.min(points[:, 0]) + x_frac_min * (np.max(points[:, 0]) - np.min(points[:, 0]))
            x_max_r = np.min(points[:, 0]) + x_frac_max * (np.max(points[:, 0]) - np.min(points[:, 0]))
            y_min_r = np.min(points[:, 1]) + y_frac_min * (np.max(points[:, 1]) - np.min(points[:, 1]))
            y_max_r = np.min(points[:, 1]) + y_frac_max * (np.max(points[:, 1]) - np.min(points[:, 1]))
            region_mask = (
                (points[:, 0] >= x_min_r) & (points[:, 0] <= x_max_r)
                & (points[:, 1] >= y_min_r) & (points[:, 1] <= y_max_r)
            )
            points, labels = add_raveling(
                points, labels,
                region_mask=region_mask, severity=severity, seed=seed,
            )

        # --- 移除 NaN 点（来自 raveling）---------------------------------
        valid_mask = ~np.any(np.isnan(points), axis=1)
        if np.any(valid_mask):
            points = points[valid_mask]
            labels = labels[valid_mask]
            normals = normals[valid_mask]
            intensity = intensity[valid_mask]
            curvature = curvature[valid_mask]
        else:
            raise RuntimeError("All points removed by raveling — no valid geometry remains.")

        # 确保非空
        if points.shape[0] == 0:
            raise RuntimeError("Empty point cloud after NaN filtering.")

        # ================================================================
        # 7. LiDAR 噪声仿真
        # ================================================================
        # 保存噪声前状态以传递标签/特征
        pre_points = points.copy()
        pre_labels = labels.copy()
        pre_intensity = intensity.copy()
        pre_curvature = curvature.copy()
        pre_normals = normals.copy()

        noisy_points = simulate_lidar_noise(
            points,
            distance_noise_std=self.config.lidar_noise.distance_noise_std,
            dropout_rate=self.config.lidar_noise.dropout_rate,
            angular_jitter_deg=self.config.lidar_noise.angular_jitter_deg,
            seed=int(rng.integers(0, 2 ** 31)),
        )

        if noisy_points.shape[0] == 0:
            raise RuntimeError("All points dropped during LiDAR noise simulation.")

        # ================================================================
        # 8. 最近邻传递标签/特征/法向量
        # ================================================================
        from scipy.spatial import KDTree
        tree = KDTree(pre_points)
        _, nn_idx = tree.query(noisy_points)

        labels = pre_labels[nn_idx]
        intensity = pre_intensity[nn_idx]
        curvature = pre_curvature[nn_idx]
        normals = pre_normals[nn_idx]

        # ================================================================
        # 9. 重采样至 num_points
        # ================================================================
        points_final, labels_final, intensity_final, curvature_final, normals_final = (
            self._resample_to_target(
                noisy_points, labels, intensity, curvature, normals,
                self.config.num_points, rng,
            )
        )

        # ================================================================
        # 10. 坐标归一化
        # ================================================================
        if self.config.normalize:
            points_final = self._normalize(points_final)

        # ================================================================
        # 特征拼接
        # ================================================================
        feats = np.stack([intensity_final, curvature_final], axis=1).astype(np.float32)

        return {
            "points": points_final.astype(np.float32),
            "labels": labels_final.astype(np.int64),
            "feats": feats,
            "normals": normals_final.astype(np.float32),
            "pavement_type": pavement_type,
        }

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _select_pavement_type(self, rng: np.random.Generator) -> str:
        """根据配置的病害概率选择最合适的路面类型。

        如果配置中只有沥青病害概率 > 0，返回 'asphalt'；
        如果只有水泥病害概率 > 0，返回 'concrete'；
        否则等概率随机选择。

        Returns:
            'asphalt' 或 'concrete'。
        """
        disease_probs = self.config.disease.disease_probs
        has_asphalt = any(disease_probs.get(k, 0) > 0 for k in
                         ["crack", "pothole", "raveling", "depression",
                          "rutting", "corrugation", "bleeding"])
        has_concrete = disease_probs.get("concrete_damage", 0) > 0

        config_type = self.config.road.pavement_type
        if config_type == "asphalt":
            return "asphalt"
        elif config_type == "concrete":
            return "concrete"
        elif has_asphalt and not has_concrete:
            return "asphalt"
        elif has_concrete and not has_asphalt:
            return "concrete"
        else:
            return str(rng.choice(["asphalt", "concrete"]))

    def _select_diseases(
        self,
        rng: np.random.Generator,
        pavement_type: str,
    ) -> List[Tuple[str, str]]:
        """根据路面类型和配置概率随机选择病害组合。

        Args:
            rng: 随机数生成器。
            pavement_type: 路面类型。

        Returns:
            [(disease_type, severity), ...] 列表。
        """
        disease_probs = self.config.disease.disease_probs
        max_diseases = self.config.disease.max_diseases_per_scene
        severity_ratio = self.config.disease.severity_ratio

        if pavement_type == "asphalt":
            available = ASPHALT_DISEASE_KEYS
        else:
            available = ["concrete_damage"]

        # 按概率独立选择
        selected: List[str] = []
        for disease in available:
            prob = disease_probs.get(disease, 0.0)
            if prob > 0.0 and rng.random() < prob:
                selected.append(disease)

        # 限制最大数量
        if len(selected) > max_diseases:
            selected = list(rng.choice(selected, size=max_diseases, replace=False))

        # 分配严重程度
        result: List[Tuple[str, str]] = []
        for disease in selected:
            if disease == "bleeding":
                # 泛油无严重程度区分
                result.append((disease, "light"))
            else:
                severity = "light" if rng.random() < severity_ratio else "severe"
                result.append((disease, severity))

        return result

    def _compute_intensity(
        self,
        labels: np.ndarray,
        pavement_type: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """模拟 LiDAR 强度反射率。

        每种路面类型有基础反射率范围，病害区域在此基础上叠加偏移量。

        Args:
            labels: 标签数组 (N,)。
            pavement_type: 路面类型。
            rng: 随机数生成器。

        Returns:
            强度值 (N,) float32，范围 [0, 1]。
        """
        N = labels.shape[0]
        if pavement_type == "asphalt":
            base = float(rng.uniform(0.3, 0.7))
        else:
            base = float(rng.uniform(0.5, 0.9))

        intensity = np.full(N, base, dtype=np.float64)

        # 病害区域强度修正
        intensity[labels == 19] += self.config.bleeding.intensity_boost  # 泛油
        intensity[(labels == 11) | (labels == 12)] -= 0.15  # 松散
        intensity[labels == 36] -= 0.20  # 露骨
        intensity[(labels == 20) | (labels == 37)] += 0.10  # 修补
        intensity[labels == 29] += 0.20  # 唧泥（湿润）
        intensity[labels == 34] -= 0.10  # 坑洞

        return np.clip(intensity, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _resample_to_target(
        points: np.ndarray,
        labels: np.ndarray,
        intensity: np.ndarray,
        curvature: np.ndarray,
        normals: np.ndarray,
        target_num: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """重采样到目标点数。

        多于目标则随机下采样，少于目标则随机重复补齐。

        Args:
            points: 点云 (N, 3)。
            labels: 标签 (N,)。
            intensity: 强度 (N,)。
            curvature: 曲率 (N,)。
            normals: 法向量 (N, 3)。
            target_num: 目标点数。
            rng: 随机数生成器。

        Returns:
            (points, labels, intensity, curvature, normals) 均重采样至 target_num。
        """
        N = points.shape[0]
        if N == target_num:
            return points, labels, intensity, curvature, normals

        if N > target_num:
            idx = rng.choice(N, size=target_num, replace=False)
        else:
            # 点数不足时重复随机点
            n_extra = target_num - N
            extra_idx = rng.choice(N, size=n_extra, replace=True)
            idx = np.concatenate([np.arange(N), extra_idx])

        return (
            points[idx],
            labels[idx],
            intensity[idx],
            curvature[idx],
            normals[idx],
        )

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        """归一化点云到单位球：平移到原点，缩放至最大半径为 1。

        Args:
            points: 点云 (N, 3)。

        Returns:
            归一化后点云 (N, 3)。
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        max_radius = np.max(np.linalg.norm(centered, axis=1))
        if max_radius > 1e-12:
            centered /= max_radius
        return centered


# ===========================================================================
# 工具函数
# ===========================================================================


def _compute_grid_curvature(z_grid: np.ndarray, grid_res: float) -> np.ndarray:
    """在规则网格上计算局部曲率（Laplacian of z）。

    使用中心有限差分计算 z 的 Laplacian ∇²z = ∂²z/∂x² + ∂²z/∂y²，
    该量在小坡度近似下正比于平均曲率。

    Args:
        z_grid: 高度场 (nx, ny)。
        grid_res: 网格分辨率 (m)。

    Returns:
        曲率场 (nx, ny)，与 z_grid 同形状。
    """
    nx, ny = z_grid.shape
    curvature = np.zeros_like(z_grid, dtype=np.float64)

    # 内部点：二阶中心差分
    dx2 = grid_res * grid_res
    curvature[1:-1, 1:-1] = (
        (z_grid[2:, 1:-1] - 2.0 * z_grid[1:-1, 1:-1] + z_grid[:-2, 1:-1]) / dx2
        + (z_grid[1:-1, 2:] - 2.0 * z_grid[1:-1, 1:-1] + z_grid[1:-1, :-2]) / dx2
    )

    # 边界：一阶差分近似
    curvature[0, :] = (z_grid[1, :] - z_grid[0, :]) / dx2
    curvature[-1, :] = (z_grid[-1, :] - z_grid[-2, :]) / dx2
    curvature[:, 0] = (z_grid[:, 1] - z_grid[:, 0]) / dx2
    curvature[:, -1] = (z_grid[:, -1] - z_grid[:, -2]) / dx2

    return curvature


# ===========================================================================
# 自检脚本
# ===========================================================================

if __name__ == "__main__":
    """自检：生成 5 个场景并验证输出。"""
    print("=" * 72)
    print("  SyntheticRoadDataset Self-Test")
    print("  生成 5 个场景并验证输出形状 / 标签范围 / 归一化")
    print("=" * 72)

    # 小尺寸测试配置（加快测试速度）
    from config import GeneratorConfig, RoadSurfaceConfig, DiseaseConfig

    cfg = GeneratorConfig(
        road=RoadSurfaceConfig(
            width=0.5,
            length=0.5,
            grid_res=0.02,
        ),
        disease=DiseaseConfig(
            max_diseases_per_scene=2,
            severity_ratio=0.6,
        ),
        seed=42,
        num_points=1024,
        normalize=True,
    )

    dataset = SyntheticRoadDataset(config=cfg, dataset_size=5)
    passed = 0
    total = 5

    print(f"\n数据集大小: {len(dataset)}")
    print(f"num_points: {cfg.num_points}")
    print(f"normalize: {cfg.normalize}")
    print(f"seed: {cfg.seed}")

    pavement_counts: Dict[str, int] = {"asphalt": 0, "concrete": 0}
    all_labels: set = set()

    for i in range(len(dataset)):
        print(f"\n--- Scene {i} ---")
        try:
            data = dataset[i]

            # 验证返回值类型
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            for key in ("points", "labels", "feats", "normals", "pavement_type"):
                assert key in data, f"Missing key: {key}"

            pts = data["points"]
            lbl = data["labels"]
            feats = data["feats"]
            nrm = data["normals"]
            ptype = data["pavement_type"]

            # 验证形状
            N = pts.shape[0]
            assert N == cfg.num_points, (
                f"points.shape[0]={N} != {cfg.num_points}"
            )
            assert pts.shape == (N, 3), f"points shape: {pts.shape}"
            assert lbl.shape == (N,), f"labels shape: {lbl.shape}"
            assert feats.shape == (N, 2), f"feats shape: {feats.shape}"
            assert nrm.shape == (N, 3), f"normals shape: {nrm.shape}"
            assert ptype in ("asphalt", "concrete"), f"pavement_type: {ptype}"

            # 验证类型
            assert pts.dtype == torch.float32, f"points dtype: {pts.dtype}"
            assert lbl.dtype == torch.int64, f"labels dtype: {lbl.dtype}"
            assert feats.dtype == torch.float32, f"feats dtype: {feats.dtype}"
            assert nrm.dtype == torch.float32, f"normals dtype: {nrm.dtype}"

            # 验证标签范围
            unique_labels = torch.unique(lbl).tolist()
            for ul in unique_labels:
                assert 0 <= ul < NUM_CLASSES, (
                    f"Label {ul} out of range [0, {NUM_CLASSES - 1}]"
                )
            all_labels.update(unique_labels)

            # 验证归一化
            if cfg.normalize:
                centered = pts - pts.mean(dim=0)
                max_r = torch.max(torch.norm(centered, dim=1))
                assert max_r <= 1.0 + 1e-4, (
                    f"max radius after normalize: {max_r:.6f}"
                )

            # 验证非零
            assert not torch.isnan(pts).any(), "NaN in points"
            assert not torch.isinf(pts).any(), "Inf in points"
            assert not torch.isnan(nrm).any(), "NaN in normals"

            # 验证法向量近似单位
            normal_norms = torch.norm(nrm, dim=1)
            assert normal_norms.min() > 0.9, "Normals too short"
            assert normal_norms.max() < 1.1, "Normals too long"

            # 验证特征值范围
            assert feats[:, 0].min() >= 0.0, f"Intensity < 0: {feats[:, 0].min()}"
            assert feats[:, 0].max() <= 1.0, f"Intensity > 1: {feats[:, 0].max()}"

            pavement_counts[ptype] = pavement_counts.get(ptype, 0) + 1

            print(f"  [PASS] N={N}, labels=[{min(unique_labels)}, {max(unique_labels)}], "
                  f"pavement={ptype}, feat_range=[{feats[:, 0].min():.3f}, {feats[:, 0].max():.3f}]")
            passed += 1

        except Exception as e:
            print(f"  [FAIL] Scene {i}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n" + "=" * 72)
    print(f"  场景总数: {total}")
    print(f"  通过: {passed}/{total}")

    if passed == total:
        print("  [OK] 全部通过")

    print(f"\n  路面类型分布: {pavement_counts}")
    print(f"  覆盖标签数: {len(all_labels)}/{NUM_CLASSES}")
    print(f"  标签范围: [{min(all_labels)}, {max(all_labels)}]")

    # 验证不同场景输出不同
    print("\n  检验多样性...")
    scenes = [dataset[i] for i in range(len(dataset))]
    pts_list = [s["points"] for s in scenes]
    different = any(
        not torch.allclose(pts_list[0], pts_list[j])
        for j in range(1, len(pts_list))
    )
    if different:
        print("  [PASS] 不同场景产生不同输出")
    else:
        print("  [WARN] 所有场景输出相同（可能种子管理有问题）")

    print("\n" + "=" * 72)
