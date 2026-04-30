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
        resample_to_lidar_pattern,
    )
except ImportError:
    # Fallback for standalone / -m execution
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parents[2])  # project root
    _parent = str(_Path(__file__).resolve().parent)     # synthetic/ dir
    for _p in (_root, _parent):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
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
        resample_to_lidar_pattern,
    )

# 常量

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

# P1-3 修复：病害标签优先级，数值越高优先级越高
# 当多个病害覆盖同一区域时，仅保留优先级最高的标签
# 设计原则：坑槽 > 龟裂 > 纵向裂缝 > 块状裂缝 > 横向裂缝，
# 大面积病害（车辙、沉陷、波浪拥包）优先级低于局部病害
LABEL_PRIORITY: Dict[int, int] = {
    0: 0,    # 背景
    # 沥青路面 — 局部破损优先级最高，大面积病害较低
    10: 10,  # 重坑槽 — 最高优先级
    9: 9,    # 轻坑槽
    2: 8,    # 重龟裂
    1: 7,    # 轻龟裂
    6: 6,    # 重纵向裂缝
    5: 5,    # 轻纵向裂缝
    8: 4,    # 重横向裂缝
    4: 3,    # 重块状裂缝
    7: 3,    # 轻横向裂缝
    3: 2,    # 轻块状裂缝
    12: 5,   # 重松散
    11: 4,   # 轻松散
    14: 2,   # 重沉陷
    13: 1,   # 轻沉陷
    16: 2,   # 重车辙
    15: 1,   # 轻车辙
    18: 2,   # 重波浪拥包
    17: 1,   # 轻波浪拥包
    19: 5,   # 泛油（标签特殊，面积区域覆盖）
    20: 6,   # 修补（人工标记，高优先级）
    # 水泥路面
    22: 10,  # 重破碎板
    21: 9,   # 轻破碎板
    24: 8,   # 重水泥裂缝
    23: 7,   # 轻水泥裂缝
    26: 6,   # 重板角断裂
    25: 5,   # 轻板角断裂
    28: 4,   # 重错台
    27: 3,   # 轻错台
    31: 4,   # 重边角剥落
    30: 3,   # 轻边角剥落
    33: 2,   # 重接缝料损坏
    32: 1,   # 轻接缝料损坏
    29: 5,   # 唧泥
    34: 6,   # 坑洞
    35: 7,   # 拱起
    36: 4,   # 露骨
    37: 8,   # 水泥修补
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
    - **feats**: ``(N, 3)`` torch.float32 — [强度反射率, 局部曲率, 裂缝边界距离]
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

    # 标准 Dataset 接口

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

    # 核心场景生成

    def generate_scene(self, idx: int) -> Dict:
        """生成一个完整的合成道路场景。

        内部流程按以下顺序执行：

        1. 选择路面类型 (asphalt / concrete)
        2. ISO 8608 PSD 路面轮廓 + fBm 微观纹理
        2.5 LiDAR 扫描线重采样 (可选, P1-1)
        3. 病害组合随机选择 & 应用（按 DISEASE_APPLY_ORDER 排序 + LABEL_PRIORITY）
        4. KDTree 局部曲率计算 (P2-2)
        5. 应用松散 (raveling) — 产生 NaN 点
        6. 松散标签膨胀 + NaN 过滤 (P0-3)
        7. 强度反射率计算 (P0-2)
        8. LiDAR 噪声仿真 (P2-3: 仅高曲率边缘混合)
        9. 最近邻传递标签/特征/法向量 (P0-4: 带距离阈值)
        10. 体素下采样 (P1-2) 或 num_points 重采样
        11. 坐标归一化
        12. 特征拼接

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
        # 2.5 P1-1: LiDAR 扫描线密度重采样（可选）
        # ================================================================
        if self.config.lidar_scan.enable:
            scan_idx = resample_to_lidar_pattern(
                points,
                scan_lines=self.config.lidar_scan.scan_lines,
                vertical_fov_deg=self.config.lidar_scan.vertical_fov_deg,
                scan_pattern=self.config.lidar_scan.scan_pattern,
                range_decay=self.config.lidar_scan.range_decay,
                incidence_angle_drop=self.config.lidar_scan.incidence_angle_drop,
                rng=rng,
            )
            if len(scan_idx) > 10:
                points = points[scan_idx]
                normals = normals[scan_idx]
            # 更新网格尺寸（扫描线重采样改变了点数）
            nx, ny = len(x_tmp), len(y_tmp)  # 网格维度保持不变（后续重建会用）

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
            old_labels = labels.copy()  # 保存旧标签用于优先级比较

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
                    "x_offset": self.config.concrete_damage.x_offset,
                    "y_offset": self.config.concrete_damage.y_offset,
                }
                points, labels = add_concrete_damage(
                    points, labels,
                    damage_type=damage_type, severity=sev,
                    params=params, seed=seed,
                )

            # P1-3 修复 + Q3 向量化：仅在优先级更高时保留新标签
            # 使用 numpy 查找表替代 Python 逐点 for 循环
            changed_mask = labels != old_labels
            if np.any(changed_mask):
                # 预构建优先级查找表
                max_label = max(LABEL_PRIORITY.keys()) + 1
                priority_lut = np.zeros(max_label, dtype=np.int32)
                for k, v in LABEL_PRIORITY.items():
                    priority_lut[k] = v
                # 向量化比较
                old_prio = priority_lut[old_labels[changed_mask]]
                new_prio = priority_lut[labels[changed_mask]]
                restore = old_prio > new_prio
                changed_indices = np.where(changed_mask)[0]
                labels[changed_indices[restore]] = old_labels[changed_indices[restore]]

        # ================================================================
        # 4. 曲率计算 — P2-2 修复：使用 KDTree 局部邻域，解除网格顺序依赖
        # ================================================================
        curvature = _compute_kdtree_curvature(points, k_neighbors=20)

        # ================================================================
        # 5. 应用松散 (raveling) — 产生 NaN 点
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

# ================================================================
        # 6. 松散标签膨胀 + NaN 点过滤
        # ================================================================
        # P0-3 修复：在移除 NaN 点之前，将松散标签膨胀到 NaN 点的邻域，
        # 确保噪声仿真后落在原松散区域的点能获得正确的松散标签。
        raveling_labels_mask = (labels == 11) | (labels == 12)
        valid_mask = ~np.any(np.isnan(points), axis=1)

        # 松散标签膨胀：使用网格邻域将松散标签传播到附近的背景点
        # 策略：对每个非 NaN 的松散点，将其周围 grid_res*2 半径内的
        # 背景（label=0）点标记为相同的松散标签
        raveling_valid_mask = raveling_labels_mask & valid_mask
        if np.any(raveling_valid_mask):
            raveling_xy = points[raveling_valid_mask][:, :2]
            raveling_severity = labels[raveling_valid_mask]
            # 有效背景点（非 NaN、非松散）
            background_valid_mask = valid_mask & ~raveling_labels_mask
            if np.any(background_valid_mask) and raveling_xy.shape[0] > 0:
                from scipy.spatial import KDTree as _KDTree
                bg_xy = points[background_valid_mask][:, :2]
                bg_labels = labels[background_valid_mask]
                bg_global_idx = np.where(background_valid_mask)[0]
                tree_dilate = _KDTree(raveling_xy)
                dilate_radius = grid_res * 3.0  # 膨胀半径：3倍网格分辨率
                # 查询背景点到最近的松散点距离
                nn_dist, nn_idx = tree_dilate.query(bg_xy, k=1)
                # 距离在膨胀半径内的背景点标记为松散
                in_dilate = nn_dist <= dilate_radius
                for i in np.where(in_dilate)[0]:
                    if bg_labels[i] == 0:  # 仅膨胀到背景点
                        labels[bg_global_idx[i]] = raveling_severity[nn_idx[i]]

        if np.any(valid_mask):
            points = points[valid_mask]
            labels = labels[valid_mask]
            normals = normals[valid_mask]
            curvature = curvature[valid_mask]
        else:
            raise RuntimeError("All points removed by raveling — no valid geometry remains.")

        if points.shape[0] == 0:
            raise RuntimeError("Empty point cloud after NaN filtering.")

        # ================================================================
        # 7. 强度反射率计算（在松散标签应用之后）
        # P0-2 修复：移到松散应用之后，确保松散区域的标签 11/12 被正确反映
        # ================================================================
        intensity = self._compute_intensity(labels, pavement_type, rng)

        # ================================================================
        # 8. LiDAR 噪声仿真
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
            enable_edge_mixing=self.config.lidar_noise.enable_edge_mixing,
            mixed_pixel_prob=self.config.lidar_noise.mixed_pixel_prob,
            curvature=pre_curvature,  # P2-3: 仅高曲率边缘点混合
            curvature_threshold=0.5,
        )

        if noisy_points.shape[0] == 0:
            raise RuntimeError("All points dropped during LiDAR noise simulation.")

        # ================================================================
        # 9. 最近邻传递标签/特征/法向量（带距离阈值保护）
        # P0-4 修复：增加距离阈值，防止标签溢出/侵蚀
        # ================================================================
        from scipy.spatial import KDTree
        tree = KDTree(pre_points)
        nn_dist, nn_idx = tree.query(noisy_points, k=1)

        labels = pre_labels[nn_idx]
        intensity = pre_intensity[nn_idx]
        curvature = pre_curvature[nn_idx]
        normals = pre_normals[nn_idx]

        # P0-4 距离阈值保护：超过 3σ 距离的点回退到背景标签
        # 防止裂缝边缘噪声点携带错误标签到远处
        max_transfer_dist = self.config.lidar_noise.distance_noise_std * 3.0
        uncertain_mask = nn_dist > max_transfer_dist
        # 对不确定点：仅保留非裂缝标签（标签 > 0 且 <= 8 的裂缝标签降为背景）
        # 但保留松散、车辙、沉陷等大面积病害标签
        crack_mask = (labels >= 1) & (labels <= 8)  # Q1: 布尔掩码替代 set + for
        restore_mask = uncertain_mask & crack_mask
        labels[restore_mask] = 0  # 不确定的裂缝标签降为背景

        # ================================================================
        # 10. P2-4: 裂缝边界软标签 + P1-2: 体素下采样
        # ================================================================
        # P2-4: 为每个点计算到最近裂缝边界的距离，转换为裂缝概率软标签
        # 裂缝边界点: label ∈ [1,8] 且邻域有背景点的点
        crack_boundary_dist = np.zeros(len(noisy_points), dtype=np.float32)
        crack_mask_p2_4 = (labels >= 1) & (labels <= 8)
        if np.any(crack_mask_p2_4) and np.any(~crack_mask_p2_4):
            from scipy.spatial import KDTree as _KDTree
            crack_pts = noisy_points[crack_mask_p2_4]
            bg_pts = noisy_points[~crack_mask_p2_4]
            if len(crack_pts) > 1 and len(bg_pts) > 1:
                # 对每个背景点，计算到最近裂缝点的距离
                crack_tree = _KDTree(crack_pts)
                bg_dists, _ = crack_tree.query(bg_pts, k=1)
                # 对裂缝点，计算到最近背景点的距离（双向检测）
                bg_tree = _KDTree(bg_pts)
                crack_dists, _ = bg_tree.query(crack_pts, k=1)
                # 填充距离数组
                crack_boundary_dist[crack_mask_p2_4] = crack_dists
                crack_boundary_dist[~crack_mask_p2_4] = bg_dists
                # 距离→概率：exp(-d²/2σ²)，σ 取 3×grid_res
                sigma = grid_res * 3.0
                # crack_probability: 1 表示确定是裂缝区域（低距离），0 表示远离裂缝
                # 使用对称 sigmoid: probability = exp(-d²/2σ²) 用于非裂缝点
                # 裂缝点内部概率=1，边界点概率衰减
        # P1-2: 使用体素下采样替代随机重采样
        if self.config.target_density is not None and self.config.target_density > 0:
            road_area = width * length
            target_count = int(road_area * self.config.target_density)
            # 计算合适的体素大小以逼近目标点数
            # N_voxels ≈ area / voxel_size²（仅用 xy 维度）
            effective_voxel_size = max(
                np.sqrt(road_area / max(target_count, 1)),
                grid_res * 1.5
            )
            # 体素质心下采样
            points_final, labels_final, intensity_final, curvature_final, normals_final = (
                self._voxel_downsample(
                    noisy_points, labels, intensity, curvature, normals,
                    voxel_size=effective_voxel_size,
                )
            )
            # P2-4: 最近邻传递裂缝边界距离
            from scipy.spatial import KDTree as _KDTree
            bd_tree = _KDTree(noisy_points)
            _, bd_idx = bd_tree.query(points_final, k=1)
            crack_boundary_dist_final = crack_boundary_dist[bd_idx]
        else:
            # 回退到旧的 num_points 硬重采样
            points_final, labels_final, intensity_final, curvature_final, normals_final = (
                self._resample_to_target(
                    noisy_points, labels, intensity, curvature, normals,
                    self.config.num_points, rng,
                )
            )
            # P2-4: KDTree 传递裂缝边界距离到重采样后的点
            from scipy.spatial import KDTree as _KDTree
            bd_tree = _KDTree(noisy_points)
            _, bd_idx = bd_tree.query(points_final, k=1)
            crack_boundary_dist_final = crack_boundary_dist[bd_idx]

        # ================================================================
        # 11. 坐标归一化
        # ================================================================
        if self.config.normalize:
            points_final = self._normalize(points_final)

        # ================================================================
        # 12. 特征拼接 (P2-4: 增加裂缝边界距离通道)
        # ================================================================
        feats = np.stack(
            [intensity_final, curvature_final, crack_boundary_dist_final], axis=1
        ).astype(np.float32)

        return {
            "points": points_final.astype(np.float32),
            "labels": labels_final.astype(np.int64),
            "feats": feats,          # (N, 3): [intensity, curvature, crack_boundary_dist]
            "normals": normals_final.astype(np.float32),
            "pavement_type": pavement_type,
        }

    # 辅助方法

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
        """模拟 LiDAR 强度反射率（基于物理反射率模型）。

        P1-6 修复：使用物理合理的反射率范围和单位。
        沥青基底反射率 0.10-0.25 (低反射率深色路面)，
        水泥基底反射率 0.30-0.50 (浅色路面)。
        输出范围 [0, 1]，代表归一化反射率。

        Args:
            labels: 标签数组 (N,)。
            pavement_type: 路面类型。
            rng: 随机数生成器。

        Returns:
            强度值 (N,) float32，范围 [0, 1]。
        """
        N = labels.shape[0]
        # P1-6 修复：使用物理合理的反射率基线
        if pavement_type == "asphalt":
            # 沥青路面反射率低，0.10-0.25 (深灰到黑)
            base = float(rng.uniform(0.10, 0.25))
        else:
            # 水泥路面反射率较高，0.30-0.50 (浅灰)
            base = float(rng.uniform(0.30, 0.50))

        intensity = np.full(N, base, dtype=np.float64)
        # 随机空间变化（模拟路面局部反射率差异）
        intensity += rng.normal(0.0, 0.02, size=N)

        # 病害区域强度修正（基于 JTG 标准中病害的视觉反射率特征）
        intensity[labels == 19] += 0.20   # 泛油：反射率显著增加 (油膜反射)
        intensity[(labels == 11) | (labels == 12)] -= 0.10  # 松散：反射率降低 (粗糙表面散射)
        intensity[labels == 36] -= 0.08   # 露骨：骨料反射率略低于均匀水泥表面
        intensity[(labels == 20) | (labels == 37)] += 0.15  # 修补：修补材料通常反射率较高
        intensity[labels == 29] += 0.12   # 唧泥：湿润泥浆反射率高于干燥路面
        intensity[labels == 34] -= 0.05   # 坑洞：深处反射率极低
        # 裂缝：反射率轻微降低 (阴影 + 粗糙边缘)
        crack_mask = (labels >= 1) & (labels <= 8)
        intensity[crack_mask] -= 0.05
        # 坑槽：反射率降低
        intensity[(labels == 9) | (labels == 10)] -= 0.08

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
    def _voxel_downsample(
        points: np.ndarray,
        labels: np.ndarray,
        intensity: np.ndarray,
        curvature: np.ndarray,
        normals: np.ndarray,
        voxel_size: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """P1-2: 体素质心下采样，保留密度结构。"""
        if points.shape[0] < 10:
            return points, labels, intensity, curvature, normals

        # 体素索引 — 仅使用 xy 维度（路面高度变化不应用来分割体素）
        voxel_indices = np.floor(points[:, :2] / voxel_size).astype(np.int64)
        # 用字典聚合
        voxel_dict: Dict[Tuple[int, int], List[int]] = {}
        for i, vi in enumerate(voxel_indices):
            key = (int(vi[0]), int(vi[1]))
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        N_voxels = len(voxel_dict)

        # 聚合
        result_pts = np.zeros((N_voxels, 3), dtype=np.float64)
        result_labels = np.zeros(N_voxels, dtype=np.int64)
        result_intensity = np.zeros(N_voxels, dtype=np.float32)
        result_curvature = np.zeros(N_voxels, dtype=np.float32)
        result_normals = np.zeros((N_voxels, 3), dtype=np.float32)

        for vi, (key, idx_list) in enumerate(voxel_dict.items()):
            idx_arr = np.array(idx_list)
            # 质心
            result_pts[vi] = np.mean(points[idx_arr], axis=0)
            # 多数投票标签
            lbl_counts = np.bincount(labels[idx_arr].astype(np.int64))
            result_labels[vi] = np.argmax(lbl_counts)
            # 均值特征
            result_intensity[vi] = np.mean(intensity[idx_arr])
            result_curvature[vi] = np.mean(curvature[idx_arr])
            result_normals[vi] = np.mean(normals[idx_arr], axis=0)
            # 重新归一化法向量
            nrm = np.linalg.norm(result_normals[vi])
            if nrm > 1e-12:
                result_normals[vi] /= nrm

        return (
            result_pts.astype(np.float32),
            result_labels,
            result_intensity,
            result_curvature,
            result_normals,
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


# 工具函数


def _compute_kdtree_curvature(
    points: np.ndarray, k_neighbors: int = 20
) -> np.ndarray:
    """P2-2: Compute local curvature using KDTree neighborhood PCA.

    对每个点，用其 k 近邻做 PCA，最小特征值与特征值总和之比作为局部曲率估计。
    不依赖网格顺序，适用于任意点排列。

    .. math::
        C_i = \\frac{\\lambda_{\\min}}{\\lambda_1 + \\lambda_2 + \\lambda_3}

    其中 :math:`\\lambda` 为邻域协方差矩阵的特征值。

    Args:
        points: 点云 (N, 3)。
        k_neighbors: 邻域点数，默认 20。

    Returns:
        曲率值 (N,) float32，范围约 [0, 1]。
    """
    from scipy.spatial import KDTree as _KDTree
    N = points.shape[0]
    k = min(k_neighbors, N - 1)
    if k < 3:
        return np.zeros(N, dtype=np.float32)

    tree = _KDTree(points)
    curvature = np.zeros(N, dtype=np.float64)

    for i in range(N):
        _, idx = tree.query(points[i], k=k + 1)  # k+1 包含自身
        neighbors = points[idx[1:]]  # 排除自身
        # 中心化
        centered = neighbors - neighbors.mean(axis=0)
        # 3x3 协方差矩阵
        cov = np.dot(centered.T, centered) / (k - 1)
        # 特征值分解
        try:
            eigvals = np.linalg.eigvalsh(cov)
            # PCA 曲率 = λ_min / (λ_1 + λ_2 + λ_3)
            total = np.sum(np.abs(eigvals))
            if total > 1e-12:
                curvature[i] = np.abs(eigvals[0]) / total
        except np.linalg.LinAlgError:
            curvature[i] = 0.0

    return curvature.astype(np.float32)


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


# 自检脚本

if __name__ == "__main__":
    """自检：生成 5 个场景并验证输出。"""
    print("=" * 72)
    print("  SyntheticRoadDataset Self-Test")
    print("  生成 5 个场景并验证输出形状 / 标签范围 / 归一化")
    print("=" * 72)

    # 小尺寸测试配置（加快测试速度）
    try:
        from .config import GeneratorConfig, RoadSurfaceConfig, DiseaseConfig
    except ImportError:
        from config import GeneratorConfig, RoadSurfaceConfig, DiseaseConfig

    cfg = GeneratorConfig(
        road=RoadSurfaceConfig(
            width=2.0,
            length=2.0,
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
            assert feats.shape == (N, 3), f"feats shape: {feats.shape}"
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
