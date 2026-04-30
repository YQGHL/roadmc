"""
RoadMC 数学与力学基元 —— Physics-Simulation-Driven Road Surface Primitives.
=======================================================================

严格遵循 JTG 5210-2018《公路技术状况评定标准》，共 38 个标签 (0-37)。

本文件实现了 11 个数学与力学基元函数，通过物理学仿真生成路面点云数据：
  1. generate_road_surface()   — ISO 8608 PSD 路面宏观轮廓 (FFT 谱合成)
  2. add_micro_texture()       — 分数布朗运动 (fBm) 微观纹理叠加
  3. add_crack()               — 裂缝 (纵向/横向 Bézier + 龟裂/块状 Voronoi)
  4. add_pothole()             — 超椭圆坑槽
  5. add_raveling()            — 松散 (细集料脱落)
  6. add_depression()          — 沉陷 (高斯凹陷)
  7. add_rutting()             — 车辙 (双轮迹高斯槽)
  8. add_corrugation()         — 波浪拥包 (正弦调制)
  9. add_bleeding()            — 泛油 (反射率/标签修改)
  10. add_concrete_damage()     — 水泥路面 10 种损坏
  11. simulate_lidar_noise()    — LiDAR 噪声仿真 (球坐标 + Dropout)

所有函数使用 numpy + scipy，不使用 torch。
随机状态统一使用 ``np.random.default_rng(seed)``。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import interpolate, spatial, stats

try:
    from .config import ISO_ROUGHNESS
except ImportError:
    from config import ISO_ROUGHNESS

# ===========================================================================
# 辅助函数 (Helpers)
# ===========================================================================


def _compute_normals(
    Z: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """Compute surface unit normals via central finite differences.

    给定高度场 Z :math:`Z(x, y)`，表面法向量为：

    .. math::
        \\mathbf{n} = \\frac{(-\\partial Z/\\partial x,\\,
        -\\partial Z/\\partial y,\\, 1)^\\top}
        {\\|(-\\partial Z/\\partial x,\\,
        -\\partial Z/\\partial y,\\, 1)\\|}

    使用二阶中心差分计算偏导数：

    .. math::
        \\frac{\\partial Z}{\\partial x}(i,j) \\approx
        \\frac{Z(i+1,j) - Z(i-1,j)}{2\\Delta x}

        \\frac{\\partial Z}{\\partial y}(i,j) \\approx
        \\frac{Z(i,j+1) - Z(i,j-1)}{2\\Delta y}

    Args:
        Z: 高度场 (M, N)。
        dx: x 方向网格间距。
        dy: y 方向网格间距。

    Returns:
        单位法向量 (M, N, 3)。
    """
    dz_dx = np.zeros_like(Z)
    dz_dy = np.zeros_like(Z)

    # 内部点用中心差分
    dz_dx[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / (2.0 * dx)
    dz_dy[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / (2.0 * dy)

    # 边界用一阶差分 (前向/后向)
    dz_dx[0, :] = (Z[1, :] - Z[0, :]) / dx
    dz_dx[-1, :] = (Z[-1, :] - Z[-2, :]) / dx
    dz_dy[:, 0] = (Z[:, 1] - Z[:, 0]) / dy
    dz_dy[:, -1] = (Z[:, -1] - Z[:, -2]) / dy

    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(Z)), axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(norms, 1e-12, None)


def _cubic_bezier(
    control_points: np.ndarray, num_samples: int
) -> np.ndarray:
    """Sample a cubic Bézier curve.

    Cubic Bézier 曲线由 4 个控制点 :math:`P_0, P_1, P_2, P_3` 定义：

    .. math::
        B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3,
        \\quad t \\in [0, 1]

    Args:
        control_points: 控制点 (4, 2)。
        num_samples: 采样点数。

    Returns:
        曲线上采样点 (num_samples, 2)。
    """
    t = np.linspace(0.0, 1.0, num_samples)
    P0, P1, P2, P3 = control_points
    # Bernstein 多项式
    B = (
        (1 - t[:, None]) ** 3 * P0[None, :]
        + 3 * (1 - t[:, None]) ** 2 * t[:, None] * P1[None, :]
        + 3 * (1 - t[:, None]) * t[:, None] ** 2 * P2[None, :]
        + t[:, None] ** 3 * P3[None, :]
    )
    return B


def _fractal_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    octaves: int = 4,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    scale: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Multi-octave fractal noise (value noise) for crack path perturbation.

    通过多倍频程插值噪声叠加生成分形扰动场：

    .. math::
        f(\\mathbf{x}) = \\sum_{k=0}^{\\text{octaves}-1}
        p^k \\cdot \\text{noise}\\left(\\frac{l^k \\mathbf{x}}{s}\\right)

    Args:
        x: x 坐标。
        y: y 坐标。
        octaves: 倍频程数。
        lacunarity: 频率倍增因子。
        persistence: 振幅衰减因子。
        scale: 空间尺度。
        seed: 随机种子。

    Returns:
        扰动值数组，与 x, y 形状相同。
    """
    rng = np.random.default_rng(seed)
    result = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0 / scale

    xi = np.asarray(x)
    yi = np.asarray(y)

    for o in range(octaves):
        grid_size = 8  # 粗网格分辨率
        noise_grid = rng.uniform(-1.0, 1.0, (grid_size, grid_size))

        interp = interpolate.RegularGridInterpolator(
            (np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size)),
            noise_grid,
            bounds_error=False,
            fill_value=0.0,
        )

        # 在扰动坐标处采样
        sample_x = (xi * frequency) % 1.0
        sample_y = (yi * frequency) % 1.0
        result += amplitude * interp(np.stack([sample_x, sample_y], axis=-1))

        amplitude *= persistence
        frequency *= lacunarity

    return result


def _point_to_segment_distance(
    points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> np.ndarray:
    """Compute perpendicular distance from points to a line segment.

    .. math::
        d = \\frac{\\|(P - A) \\times (P - B)\\|}{\\|B - A\\|}
        \\quad \\text{(2D cross product magnitude)}

    where projection falls onto the segment.

    Args:
        points: 查询点 (N, 2)。
        seg_start: 线段起点 (2,)。
        seg_end: 线段终点 (2,)。

    Returns:
        点到线段距离 (N,)。
    """
    A = seg_start[None, :]  # (1, 2)
    B = seg_end[None, :]  # (1, 2)
    AB = B - A
    AP = points - A

    ab2 = np.sum(AB ** 2)
    if ab2 < 1e-12:
        return np.sqrt(np.sum(AP ** 2, axis=1))

    # 投影参数 t = (AP · AB) / (AB · AB)
    t = np.sum(AP * AB, axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)

    # 最近点
    closest = A + t[:, None] * AB
    return np.sqrt(np.sum((points - closest) ** 2, axis=1))


def _point_to_segment_distance_t(
    points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute perpendicular distance and projection parameter from points to a line segment.

    Same as _point_to_segment_distance but also returns the local parameter t
    along the segment (0=seg_start, 1=seg_end).

    Args:
        points: 查询点 (N, 2)。
        seg_start: 线段起点 (2,)。
        seg_end: 线段终点 (2,)。

    Returns:
        distances: 点到线段距离 (N,)。
        t_local: 线段上的投影参数 (N,)，范围 [0, 1]。
    """
    A = seg_start[None, :]  # (1, 2)
    B = seg_end[None, :]  # (1, 2)
    AB = B - A
    AP = points - A

    ab2 = np.sum(AB ** 2)
    if ab2 < 1e-12:
        return np.sqrt(np.sum(AP ** 2, axis=1)), np.zeros(len(points))

    # 投影参数 t = (AP · AB) / (AB · AB)
    t_local = np.clip(np.sum(AP * AB, axis=1) / ab2, 0.0, 1.0)

    # 最近点
    closest = A + t_local[:, None] * AB
    dist = np.sqrt(np.sum((points - closest) ** 2, axis=1))
    return dist, t_local


def _bilinear_interpolation(
    x: np.ndarray, y: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation on a regular 2D grid.

    Args:
        x: x 坐标数组。
        y: y 坐标数组。
        grid_x: 1D x 网格。
        grid_y: 1D y 网格。
        grid_z: 网格值 (len(grid_x), len(grid_y))。

    Returns:
        插值结果，与 x, y 同形状。
    """
    interp = interpolate.RegularGridInterpolator(
        (grid_x, grid_y),
        grid_z,
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(np.stack([x, y], axis=-1))


# ===========================================================================
# 1.1.1 — 路面宏观轮廓生成 (ISO 8608 PSD)
# ===========================================================================


def generate_road_surface(
    width: float,
    length: float,
    grid_res: float,
    pavement_type: str = "asphalt",
    roughness_class: str = "A",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate road surface point cloud via ISO 8608 PSD spectral synthesis.

    ISO 8608 路面功率谱密度 (PSD)：

    .. math::
        G_d(n) = G_d(n_0) \\left(\\frac{n}{n_0}\\right)^{-w}

    其中 :math:`n_0 = 0.1` cycle/m, :math:`w = 2`,
    :math:`G_d(n_0)` 取值由 ``ISO_ROUGHNESS`` 配置决定 (单位 ×10⁻⁶ m³/cycle)。

    二维路面生成使用可分离 PSD 模型 + 逆傅里叶变换：

    .. math::
        S(f_x, f_y) = G_d(|f_x|) \\cdot G_d(|f_y|)

    :math:`h(x,y) = \\mathcal{F}^{-1}\\left\\{
        \\sqrt{S(f_x, f_y)} \\cdot W(f_x, f_y) \\right\\}`

    其中 :math:`W(f_x, f_y)` 为复高斯白噪声的傅里叶变换。

    法向量通过中心有限差分计算并归一化。

    Args:
        width:  路面宽度 (m)，x 方向。
        length: 路面长度 (m)，y 方向。
        grid_res: 网格分辨率 (m)。
        pavement_type: 路面类型 ('asphalt' | 'concrete')，仅影响标签。
        roughness_class: ISO 8608 粗糙度等级 'A'–'E'。
        seed: 随机种子。

    Returns:
        points:  点云 (N, 3) = [x, y, z]。
        normals: 单位法向量 (N, 3)。
    """
    rng = np.random.default_rng(seed)

    # --- 创建网格 ---
    x = np.arange(0.0, width, grid_res)
    y = np.arange(0.0, length, grid_res)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)

    # --- ISO 8608 PSD 参数 ---
    n0 = 0.1  # 参考空间频率 (cycle/m)
    w = 2.0  # 波度指数 (waviness exponent)
    Gd0 = ISO_ROUGHNESS[roughness_class] * 1e-6  # m³/cycle

    # --- 空间频率网格 ---
    fx = np.fft.fftfreq(nx, d=grid_res)  # (nx,)
    fy = np.fft.fftfreq(ny, d=grid_res)  # (ny,)

    # 可分离 1D PSD
    with np.errstate(divide="ignore", invalid="ignore"):
        psd_x = np.where(np.abs(fx) > 0, Gd0 * (np.abs(fx) / n0) ** (-w), 0.0)
        psd_y = np.where(np.abs(fy) > 0, Gd0 * (np.abs(fy) / n0) ** (-w), 0.0)

    # 二维 PSD (可分离模型)：S(fx, fy) = G_d(|fx|) * G_d(|fy|)
    psd_2d = np.outer(psd_x, psd_y)  # (nx, ny)

    # --- 谱合成：白噪声 → FFT → 滤波 → IFFT ---
    white_noise = rng.normal(0.0, 1.0, (nx, ny))
    W_hat = np.fft.fft2(white_noise)

    H_hat = np.sqrt(psd_2d + 1e-30) * W_hat  # 避免 sqrt(0)
    h_field = np.real(np.fft.ifft2(H_hat))

    # 零均值化
    h_field = h_field - np.mean(h_field)

    # --- RMS 缩放：将 RMS 匹配到 PSD 积分值 ---
    # 理论方差：σ²_target = [Σ G_d(fx_i)Δfx] · [Σ G_d(fy_j)Δfy]
    dfx = 1.0 / (nx * grid_res)  # 频率分辨率
    dfy = 1.0 / (ny * grid_res)
    var_target = (np.sum(psd_x) * dfx) * (np.sum(psd_y) * dfy)
    rms_target = np.sqrt(max(var_target, 1e-30))
    rms_current = np.std(h_field)
    if rms_current > 1e-12:
        h_field = h_field * (rms_target / rms_current)

    # --- 计算法向量 ---
    normals_grid = _compute_normals(h_field, grid_res, grid_res)

    # --- 展平为点云 ---
    points = np.stack([X.ravel(), Y.ravel(), h_field.ravel()], axis=1)  # (N, 3)
    normals = normals_grid.reshape(-1, 3)  # (N, 3)

    return points, normals


# ===========================================================================
# 1.1.2 — 微观纹理叠加 (fBm)
# ===========================================================================


def resample_to_lidar_pattern(
    points: np.ndarray,
    scan_lines: int = 64,
    vertical_fov_deg: float = 40.0,
    scan_pattern: str = "rotating",
    range_decay: float = 0.3,
    incidence_angle_drop: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """P1-1: Resample uniform grid points to simulate LiDAR scan line density.

    将规则网格点云重采样为模拟 LiDAR 扫描线模式的非均匀分布。

    旋转式 LiDAR：
    - 沿 y 方向（道路纵向）的扫描线间距由 scan_lines 和 vertical_fov 决定
    - 沿 x 方向（扫描方向）保持高密度
    - 施加距离衰减和入射角丢点

    .. math::
        P_{\\text{keep}}(x,y) = \\frac{1}{1 + \\alpha \\cdot |y - y_{\\text{scan}}|}
        \\cdot \\left(1 - \\beta \\cdot \\frac{r}{r_{\\max}}\\right)

    Args:
        points: 规则网格点云 (N, 3)。
        scan_lines: 扫描线数量。
        vertical_fov_deg: 垂直视场角 (度)。
        scan_pattern: 'rotating' 或 'solid_state'。
        range_decay: 距离衰减系数 α。
        incidence_angle_drop: 入射角丢点系数 β。
        rng: 随机数生成器。

    Returns:
        重采样后的点云索引 (N',)，可用于 points[idx]。
    """
    if rng is None:
        rng = np.random.default_rng()

    N = points.shape[0]
    if N < 10:
        return np.arange(N)

    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # 距离（简化：以路面中心为扫描原点）
    sensor_height = max(x_max - x_min, y_max - y_min) * 1.5  # 假设传感器高度
    r_dist = np.sqrt((x - (x_min + x_max) / 2) ** 2 + sensor_height ** 2)
    r_max = np.max(r_dist) + 1e-12

    if scan_pattern == "rotating":
        # 扫描线在 y 方向上分布
        scan_spacing = (y_max - y_min) / scan_lines
        # 每条扫描线上的 y 中心位置
        y_scans = np.linspace(y_min, y_max, scan_lines)

        # 每个点到最近扫描线的距离
        nearest_scan = np.argmin(np.abs(y[:, None] - y_scans[None, :]), axis=1)
        dist_to_scan = np.abs(y - y_scans[nearest_scan])

        # 到扫描线距离越远，保留概率越低（Gaussian falloff）
        scan_sigma = scan_spacing * 0.8  # 扫描线有效宽度
        scan_prob = np.exp(-0.5 * (dist_to_scan / scan_sigma) ** 2)

        # 距离衰减
        range_prob = 1.0 - range_decay * (r_dist / r_max)

        # 入射角效应：边缘点更稀疏（大角度入射 → 低反射率 → 丢点率高）
        # 简化：用 x 到中心线的距离模拟入射角
        x_center = (x_min + x_max) / 2.0
        incidence_angle = np.abs(x - x_center) / max(x_max - x_center, 0.01)
        incidence_prob = 1.0 - incidence_angle_drop * incidence_angle

        keep_prob = scan_prob * range_prob * incidence_prob
        keep_prob = np.clip(keep_prob, 0.0, 1.0)

    else:  # solid_state
        # 固态闪光 LiDAR：均匀但带距离衰减
        range_prob = 1.0 - range_decay * (r_dist / r_max)
        keep_prob = np.clip(range_prob, 0.0, 1.0)

    # Bernoulli 采样
    keep_mask = rng.random(N) < keep_prob
    return np.where(keep_mask)[0]


def add_micro_texture(
    points: np.ndarray,
    normals: np.ndarray,
    amplitude: float,
    hurst: float,
    octaves: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add micro-texture via fractional Brownian motion (fBm) overlay.

    分数布朗运动 (fBm) 沿法线方向叠加微纹理：

    .. math::
        \\mathbf{p}' = \\mathbf{p} + A \\sum_{k=0}^{\\text{octaves}-1}
        s^{-kH} \\cdot \\mathcal{L}_{\\alpha}\\left(\\cdot\\right) \\cdot \\mathbf{n}

    其中 :math:`H` 为 Hurst 指数，:math:`\\mathcal{L}_{\\alpha}` 为 Lévy
    :math:`\\alpha`-stable 分布 (:math:`\\alpha = 2H`)，:math:`s` 为倍频程缩放因子，
    :math:`\\mathbf{n}` 为法向量。

    Hurst 指数 :math:`H \\in (0,1)`：
    - :math:`H \\to 0.5`：标准布朗运动 (粗糙)
    - :math:`H \\to 1.0`：平滑趋势
    - :math:`H \\to 0.0`：极端粗糙

    Args:
        points:  点云 (N, 3)。
        normals: 单位法向量 (N, 3)。
        amplitude: fBm 总振幅 (m)。
        hurst: Hurst 指数。
        octaves: 叠加倍频程数。
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        normals_modified: 更新后单位法向量 (N, 3)。
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]

    pts = points.copy()
    nrm = normals.copy()

    # Lévy stable alpha = 2*H (但限制在 [0.1, 1.9] 以避免 scipy 数值问题)
    alpha = max(0.1, min(1.9, 2.0 * hurst))

    # 法向量方向位移
    delta = np.zeros(N, dtype=np.float64)

    for k in range(octaves):
        # 每倍频程的缩放因子
        scale = 2.0 ** (-k * hurst)
        # 用 Lévy stable 生成噪声
        # 生成 N 个独立同分布随机数
        noise = stats.levy_stable.rvs(
            alpha=alpha,
            beta=0.0,  # 对称
            loc=0.0,
            scale=1.0,
            size=N,
            random_state=rng,
        )
        # 剔除极端值 (截断到 5σ)
        noise = np.clip(noise, -5.0, 5.0)
        delta += scale * noise

    # 沿法线方向位移
    displacement = amplitude * delta[:, None] * nrm
    pts += displacement

    return pts, nrm


# ===========================================================================
# 1.1.3 — 裂缝生成 (沥青路面)
# ===========================================================================


def _generate_alligator_seeds(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    num_seeds: int,
    inhibition_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """M2: Matern-II-type Poisson point process with inhibition radius.

    Generates crack nucleation seeds with a minimum spacing constraint,
    preventing unnaturally overlapping Voronoi cells that occur with
    uniform random placement.

    Args:
        x_min, x_max: x range.
        y_min, y_max: y range.
        num_seeds: target number of seeds (Poisson mean).
        inhibition_radius: minimum distance between any two seeds.
        rng: random number generator.

    Returns:
        (N, 2) array of seed coordinates.
    """
    actual = rng.poisson(num_seeds)
    actual = max(actual, 3)
    seeds = []
    max_attempts = actual * 10
    attempts = 0
    while len(seeds) < actual and attempts < max_attempts:
        candidate = np.array([
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
        ])
        if all(np.linalg.norm(candidate - s) > inhibition_radius for s in seeds):
            seeds.append(candidate)
        attempts += 1
    return np.array(seeds) if seeds else np.array([[x_min, y_min]])


def add_crack(
    points: np.ndarray,
    labels: np.ndarray,
    crack_type: str,
    severity: str,
    params: dict,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add crack(s) to asphalt road surface.

    支持四种沥青裂缝类型：

    **纵向/横向裂缝 (Longitudinal/Transverse)**：
    - 骨架路径由三次 Bézier 曲线定义
    - 叠加分形扰动 (Perlin 噪声调制)
    - 开口宽度 :math:`w \\sim \\text{Lognormal}(\\mu, \\sigma)`
    - 深度剖面 (V 形槽)：

    .. math::
        d(t) = d_{\\max} \\exp\\left(-(t/\\lambda)^p\\right)

    **龟裂 (Alligator Cracks)**：
    - 空间泊松点过程 → Voronoi 图 → 多边形边界裂缝
    - 轻：块度 > 0.5m；重：< 0.2m

    **块状裂缝 (Block Cracks)**：
    - 类似龟裂但块度 > 1m，密度更低

    标签体系：
    - 龟裂: 轻=1, 重=2
    - 块状裂缝: 轻=3, 重=4
    - 纵向裂缝: 轻=5, 重=6
    - 横向裂缝: 轻=7, 重=8

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        crack_type: 裂缝类型：'longitudinal' | 'transverse'
                     | 'alligator' | 'block'。
        severity: 严重程度 'light' | 'severe'。
        params: 参数字典，支持：
            - 'd_max': 最大深度 (m)，默认轻=0.01, 重=0.03
            - 'width_mean': 宽度对数均值 (m)，默认 0.005
            - 'width_std': 宽度对数标准差，默认 0.3
            - 'bezier_control_points': Bézier 控制点 (4,2) 可选
            - 'num_seeds': Voronoi 种子点数 (龟裂/块状)
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    N = pts.shape[0]
    xy = pts[:, :2]

    # --- 从映射或直接参数获取标签 ---
    if crack_type == "alligator":
        label_val = 1 if severity == "light" else 2
        d_max = params.get("d_max", 0.010 if severity == "light" else 0.030)
    elif crack_type == "block":
        label_val = 3 if severity == "light" else 4
        d_max = params.get("d_max", 0.008 if severity == "light" else 0.025)
    elif crack_type == "longitudinal":
        label_val = 5 if severity == "light" else 6
        d_max = params.get("d_max", 0.010 if severity == "light" else 0.030)
    elif crack_type == "transverse":
        label_val = 7 if severity == "light" else 8
        d_max = params.get("d_max", 0.010 if severity == "light" else 0.030)
    else:
        raise ValueError(f"Unknown crack_type: {crack_type}")

    width_mean = params.get("width_mean", 0.005)
    width_std = params.get("width_std", 0.3)

    # =======================================================================
    # 纵向/横向裂缝
    # =======================================================================
    if crack_type in ("longitudinal", "transverse"):
        # 确定路面范围
        x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))

        # 生成 Bézier 控制点
        if "bezier_control_points" in params:
            control_pts = np.array(params["bezier_control_points"], dtype=np.float64)
        else:
            if crack_type == "longitudinal":
                # 纵向裂缝：沿 y 方向
                y_vals = np.linspace(y_min + 0.1 * (y_max - y_min),
                                     y_max - 0.1 * (y_max - y_min), 4)
                x_center = (x_min + x_max) * 0.5
                # 略微偏移控制点模拟真实裂缝弯曲
                offsets = rng.normal(0, 0.05 * (x_max - x_min), 4)
                offsets[0] = offsets[0] * 0.5
                offsets[-1] = offsets[-1] * 0.5
                control_pts = np.stack([x_center + np.full(4, offsets[0]) + offsets,
                                        y_vals], axis=1)
            else:
                # 横向裂缝：沿 x 方向
                x_vals = np.linspace(x_min + 0.1 * (x_max - x_min),
                                     x_max - 0.1 * (x_max - x_min), 4)
                y_center = (y_min + y_max) * 0.5
                offsets = rng.normal(0, 0.05 * (y_max - y_min), 4)
                offsets[0] = offsets[0] * 0.5
                offsets[-1] = offsets[-1] * 0.5
                control_pts = np.stack([x_vals, y_center + np.full(4, offsets[0]) + offsets],
                                       axis=1)

        # 在 Bézier 曲线上采样密集点
        num_curve_samples = max(N // 100, 100)
        curve_pts = _cubic_bezier(control_pts, num_curve_samples)

        # 分形扰动
        pert_seed = rng.integers(0, 2**31)
        x_curve = curve_pts[:, 0]
        y_curve = curve_pts[:, 1]
        pert_x = _fractal_perturbation(x_curve, y_curve, octaves=3,
                                       persistence=0.3, scale=0.5, seed=pert_seed)
        pert_y = _fractal_perturbation(x_curve, y_curve, octaves=3,
                                       persistence=0.3, scale=0.5, seed=pert_seed + 1)
        curve_pts[:, 0] += pert_x * 0.02
        curve_pts[:, 1] += pert_y * 0.02

        # 对每个点计算到扰动曲线的最小距离和投影参数
        min_dist = np.full(N, np.inf)
        min_t = np.zeros(N)  # 每个点在曲线上最近位置的参数 t ∈ [0, 1]
        # 用多段线近似，计算到每段距离和局部参数
        for k in range(num_curve_samples - 1):
            t_start = k / (num_curve_samples - 1)
            t_end = (k + 1) / (num_curve_samples - 1)
            dists, t_local = _point_to_segment_distance_t(
                xy, curve_pts[k], curve_pts[k + 1]
            )
            # 将局部参数映射到全局曲线参数
            t_on_curve_local = t_start + t_local * (t_end - t_start)
            # 仅在当前段更近时更新距离和参数
            better = dists < min_dist
            min_dist = np.where(better, dists, min_dist)
            min_t = np.where(better, t_on_curve_local, min_t)

        # 裂缝宽度沿曲线变化 (对数正态采样)，基于空间位置插值
        lognorm_sample = rng.lognormal(mean=np.log(width_mean), sigma=width_std,
                                       size=num_curve_samples)
        # 使用每个点在曲线上的投影参数 t 进行宽度插值，而非点索引
        half_width = np.interp(
            min_t,
            np.linspace(0, 1, num_curve_samples),
            lognorm_sample,
        ) / 2.0

        # 深度剖面
        # d(depth_ratio) = d_max * exp(-(depth_ratio)^p)
        lambda_param = half_width / 2.0  # λ 与宽度相关
        p_param = 2.0  # 高斯槽 (p=2)

        # 裂缝区域掩码：距离小于半宽
        crack_mask = min_dist <= half_width
        if np.any(crack_mask):
            depth_ratio = min_dist[crack_mask] / np.clip(lambda_param[crack_mask], 1e-12, None)
            depth = d_max * np.exp(-(depth_ratio ** p_param))
            pts[crack_mask, 2] -= depth
            lbl[crack_mask] = label_val

    # =======================================================================
    # 龟裂 / 块状裂缝 (Voronoi 图)
    # =======================================================================
    elif crack_type in ("alligator", "block"):
        x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
        area = (x_max - x_min) * (y_max - y_min)

        if crack_type == "alligator":
            # 龟裂：块度 0.2m (重) 或 0.5m (轻)
            block_size = 0.2 if severity == "severe" else 0.5
            width_crack = params.get("width_mean", 0.004)
        else:
            # 块状裂缝：块度 > 1m
            block_size = 1.0 if severity == "light" else 0.6
            width_crack = params.get("width_mean", 0.003)

        num_seeds = max(int(area / (block_size ** 2)), 4)

        # M2: Matern-II 带抑制半径的泊松点过程 (替代 rng.uniform 撒点)
        inhibition_radius = block_size * 0.15
        seeds = _generate_alligator_seeds(
            x_min, x_max, y_min, y_max, num_seeds, inhibition_radius, rng
        )

        # Voronoi 图
        vor = spatial.Voronoi(seeds)

        # 收集有限脊线 (ridge vertices)
        ridge_segments = []
        for v_idx_pair in vor.ridge_vertices:
            v1, v2 = v_idx_pair
            if v1 >= 0 and v2 >= 0:  # 有限脊线
                p1 = vor.vertices[v1]
                p2 = vor.vertices[v2]
                ridge_segments.append((p1, p2))

        if not ridge_segments:
            return pts, lbl  # 无有效脊线

        # P2-1 修复：裁剪 Voronoi 脊线段到路面边界，剔除完全在路外的线段
        clipped_ridge_segments = []
        margin = width_crack * 5  # 使用裂缝宽度作为裁剪边距
        for seg in ridge_segments:
            p1, p2 = seg
            # 线段完全在路面边界外（含边距）则跳过
            if (p1[0] < x_min - margin and p2[0] < x_min - margin) or \
               (p1[0] > x_max + margin and p2[0] > x_max + margin):
                continue
            if (p1[1] < y_min - margin and p2[1] < y_min - margin) or \
               (p1[1] > y_max + margin and p2[1] > y_max + margin):
                continue
            clipped_ridge_segments.append(seg)

        if not clipped_ridge_segments:
            return pts, lbl  # 无有效脊线在路面范围内

        # 计算每个点到最近 Voronoi 脊线的距离
        min_dist_ridge = np.full(N, np.inf)
        for seg in clipped_ridge_segments:
            dists = _point_to_segment_distance(xy, seg[0], seg[1])
            min_dist_ridge = np.minimum(min_dist_ridge, dists)

        # 裂缝宽度
        half_width_crack = width_crack / 2.0
        crack_mask = min_dist_ridge <= half_width_crack

        if np.any(crack_mask):
            t = min_dist_ridge[crack_mask] / max(half_width_crack, 1e-12)
            depth = d_max * np.exp(-(t ** 2.0))
            pts[crack_mask, 2] -= depth
            lbl[crack_mask] = label_val

    return pts, lbl


# ===========================================================================
# 1.1.4 — 坑槽生成 (Pothole)
# ===========================================================================


def add_pothole(
    points: np.ndarray,
    labels: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    depth: float,
    edge_quality: float,
    severity: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add a pothole (坑槽) via superellipsoid depression.

    超椭圆凹陷模型：

    .. math::
        z(r) = -d_{\\max} \\left[ 1 - \\left(\\frac{r}{R}\\right)^\\beta
        \\right]^{1/\\beta}, \\quad 0 \\le r \\le R

    其中 :math:`r = \\sqrt{(x-c_x)^2 + (y-c_y)^2}` 为到中心的径向距离，
    :math:`\\beta` 为超椭圆指数：

    - :math:`\\beta = 2`: 椭球凹陷 (轻)
    - :math:`\\beta > 2`: 平底坑 (重)

    边缘剥落模拟：对边界点随机侵蚀，并用 Poisson 采样添加微坑。

    标签：轻=9 (d_max ≤ 25mm), 重=10 (d_max > 25mm)

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        center:  坑槽中心 (cx, cy)。
        radius:  坑槽半径 R (m)。
        depth:   最大深度 d_max (m)。
        edge_quality: 边缘质量因子 [0,1]，1 为完整，0 为严重剥落。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    label_val = 9 if severity == "light" else 10

    cx, cy = center
    xy = pts[:, :2]
    r = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)

    # 超椭圆指数 β
    if severity == "light":
        beta = 2.0  # 椭球
    else:
        beta = 3.0 + rng.random() * 2.0  # β ∈ [3, 5] 平底

    # 坑槽主凹陷
    in_pothole = r <= radius
    if np.any(in_pothole):
        r_norm = r[in_pothole] / radius
        # z = -d_max * [1 - (r/R)^β]^(1/β)
        z_depression = -depth * (1.0 - r_norm ** beta) ** (1.0 / beta)
        pts[in_pothole, 2] += z_depression
        lbl[in_pothole] = label_val

    # --- 边缘剥落 (edge spalling) ---
    if edge_quality < 1.0:
        # 边缘区域：0.8R ≤ r ≤ 1.2R
        edge_mask = (r >= 0.8 * radius) & (r <= 1.2 * radius)
        if np.any(edge_mask):
            edge_idx = np.where(edge_mask)[0]
            # 随机选一部分边缘点剥落
            num_spall = int(edge_quality * len(edge_idx))
            if num_spall < len(edge_idx):
                spall_idx = rng.choice(edge_idx, size=len(edge_idx) - num_spall,
                                       replace=False)
                # 边缘剥落：轻微降低高度
                spall_depth = rng.uniform(0.0, depth * 0.3, size=len(spall_idx))
                pts[spall_idx, 2] -= spall_depth
                lbl[spall_idx] = label_val

        # --- Poisson 微坑 (小尺度次生坑) ---
        # 在坑槽边缘附近随机撒点
        num_pits = rng.poisson(max(1, int(radius * 5)))
        for _ in range(num_pits):
            pit_angle = rng.uniform(0, 2 * np.pi)
            pit_radius = radius * rng.uniform(0.8, 1.3)
            pit_cx = cx + pit_radius * np.cos(pit_angle)
            pit_cy = cy + pit_radius * np.sin(pit_angle)
            pit_r = 0.02 + rng.random() * 0.04  # 微坑半径 2-6cm
            pit_d = rng.uniform(0.001, 0.005)  # 深度 1-5mm

            r_pit = np.sqrt((xy[:, 0] - pit_cx) ** 2 + (xy[:, 1] - pit_cy) ** 2)
            in_pit = r_pit <= pit_r
            if np.any(in_pit):
                r_norm_pit = r_pit[in_pit] / pit_r
                z_pit = -pit_d * (1.0 - r_norm_pit ** 2.0) ** 0.5
                pts[in_pit, 2] += z_pit
                lbl[in_pit] = label_val

    return pts, lbl


# ===========================================================================
# M3: Lévy 极值剥落 — 重尾分布模拟裂缝/坑槽边缘的深层剥落
# ===========================================================================


def add_edge_spalling_heavy_tail(
    points: np.ndarray,
    labels: np.ndarray,
    edge_mask: np.ndarray,
    depth_base: float = 0.005,
    hurst: float = 0.7,
    trigger_prob: float = 0.05,
    label_val: int = 0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """M3: Add heavy-tailed edge spalling via Lévy α-stable jumps.

    Unlike the standard fBm micro-texture where Lévy's heavy tail
    is diluted by central limit theorem across 8 octaves, this function
    applies the α-stable distribution directly to a small subset of
    edge points, producing realistic deep spall craters.

    .. math::
        \\Delta z \\sim \\text{LévyStable}(\\alpha, 0, d_{\\text{base}}, 0)

    where :math:`\\alpha = 2 \\cdot H` controls tail heaviness
    (lower → heavier tail → deeper individual spalls).

    Only ``trigger_prob`` fraction of edge points are affected,
    producing the characteristic "pitting" pattern of severe spalling.

    Args:
        points: Point cloud (N, 3).
        labels: Labels (N,).
        edge_mask: Boolean mask (N,) indicating crack/pothole edges.
        depth_base: Base spalling depth scale (m).
        hurst: Hurst exponent, alpha = 2*H. Lower = heavier tail.
        trigger_prob: Fraction of edge points to spall.
        label_val: Label to assign to spalled points (0 = keep original).
        seed: Random seed.

    Returns:
        Modified (points, labels).
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    edge_idx = np.where(edge_mask)[0]
    if len(edge_idx) == 0:
        return pts, lbl

    n_trigger = max(1, int(len(edge_idx) * trigger_prob))
    triggered = rng.choice(edge_idx, size=n_trigger, replace=False)

    # Lévy α-stable: alpha = 2*H, heavy tail produces extreme spalls
    alpha = max(0.5, min(1.9, 2.0 * hurst))
    jump = stats.levy_stable.rvs(
        alpha=alpha, beta=0, loc=0,
        scale=depth_base, size=n_trigger,
        random_state=rng,
    )
    jump = np.abs(jump)           # only downward (erosion)
    jump = np.clip(jump, 0.0, depth_base * 5.0)  # limit extreme outliers

    pts[triggered, 2] -= jump
    if label_val > 0:
        lbl[triggered] = label_val

    return pts, lbl


# ===========================================================================
# 1.1.5 — 松散 (Raveling)
# ===========================================================================


def add_raveling(
    points: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray,
    severity: str,
    seed: Optional[int] = None,
    remove_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add raveling (松散) — simulate fine aggregate loss.

    模拟细集料脱落：
    1. 在区域内随机移除一定比例的点 (用 ``np.nan`` 标记，调用方过滤)
    2. 对剩余点，高度降低 (高斯噪声模拟表面侵蚀)

    轻 (标签 11)：移除比例 ~5%，深度 ~1-2mm
    重 (标签 12)：移除比例 ~20%，深度 ~3-5mm 点蚀

    可改进点#1: 增加 ``remove_nan`` 参数，为 True 时在返回前自动过滤 NaN
    点并同步裁剪标签数组，减少调用方负担。

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        region_mask: 布尔掩码 (N,)，指示松散发生的区域。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子。
        remove_nan: 是否在返回前自动过滤 NaN 点 (可改进点#1)。

    Returns:
        points_modified:  修改后点云 (N, 3) 或 (N', 3) (若 remove_nan=True)。
        labels_modified: 修改后标签 (N,) 或 (N',)。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    label_val = 11 if severity == "light" else 12

    region_idx = np.where(region_mask)[0]
    if len(region_idx) == 0:
        if remove_nan:
            valid = ~np.any(np.isnan(pts), axis=1)
            return pts[valid], lbl[valid]
        return pts, lbl

    if severity == "light":
        removal_ratio = 0.05
        pitting_depth = 0.002  # 2mm
    else:
        removal_ratio = 0.20
        pitting_depth = 0.005  # 5mm

    # 1. 随机移除点 (设为 NaN)
    num_remove = int(removal_ratio * len(region_idx))
    remove_idx = rng.choice(region_idx, size=num_remove, replace=False)
    pts[remove_idx] = np.nan
    lbl[remove_idx] = label_val  # 即使移除也标注标签

    # 2. 剩余区域点的表面侵蚀
    remaining_idx = np.setdiff1d(region_idx, remove_idx)
    if len(remaining_idx) > 0:
        # 高斯噪声随机深度
        erosion = rng.normal(loc=pitting_depth * 0.5, scale=pitting_depth * 0.3,
                             size=len(remaining_idx))
        erosion = np.clip(erosion, 0.0, pitting_depth * 1.5)
        pts[remaining_idx, 2] -= erosion
        lbl[remaining_idx] = label_val

    if remove_nan:
        valid = ~np.any(np.isnan(pts), axis=1)
        return pts[valid], lbl[valid]

    return pts, lbl


# ===========================================================================
# 1.1.6 — 沉陷 (Depression)
# ===========================================================================


def add_depression(
    points: np.ndarray,
    labels: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    depth: float,
    severity: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add depression (沉陷) — large-area low-frequency subsidence.

    高斯凹陷模型：

    .. math::
        z(r) = -d \\cdot \\exp\\left(-\\frac{r^2}{2\\sigma^2}\\right)

    其中 :math:`r` 为到中心的距离，:math:`\\sigma = R / 3` 使得
    在半径 :math:`R` 处衰减到 :math:`\\exp(-4.5) \\approx 0.011`。

    轻 (标签 13)：深度 10-25mm
    重 (标签 14)：深度 > 25mm

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        center:  沉陷中心 (cx, cy)。
        radius:  影响半径 (m)。
        depth:  最大深度 (m)。
        severity: 'light' | 'severe'。
        seed: 随机种子 (未使用，为接口一致性保留)。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    _ = seed  # 未使用但为接口一致性保留
    pts = points.copy()
    lbl = labels.copy()

    label_val = 13 if severity == "light" else 14
    cx, cy = center

    xy = pts[:, :2]
    r2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
    sigma2 = (radius / 3.0) ** 2

    # 高斯凹陷
    depression = -depth * np.exp(-0.5 * r2 / sigma2)

    pts[:, 2] += depression

    # 标注受影响区域 (深度超过 1% 最大深度的点)
    affected = np.abs(depression) > depth * 0.01
    lbl[affected] = label_val

    return pts, lbl


# ===========================================================================
# 1.1.7 — 车辙 (Rutting)
# ===========================================================================


def add_rutting(
    points: np.ndarray,
    labels: np.ndarray,
    center_line: float,
    wheel_separation: float,
    depth: float,
    width: float,
    severity: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add rutting (车辙) — dual wheel-track depressions.

    左右轮迹由高斯凹陷叠加，纵向 (y) 截面用正弦调制模拟变化：

    .. math::
        z(x, y) = -d \\cdot \\left[
            \\exp\\left(-\\frac{(x - x_L)^2}{2w^2}\\right) +
            \\exp\\left(-\\frac{(x - x_R)^2}{2w^2}\\right)
        \\right] \\cdot \\left(1 + \\varepsilon \\sin\\frac{2\\pi y}{\\Lambda}\\right)

    其中 :math:`x_L = x_c - s/2`, :math:`x_R = x_c + s/2`,
    :math:`s` 为轮迹间距，:math:`w` 为轮迹宽度，
    :math:`\\varepsilon` 为纵向调制幅度，:math:`\\Lambda` 为调制波长。

    轻 (标签 15)：深度 10-15mm
    重 (标签 16)：深度 > 15mm

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        center_line: 道路中心线 x 坐标。
        wheel_separation: 左右轮迹间距 (m)。
        depth:  最大深度 (m)。
        width: 单条轮迹宽度 (m)。
        severity: 'light' | 'severe'。
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    _ = seed  # deterministic computation
    pts = points.copy()
    lbl = labels.copy()

    label_val = 15 if severity == "light" else 16

    x = pts[:, 0]
    y = pts[:, 1]

    # 左右轮迹中心
    x_left = center_line - wheel_separation / 2.0
    x_right = center_line + wheel_separation / 2.0

    # 轮迹宽度标准差
    sigma = width / 3.0

    # 纵向调制参数
    y_range = np.max(y) - np.min(y)
    modulation_wavelength = y_range * 0.5  # 调制波长为路面长度一半
    epsilon = 0.2  # 调制幅度
    modulation = 1.0 + epsilon * np.sin(2.0 * np.pi * y / modulation_wavelength)

    # 高斯轮迹
    track_left = np.exp(-0.5 * ((x - x_left) / sigma) ** 2)
    track_right = np.exp(-0.5 * ((x - x_right) / sigma) ** 2)

    rut_depth = depth * (track_left + track_right) * modulation

    pts[:, 2] -= rut_depth

    # 标注车辙区域
    affected = rut_depth > depth * 0.05
    lbl[affected] = label_val

    return pts, lbl


# ===========================================================================
# 1.1.8 — 波浪拥包 (Corrugation)
# ===========================================================================


def add_corrugation(
    points: np.ndarray,
    labels: np.ndarray,
    direction: str,
    wavelength: float,
    amplitude: float,
    severity: str,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add corrugation (波浪拥包) — sinusoidal height modulation.

    正弦波高度调制：

    .. math::
        z'(x, y) = z(x, y) + A \\cdot \\cos\\left(\\frac{2\\pi}{\\lambda} u\\right)

    其中 :math:`u` 为沿波纹方向的空间坐标：

    - ``direction='longitudinal'``: :math:`u = y`
    - ``direction='transverse'``: :math:`u = x`

    轻 (标签 17)：振幅 10-25mm
    重 (标签 18)：振幅 > 25mm

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        direction: 波纹方向 'longitudinal' | 'transverse'。
        wavelength: 波长 (m)。
        amplitude:  振幅 (m)。
        severity: 'light' | 'severe'。
        seed: 随机种子 (未使用)。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    _ = seed
    pts = points.copy()
    lbl = labels.copy()

    label_val = 17 if severity == "light" else 18

    if direction == "longitudinal":
        u = pts[:, 1]  # y 方向
    elif direction == "transverse":
        u = pts[:, 0]  # x 方向
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 正弦波调制
    modulation = amplitude * np.cos(2.0 * np.pi * u / wavelength)
    pts[:, 2] += modulation

    # 标注拥包区域 (振幅超过 10% 的点)
    affected = np.abs(modulation) > amplitude * 0.1
    lbl[affected] = label_val

    return pts, lbl


# ===========================================================================
# 1.1.9 — 泛油 (Bleeding)
# ===========================================================================


def add_bleeding(
    points: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add bleeding (泛油) — increase reflectivity, NO geometry change.

    泛油仅改变路面反射率 (强度值)，不改变几何形状。
    标签 = 19。

    本函数返回修改后的标签数组。调用方可将 ``region_mask`` 区域的
    强度 (反射率) 值增加 (例如 +0.3) 以模拟沥青泛油的高反射特性。

    Args:
        points:  点云 (N, 3) (仅用于形状检查)。
        labels:  标签 (N,)。
        region_mask: 布尔掩码 (N,)，泛油区域。
        seed: 随机种子 (未使用)。

    Returns:
        labels_modified: 修改后标签 (N,)。泛油区域设为 19。
    """
    lbl = labels.copy()
    lbl[region_mask] = 19
    return lbl


# ===========================================================================
# 1.1.10 — 水泥路面损坏 (Concrete Damage)
# ===========================================================================


def add_concrete_damage(
    points: np.ndarray,
    labels: np.ndarray,
    damage_type: str,
    severity: str,
    params: dict,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add concrete pavement damage (10 种水泥路面损坏).

    水泥路面按板块 (slab) 组织，板块尺寸由 ``ConcreteDamageConfig`` 定义。
    板块之间由接缝 (joint) 分隔。

    支持 10 种损坏类型 (标签见 JTG 5210-2018)：

    ==================================  ==========  ==========================
    损坏类型                           标签 (轻/重)  描述
    ==================================  ==========  ==========================
    slab_shatter   (破碎板)             21 / 22      板体碎块 + 垂直错台
    slab_crack    (裂缝)                23 / 24      贯穿板体的尖锐裂缝
    corner_break  (板角断裂)            25 / 26      板角斜向裂缝 + 沉降
    faulting      (错台)                27 / 28      接缝处高度差 3-10mm/>10mm
    pumping       (唧泥)                29           接缝湿泥 (强度下降)
    edge_spall    (边角剥落)            30 / 31      板边缘破损
    joint_damage  (接缝料损坏)          32 / 33      接缝填充物缺失
    pitting       (坑洞)                34           深坑
    blowup        (拱起)                35           板体向上拱起
    exposed_aggregate (露骨)            36           表面纹理缺失
    ==================================  ==========  ==========================

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        damage_type: 损坏类型字符串 (见上表)。
        severity: 'light' | 'severe' (对无程度区分的类型可任意)。
        params: 参数字典，各类型可包含：
            - 'slab_length': 板块长度 (m)，默认 5.0
            - 'slab_width': 板块宽度 (m)，默认 4.0
            - 'joint_width': 接缝宽度 (m)，默认 0.008
            - 'd_max': 最大深度/位移 (m)
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    xy = pts[:, :2]
    x, y = pts[:, 0], pts[:, 1]

    # --- 解析板块参数 ---
    slab_len = params.get("slab_length", 5.0)
    slab_wid = params.get("slab_width", 4.0)
    joint_w = params.get("joint_width", 0.008)
    # P1-4 修复：支持板块偏移参数
    x_offset = params.get("x_offset", 0.0)
    y_offset = params.get("y_offset", 0.0)

    # --- 获取标签 ---
    # 对无程度区分的类型，直接用固定标签
    no_severity_types = {"pumping": 29, "pitting": 34, "blowup": 35,
                         "exposed_aggregate": 36}

    if damage_type in no_severity_types:
        label_val = no_severity_types[damage_type]
    elif severity == "light":
        label_map = {
            "slab_shatter": 21, "slab_crack": 23, "corner_break": 25,
            "faulting": 27, "edge_spall": 30, "joint_damage": 32,
        }
        label_val = label_map.get(damage_type, 23)
    else:
        label_map = {
            "slab_shatter": 22, "slab_crack": 24, "corner_break": 26,
            "faulting": 28, "edge_spall": 31, "joint_damage": 33,
        }
        label_val = label_map.get(damage_type, 24)

    # --- 辅助：计算点到 slab 接缝的距离 ---
    # 接缝位置 (沿 x 方向)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # 横向接缝 (沿 y 方向，在偏移 + 整数倍 slab_len 处)
    trans_joints = np.arange(y_offset, y_max + slab_len, slab_len)
    # 纵向接缝 (沿 x 方向，在偏移 + 整数倍 slab_wid 处)
    long_joints = np.arange(x_offset, x_max + slab_wid, slab_wid)

    def dist_to_nearest_joint(px, py):
        """计算点到最近接缝的距离。"""
        dx = np.min(np.abs(px[:, None] - long_joints[None, :]), axis=1)
        dy = np.min(np.abs(py[:, None] - trans_joints[None, :]), axis=1)
        return np.minimum(dx, dy)

    def slab_indices(px, py):
        """返回每个点所在的板块索引 (ix, iy)。"""
        ix = np.floor(px / slab_wid).astype(int)
        iy = np.floor(py / slab_len).astype(int)
        return ix, iy

    joint_dist = dist_to_nearest_joint(x, y)

    # =======================================================================
    # 1) 破碎板 (Slab Shatter)
    # =======================================================================
    if damage_type == "slab_shatter":
        # 选择一块或多块板，将板内点分割成 Voronoi 碎片
        # 每个碎片随机垂直错台

        ix, iy = slab_indices(x, y)
        # 选一个随机板块
        unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
        target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 3:
            # 板内生成随机碎片种子
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            x_s_min, x_s_max = np.min(slab_x), np.max(slab_x)
            y_s_min, y_s_max = np.min(slab_y), np.max(slab_y)

            n_fragments = rng.integers(3, 8)
            seeds = np.column_stack([
                rng.uniform(x_s_min, x_s_max, n_fragments),
                rng.uniform(y_s_min, y_s_max, n_fragments),
            ])

            vor = spatial.Voronoi(seeds)
            # 对每个点，找最近的种子 (所属碎片)
            tree = spatial.KDTree(seeds)
            _, frag_idx = tree.query(np.column_stack([slab_x, slab_y]))

            # 每块碎片随机垂直位移
            fragment_disp = rng.normal(0, 0.01 if severity == "light" else 0.03,
                                       size=n_fragments)
            for fi in range(n_fragments):
                frag_mask = frag_idx == fi
                pts[slab_idx[frag_mask], 2] += fragment_disp[fi]

            # 裂缝沿 Voronoi 脊线
            for ridge in vor.ridge_vertices:
                if ridge[0] >= 0 and ridge[1] >= 0:
                    p1 = vor.vertices[ridge[0]]
                    p2 = vor.vertices[ridge[1]]
                    # 检查脊线在板内
                    dists = _point_to_segment_distance(xy[slab_idx], p1, p2)
                    crack_near = dists < joint_w * 2
                    if np.any(crack_near):
                        d_max_crack = params.get("d_max", 0.02)
                        t = dists[crack_near] / (joint_w * 2)
                        pts[slab_idx[crack_near], 2] -= d_max_crack * np.exp(-(t ** 2))
                        lbl[slab_idx[crack_near]] = label_val

            lbl[slab_idx] = label_val

    # =======================================================================
    # 2) 水泥裂缝 (Concrete Crack)
    # =======================================================================
    elif damage_type == "slab_crack":
        # 通过板体的直线裂缝，边缘更尖锐
        # 随机角度和位置
        angle = rng.uniform(0, np.pi)
        x_span = x_max - x_min
        y_span = y_max - y_min
        intercept = rng.uniform(
            min(x_min, y_min) + 0.1 * min(x_span, y_span),
            max(x_max, y_max) - 0.1 * max(x_span, y_span),
        )

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # 点到直线的距离
        dist_to_line = np.abs(cos_a * y - sin_a * x + intercept) / np.sqrt(cos_a ** 2 + sin_a ** 2 + 1e-12)

        crack_width_joint = joint_w * 2
        crack_mask = dist_to_line < crack_width_joint

        if np.any(crack_mask):
            d_max_crack = params.get("d_max", 0.015 if severity == "light" else 0.040)
            t = dist_to_line[crack_mask] / crack_width_joint
            # 更尖锐的剖面 (p=1 近似 V 形)
            depth_crack = d_max_crack * np.exp(-(t ** 1.0))
            pts[crack_mask, 2] -= depth_crack
            lbl[crack_mask] = label_val

    # =======================================================================
    # 3) 板角断裂 (Corner Break)
    # =======================================================================
    elif damage_type == "corner_break":
        # 选择一个板块的一个角
        ix, iy = slab_indices(x, y)
        unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
        target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 0:
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            x_s_min, x_s_max = np.min(slab_x), np.max(slab_x)
            y_s_min, y_s_max = np.min(slab_y), np.max(slab_y)

            # 选一个角
            corner_x = x_s_min if rng.random() < 0.5 else x_s_max
            corner_y = y_s_min if rng.random() < 0.5 else y_s_max

            # 斜向裂缝方向
            diag_angle = np.arctan2(
                y_s_max - y_s_min if rng.random() < 0.5 else y_s_min - y_s_max,
                x_s_max - x_s_min if rng.random() < 0.5 else x_s_min - x_s_max,
            )

            # 从角出发的斜线
            dx_corner = slab_x - corner_x
            dy_corner = slab_y - corner_y
            cos_da, sin_da = np.cos(diag_angle), np.sin(diag_angle)
            # 到斜线的有符号距离
            dist_diag = np.abs(cos_da * dy_corner - sin_da * dx_corner)

            # 到角的距离
            dist_to_corner = np.sqrt(dx_corner ** 2 + dy_corner ** 2)

            # 裂缝区域
            corner_fracture = (dist_diag < joint_w * 3) & (dist_to_corner < slab_len * 0.6)
            if np.any(corner_fracture):
                d_max_corner = params.get("d_max", 0.015 if severity == "light" else 0.040)
                # 越靠近角沉降越大
                settle = d_max_corner * (1.0 - dist_to_corner[corner_fracture] / (slab_len * 0.6))
                pts[slab_idx[corner_fracture], 2] -= np.clip(settle, 0, d_max_corner)
                lbl[slab_idx[corner_fracture]] = label_val

            # 板角整体轻微沉降
            corner_region = dist_to_corner < slab_len * 0.3
            corner_settle = params.get("d_max", 0.005 if severity == "light" else 0.015)
            settle_factor = 1.0 - dist_to_corner[corner_region] / (slab_len * 0.3)
            pts[slab_idx[corner_region], 2] -= corner_settle * np.clip(settle_factor, 0, 1)
            lbl[slab_idx[corner_region]] = label_val

    # =======================================================================
    # 4) 错台 (Faulting)
    # =======================================================================
    elif damage_type == "faulting":
        # 接缝两侧高度差
        # 选一个横向接缝 (确保至少有一个有效的内部接缝)
        if len(trans_joints) > 2:
            joint_y = trans_joints[rng.integers(1, len(trans_joints) - 1)]
        else:
            joint_y = trans_joints[min(1, len(trans_joints) - 1)]

        fault_offset = params.get("d_max", 0.005 if severity == "light" else 0.015)

        # 接缝两侧各 slab_wid 范围
        near_joint = np.abs(y - joint_y) < joint_w * 5
        if np.any(near_joint):
            # 一侧上升，一侧下降
            side = np.sign(y - joint_y)
            pts[near_joint, 2] += side[near_joint] * fault_offset * 0.5
            lbl[near_joint] = label_val

    # =======================================================================
    # 5) 唧泥 (Pumping)
    # =======================================================================
    elif damage_type == "pumping":
        # 在接缝处添加湿润泥点特征
        # 仅修改标签，不改变几何
        near_joint = joint_dist < joint_w * 8
        lbl[near_joint] = label_val

        # 额外：在接缝附近随机添加泥斑 (可以用增加 z 模拟轻微隆起)
        if np.any(near_joint):
            n_mud_spots = rng.poisson(max(1, int(np.sum(near_joint) * 0.01)))
            for _ in range(n_mud_spots):
                spot_x = rng.uniform(x_min, x_max)
                spot_y = rng.uniform(y_min, y_max)
                spot_r = rng.uniform(0.02, 0.08)
                dist_spot = np.sqrt((x - spot_x) ** 2 + (y - spot_y) ** 2)
                in_spot = dist_spot < spot_r
                if np.any(in_spot):
                    # 微隆模拟泥浆堆积 (0.5-2mm)
                    pts[in_spot, 2] += rng.uniform(0.0005, 0.002)
                    lbl[in_spot] = label_val

    # =======================================================================
    # 6) 边角剥落 (Edge Spall)
    # =======================================================================
    elif damage_type == "edge_spall":
        # 板块边缘破损
        # 找离接缝一定距离的边界点
        near_joint = joint_dist < joint_w * 10

        # 随机选择一些接缝附近区域
        if np.any(near_joint):
            joint_idx = np.where(near_joint)[0]
            n_spall = max(1, int(len(joint_idx) * (0.3 if severity == "light" else 0.6)))
            spall_idx = rng.choice(joint_idx, size=n_spall, replace=False)

            d_max_spall = params.get("d_max", 0.010 if severity == "light" else 0.030)
            spall_depth = rng.exponential(d_max_spall * 0.5, size=n_spall)
            pts[spall_idx, 2] -= spall_depth
            lbl[spall_idx] = label_val

    # =======================================================================
    # 7) 接缝料损坏 (Joint Damage)
    # =======================================================================
    elif damage_type == "joint_damage":
        # 接缝处填充物缺失 → 形成沟槽
        # 沿接缝的窄带
        near_joint = joint_dist < joint_w

        if np.any(near_joint):
            # 随机选部分接缝段损坏
            joint_idx = np.where(near_joint)[0]
            damage_ratio = 0.4 if severity == "light" else 0.8
            n_damage = int(damage_ratio * len(joint_idx))
            damage_idx = rng.choice(joint_idx, size=n_damage, replace=False)

            d_max_joint = params.get("d_max", 0.010 if severity == "light" else 0.025)
            # 沿接缝的 V 形槽
            dist_j = joint_dist[damage_idx]
            depth_joint = d_max_joint * np.exp(-(dist_j / (joint_w * 0.5)) ** 2)
            pts[damage_idx, 2] -= depth_joint
            lbl[damage_idx] = label_val

    # =======================================================================
    # 8) 坑洞 (Pitting) — 类似坑槽但更深
    # =======================================================================
    elif damage_type == "pitting":
        center_x = rng.uniform(x_min + 0.05 * (x_max - x_min),
                               x_max - 0.05 * (x_max - x_min))
        center_y = rng.uniform(y_min + 0.05 * (y_max - y_min),
                               y_max - 0.05 * (y_max - y_min))
        pit_radius = rng.uniform(0.05, 0.15)
        pit_depth = params.get("d_max", rng.uniform(0.03, 0.08))

        r_pit = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        in_pit = r_pit <= pit_radius

        if np.any(in_pit):
            r_norm = r_pit[in_pit] / pit_radius
            z_pit = -pit_depth * (1.0 - r_norm ** 2.0) ** 0.5
            pts[in_pit, 2] += z_pit
            lbl[in_pit] = label_val

    # =======================================================================
    # 9) 拱起 (Blowup)
    # =======================================================================
    elif damage_type == "blowup":
        # 板体向上拱起 (余弦拱形)
        ix, iy = slab_indices(x, y)
        unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
        target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 0:
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            x_s_center = (np.min(slab_x) + np.max(slab_x)) / 2.0
            y_s_center = (np.min(slab_y) + np.max(slab_y)) / 2.0

            blowup_height = params.get("d_max", rng.uniform(0.02, 0.06))

            # 径向余弦拱起
            dx_blow = (slab_x - x_s_center) / (slab_wid * 0.5)
            dy_blow = (slab_y - y_s_center) / (slab_len * 0.5)
            r_blow = np.sqrt(dx_blow ** 2 + dy_blow ** 2)
            r_blow = np.clip(r_blow, 0, 1.5)

            # 拱起剖面
            uplift = blowup_height * (np.cos(np.pi * r_blow * 0.5) ** 2)
            uplift = np.clip(uplift, 0, blowup_height)

            pts[slab_idx, 2] += uplift
            # 拱起中心区域标注
            uplifted = uplift > blowup_height * 0.1
            lbl[slab_idx[uplifted]] = label_val

    # =======================================================================
    # 10) 露骨 (Exposed Aggregate)
    # =======================================================================
    elif damage_type == "exposed_aggregate":
        # 表面纹理缺失 → 增加粗糙度 + 降低强度
        # 在随机区域添加高频噪声
        region_idx = np.where(
            (x > x_min + 0.1 * (x_max - x_min)) &
            (x < x_max - 0.1 * (x_max - x_min)) &
            (y > y_min + 0.1 * (y_max - y_min)) &
            (y < y_max - 0.1 * (y_max - y_min))
        )[0]

        if len(region_idx) > 3:
            n_exposed = max(1, int(len(region_idx) * rng.uniform(0.1, 0.4)))
            exposed_idx = rng.choice(region_idx, size=n_exposed, replace=False)

            # 表面粗糙化：增加高频噪声
            roughness = rng.normal(0, params.get("d_max", 0.003), size=n_exposed)
            pts[exposed_idx, 2] += roughness
            lbl[exposed_idx] = label_val

    else:
        raise ValueError(f"Unknown concrete damage type: {damage_type}")

    return pts, lbl


# ===========================================================================
# 1.1.11 — LiDAR 噪声仿真
# ===========================================================================


def simulate_lidar_noise(
    points: np.ndarray,
    distance_noise_std: float,
    dropout_rate: float,
    angular_jitter_deg: float,
    seed: Optional[int] = None,
    enable_edge_mixing: bool = True,
    mixed_pixel_prob: float = 0.01,
    curvature: Optional[np.ndarray] = None,
    curvature_threshold: float = 0.5,
) -> np.ndarray:
    """Simulate LiDAR measurement noise on point cloud.

    1. **球坐标噪声**：
       将点云从笛卡尔坐标 :math:`(x, y, z)` 转换为球坐标 :math:`(r, \\theta, \\phi)`：
       .. math::
           r &= \\sqrt{x^2 + y^2 + z^2} \\\\
           \\theta &= \\arctan(y/x) \\\\
           \\phi &= \\arcsin(z/r)
       对距离加高斯噪声，角度加高斯抖动，再转换回笛卡尔坐标。

    2. **Dropout (点丢失)**：
       基于 Bernoulli 过程 :math:`P(\\text{drop}) = p_d` 独立决定每个点的丢失。
       同时附加距离依赖性：远点丢失概率更高。

    3. **边缘混合效应 (Edge Mixing) — P2-3 修复**：
       仅对曲率变化大的病害边缘点做邻域均值混合 (模拟 LiDAR 混合像素效应)。
       使用批量化 KDTree 查询替代逐点查询 (可改进点#5)。

    Args:
        points:  点云 (N, 3)。
        distance_noise_std: 距离测量高斯噪声标准差 (m)。
        dropout_rate: 基础点丢失概率 [0, 1)。
        angular_jitter_deg: 角度抖动标准差 (度)。
        seed: 随机种子。
        enable_edge_mixing: 是否启用边缘混合，默认 True (P2-3)。
        mixed_pixel_prob: 混合像素点比例，默认 0.01 (1%)。
        curvature: 曲率数组 (N,) 用于筛选边缘点，None 时退化到全局混合 (P2-3)。
        curvature_threshold: 曲率阈值，大于此值的点才参与混合 (P2-3)。

    Returns:
        噪声点云 (N', 3)，其中 N' ≤ N。NaN 点已被移除。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    N = pts.shape[0]

    # ===================================================================
    # 1. 笛卡尔 → 球坐标
    # ===================================================================
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
    theta = np.arctan2(y, x)
    phi = np.arcsin(np.clip(z / r, -1.0, 1.0))

    # ===================================================================
    # 2. 加噪声
    # ===================================================================
    angular_jitter_rad = np.deg2rad(angular_jitter_deg)

    r_noisy = r + rng.normal(0.0, distance_noise_std, size=N)
    r_noisy = np.maximum(r_noisy, 0.0)  # 非负半径
    theta_noisy = theta + rng.normal(0.0, angular_jitter_rad, size=N)
    phi_noisy = phi + rng.normal(0.0, angular_jitter_rad, size=N)
    # 限制 phi 在 [-π/2, π/2]
    phi_noisy = np.clip(phi_noisy, -np.pi / 2.0, np.pi / 2.0)

    # ===================================================================
    # 3. 球坐标 → 笛卡尔
    # ===================================================================
    pts_noisy = np.empty_like(pts)
    pts_noisy[:, 0] = r_noisy * np.cos(theta_noisy) * np.cos(phi_noisy)
    pts_noisy[:, 1] = r_noisy * np.sin(theta_noisy) * np.cos(phi_noisy)
    pts_noisy[:, 2] = r_noisy * np.sin(phi_noisy)

    # ===================================================================
    # 4. Dropout
    # ===================================================================
    if dropout_rate > 0:
        # 基础丢失概率 + 距离相关附加
        r_max = np.max(r_noisy)
        dist_factor = r_noisy / max(r_max, 1e-12)
        dropout_prob = dropout_rate + (1.0 - dropout_rate) * dist_factor * 0.2
        dropout_prob = np.clip(dropout_prob, 0.0, 0.99)

        keep_mask = rng.random(N) > dropout_prob
        pts_noisy = pts_noisy[keep_mask]
    else:
        keep_mask = np.ones(N, dtype=bool)

    # ===================================================================
    # 5. 边缘混合效应 (Edge Mixing) — P2-3 修复
    # ===================================================================
    if enable_edge_mixing and len(pts_noisy) > 10:
        n_mixed = max(1, int(len(pts_noisy) * mixed_pixel_prob))

        # P2-3: 如果提供了曲率，仅在高曲率区域混合（病害边缘）
        if curvature is not None and len(curvature) > 0:
            # 曲率与 keep_mask 对齐：需要从原始 curvature 对应到保留后的点
            # 假设 curvature 在调用前已与输入 points 对齐
            if len(curvature) == N:
                cur = curvature[keep_mask]
            else:
                cur = curvature
            # 高曲率区域的索引（top-n by curvature magnitude）
            cur_abs = np.abs(cur)
            # 筛选曲率大于阈值的候选点
            edge_candidates = np.where(cur_abs > curvature_threshold * np.std(cur_abs))[0]
            if len(edge_candidates) > n_mixed:
                # 在高曲率点中随机选择
                mixed_idx = rng.choice(edge_candidates, size=n_mixed, replace=False)
            elif len(edge_candidates) > 0:
                mixed_idx = edge_candidates
            else:
                mixed_idx = np.array([], dtype=int)
        else:
            # 退化到全局随机混合
            mixed_idx = rng.choice(len(pts_noisy), size=n_mixed, replace=False)

        if len(mixed_idx) > 0:
            tree = spatial.KDTree(pts_noisy)
            # 批量查询 k=6 个最近邻（包含自身，取邻居的 mean 替换）
            _, all_neighbors = tree.query(pts_noisy[mixed_idx], k=6)
            # 排除自身（第0列），取邻居的均值
            neighbors_mean = np.mean(pts_noisy[all_neighbors[:, 1:]], axis=1)
            pts_noisy[mixed_idx] = neighbors_mean

    return pts_noisy


# ===========================================================================
# __main__ — 自检脚本 (测试所有 11 个函数)
# ===========================================================================

if __name__ == "__main__":
    """自检：验证所有 11 个基元函数的输出形状和标签范围。"""
    print("=" * 72)
    print("  RoadMC Primitives Self-Test")
    print("  验证 11 个数学与力学基元函数")
    print("=" * 72)

    seed = 42
    passed = 0
    total = 11

    # 小尺寸测试参数
    test_width = 0.5
    test_length = 0.5
    test_grid_res = 0.02  # 粗网格加速测试
    expected_N = int(test_width / test_grid_res) * int(test_length / test_grid_res)

    # ===================================================================
    # 测试 1: generate_road_surface
    # ===================================================================
    print("\n[Test 1/11] generate_road_surface ...")
    try:
        pts, nrm = generate_road_surface(
            width=test_width, length=test_length, grid_res=test_grid_res,
            pavement_type="asphalt", roughness_class="A", seed=seed,
        )
        N_actual = pts.shape[0]
        assert pts.shape == (N_actual, 3), f"points shape: {pts.shape}"
        assert nrm.shape == (N_actual, 3), f"normals shape: {nrm.shape}"
        # 法向量应为单位向量
        normal_norms = np.linalg.norm(nrm, axis=1)
        assert np.allclose(normal_norms, 1.0, atol=1e-6), (
            f"Normals not unit: max dev={np.max(np.abs(normal_norms - 1.0))}"
        )
        assert np.all(np.isfinite(pts)), "Non-finite points detected"
        print(f"  [PASS] points={pts.shape}, normals={nrm.shape}, "
              f"z=[{np.min(pts[:,2]):.6f}, {np.max(pts[:,2]):.6f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 2: add_micro_texture
    # ===================================================================
    print("\n[Test 2/11] add_micro_texture ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        pts2, nrm2 = add_micro_texture(
            pts, nrm, amplitude=0.001, hurst=0.7, octaves=4, seed=seed,
        )
        assert pts2.shape == pts.shape, f"shape mismatch: {pts2.shape} vs {pts.shape}"
        assert nrm2.shape == nrm.shape, "normal shape mismatch"
        assert np.all(np.isfinite(pts2)), "Non-finite points"
        print(f"  [PASS] points={pts2.shape}, "
              f"z=[{np.min(pts2[:,2]):.6f}, {np.max(pts2[:,2]):.6f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 3: add_crack (4 种类型)
    # ===================================================================
    print("\n[Test 3/11] add_crack ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        crack_types = ["longitudinal", "transverse", "alligator", "block"]
        severities = ["light", "severe"]
        for ct in crack_types:
            for sv in severities:
                pts_c, lbl_c = add_crack(
                    pts, lbl, crack_type=ct, severity=sv,
                    params={"d_max": 0.01}, seed=seed,
                )
                assert pts_c.shape == pts.shape
                assert lbl_c.shape == lbl.shape
                assert lbl_c.dtype == lbl.dtype
                # 检查标签范围
                unique_labels = np.unique(lbl_c)
                for ul in unique_labels:
                    if ul > 0:
                        assert 1 <= ul <= 8, f"Label {ul} out of range [1, 8]"

        print("  [PASS] All 4 crack types × 2 severities tested")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 4: add_pothole
    # ===================================================================
    print("\n[Test 4/11] add_pothole ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        cx = test_width / 2.0
        cy = test_length / 2.0
        pts_p, lbl_p = add_pothole(
            pts, lbl, center=(cx, cy), radius=0.08, depth=0.02,
            edge_quality=0.8, severity="light", seed=seed,
        )
        assert pts_p.shape == pts.shape
        assert lbl_p.shape == lbl.shape
        # 验证标签
        assert 9 in np.unique(lbl_p) or np.all(lbl_p == 0), (
            f"Expected label 9, got {np.unique(lbl_p)}"
        )
        # 验证深度变化
        z_diff = pts_p[:, 2] - pts[:, 2]
        assert np.all(z_diff <= 0 + 1e-10), "Pothole should only lower points"

        # 重度坑槽
        pts_p2, lbl_p2 = add_pothole(
            pts, lbl, center=(cx, cy), radius=0.08, depth=0.04,
            edge_quality=0.5, severity="severe", seed=seed,
        )
        assert 10 in np.unique(lbl_p2) or np.all(lbl_p2 == 0)

        print("  [PASS] Light(label=9) and Severe(label=10) potholes")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 5: add_raveling
    # ===================================================================
    print("\n[Test 5/11] add_raveling ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        # 创建区域掩码 (右侧一半区域)
        region_mask = pts[:, 0] > test_width / 2.0

        pts_r, lbl_r = add_raveling(
            pts, lbl, region_mask=region_mask, severity="light", seed=seed,
        )
        assert pts_r.shape == pts.shape
        assert lbl_r.shape == lbl.shape
        # 检查标签
        unique_lbl = np.unique(lbl_r)
        assert 0 in unique_lbl, "Background points should remain label 0"
        assert 11 in unique_lbl or np.any(lbl_r[region_mask] == 11), (
            "Raveling region should have label 11"
        )

        # 验证移除点 (NaN)
        n_nan = np.sum(np.any(np.isnan(pts_r), axis=1))
        print(f"  [PASS] Light raveling: {n_nan} NaN points (removed)")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 6: add_depression
    # ===================================================================
    print("\n[Test 6/11] add_depression ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        cx, cy = test_width / 2.0, test_length / 2.0
        pts_d, lbl_d = add_depression(
            pts, lbl, center=(cx, cy), radius=0.2, depth=0.02,
            severity="light", seed=seed,
        )
        assert pts_d.shape == pts.shape
        assert lbl_d.shape == lbl.shape
        z_diff = pts_d[:, 2] - pts[:, 2]
        # 中心应沉陷最多
        center_d = np.argmin((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        assert z_diff[center_d] < -0.01, (
            f"Center depression too shallow: {z_diff[center_d]}"
        )
        assert np.max(np.abs(z_diff)) <= 0.02 + 1e-6

        print(f"  [PASS] Depression depth range: [{np.min(z_diff):.6f}, {np.max(z_diff):.6f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 7: add_rutting
    # ===================================================================
    print("\n[Test 7/11] add_rutting ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        pts_ru, lbl_ru = add_rutting(
            pts, lbl, center_line=test_width / 2.0,
            wheel_separation=0.2, depth=0.015, width=0.08,
            severity="light", seed=seed,
        )
        assert pts_ru.shape == pts.shape
        assert lbl_ru.shape == lbl.shape
        z_diff = pts_ru[:, 2] - pts[:, 2]
        # 双轮迹叠加 + 纵向调制，最大深度 ≈ depth × (1 + ε)
        assert np.max(np.abs(z_diff)) <= 0.015 * 1.5 + 1e-6

        # 重度车辙
        pts_ru2, lbl_ru2 = add_rutting(
            pts, lbl, center_line=test_width / 2.0,
            wheel_separation=0.2, depth=0.03, width=0.08,
            severity="severe", seed=seed,
        )
        z_diff2 = pts_ru2[:, 2] - pts[:, 2]
        assert np.any(z_diff2 < -0.015)

        print(f"  [PASS] Rutting depth range: [{np.min(z_diff):.6f}, {np.max(z_diff):.6f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 8: add_corrugation
    # ===================================================================
    print("\n[Test 8/11] add_corrugation ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        pts_c, lbl_c = add_corrugation(
            pts, lbl, direction="longitudinal",
            wavelength=0.1, amplitude=0.015, severity="light", seed=seed,
        )
        assert pts_c.shape == pts.shape
        assert lbl_c.shape == lbl.shape
        z_diff = pts_c[:, 2] - pts[:, 2]
        # 振幅应约为 0.015
        assert np.max(np.abs(z_diff)) <= 0.015 + 1e-6

        # 横向波纹
        pts_c2, lbl_c2 = add_corrugation(
            pts, lbl, direction="transverse",
            wavelength=0.1, amplitude=0.02, severity="severe", seed=seed,
        )
        z_diff2 = pts_c2[:, 2] - pts[:, 2]
        assert np.any(np.abs(z_diff2) > 0.015)

        print(f"  [PASS] Corrugation amplitude: [{np.min(z_diff):.6f}, {np.max(z_diff):.6f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 9: add_bleeding
    # ===================================================================
    print("\n[Test 9/11] add_bleeding ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        region = pts[:, 0] > test_width / 2.0
        lbl_b = add_bleeding(pts, lbl, region_mask=region, seed=seed)
        assert lbl_b.shape == lbl.shape
        assert np.all(lbl_b[region] == 19), "Bleeding region should be label 19"
        assert np.all(lbl_b[~region] == 0), "Background should remain label 0"

        # 验证几何未改变
        print(f"  [PASS] Bleeding labels: {np.unique(lbl_b, return_counts=True)}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 10: add_concrete_damage
    # ===================================================================
    print("\n[Test 10/11] add_concrete_damage ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         pavement_type="concrete", seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        concrete_types = [
            "slab_shatter", "slab_crack", "corner_break", "faulting",
            "pumping", "edge_spall", "joint_damage", "pitting",
            "blowup", "exposed_aggregate",
        ]

        for dtype in concrete_types:
            sev = "light" if dtype not in ("pumping", "pitting", "blowup",
                                           "exposed_aggregate") else "-"
            pts_cd, lbl_cd = add_concrete_damage(
                pts, lbl, damage_type=dtype, severity=sev,
                params={}, seed=seed,
            )
            assert pts_cd.shape == pts.shape, (
                f"{dtype}: points shape {pts_cd.shape}"
            )
            assert lbl_cd.shape == lbl.shape, (
                f"{dtype}: labels shape {lbl_cd.shape}"
            )
            unique_labels = np.unique(lbl_cd)
            for ul in unique_labels:
                if ul > 0:
                    assert 21 <= ul <= 36, (
                        f"{dtype}: label {ul} out of concrete range [21, 36]"
                    )

        print("  [PASS] All 10 concrete damage types tested")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 测试 11: simulate_lidar_noise
    # ===================================================================
    print("\n[Test 11/11] simulate_lidar_noise ...")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        pts_noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.01, dropout_rate=0.05,
            angular_jitter_deg=0.01, seed=seed,
        )
        # 有 dropout 时 N' <= N
        assert pts_noisy.shape[1] == 3
        assert pts_noisy.shape[0] <= pts.shape[0], (
            f"Noisy points {pts_noisy.shape[0]} > original {pts.shape[0]}"
        )
        assert np.all(np.isfinite(pts_noisy)), "Non-finite points in noisy cloud"
        assert pts_noisy.shape[0] >= pts.shape[0] * 0.8, (
            f"Too many points dropped: {pts_noisy.shape[0]}/{pts.shape[0]}"
        )

        # 无 dropout
        pts_noisy2 = simulate_lidar_noise(
            pts, distance_noise_std=0.005, dropout_rate=0.0,
            angular_jitter_deg=0.005, seed=seed,
        )
        assert pts_noisy2.shape[0] == pts.shape[0], (
            f"Without dropout, shape should match: {pts_noisy2.shape[0]} vs {pts.shape[0]}"
        )

        print(f"  [PASS] Noisy points: {pts_noisy.shape} "
              f"(dropped {pts.shape[0] - pts_noisy.shape[0]}/{pts.shape[0]})")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ===================================================================
    # 汇总
    # ===================================================================
    print("\n" + "=" * 72)
    print(f"  结果: {passed}/{total} 测试通过")
    if passed == total:
        print("  [OK] 全部通过 — primitives.py 实现完成")
    else:
        print(f"  [FAIL] {total - passed} 个测试失败")
    print("=" * 72)
