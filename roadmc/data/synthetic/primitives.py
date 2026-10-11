"""
RoadMC 数学与力学基元 —— Physics-Simulation-Driven Road Surface Primitives.

严格遵循 JTG 5210-2018《公路技术状况评定标准》，共 38 个标签 (0-37)。

本文件实现了 12 个数学与力学基元函数，通过物理学仿真生成路面点云数据：
  1. generate_road_surface()   — 分段各向同性径向 PSD (ISO 8608 段 + 自仿射
                                  纹理段) + 横坡/纵坡设计几何 (FFT 谱合成)
  2. generate_curb_strip()     — 行车道外侧缘石顶面条带 (确定性几何，非路面点)
  3. add_micro_texture()       — 自仿射纹理谱合成 (散点兼容路径)
  4. add_crack()               — 裂缝 (纵向/横向 Bézier + 龟裂/块状 Voronoi)
  5. add_pothole()             — 超椭圆坑槽
  6. add_raveling()            — 松散 (细集料脱落)
  7. add_depression()          — 沉陷 (高斯凹陷)
  8. add_rutting()             — 车辙 (双轮迹高斯槽)
  9. add_corrugation()         — 波浪拥包 (正弦调制)
  10. add_bleeding()            — 泛油 (反射率/标签修改)
  11. add_concrete_damage()     — 水泥路面 10 种损坏
  12. simulate_lidar_noise()    — LiDAR 噪声仿真 (球坐标 + Dropout)

所有函数使用 numpy + scipy，不使用 torch。
随机状态统一使用 ``np.random.default_rng(seed)``。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import interpolate, ndimage, spatial, stats

try:
    from .config import ISO8608_BAND_MAX_CYCLES_PER_M, ISO_ROUGHNESS
except ImportError:
    from config import ISO8608_BAND_MAX_CYCLES_PER_M, ISO_ROUGHNESS

# JTG 5210-2018 线状裂缝轻/重判据：缝宽 3 mm 分界。
JTG_CRACK_WIDTH_THRESHOLD_M: float = 0.003


def _coord_hash01(x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic per-point uniform in [0, 1) keyed on the coordinates.

    分段体制的**云无关**随机决策：同一物理点无论落在哪个段的扩展云里，
    其 (x, y) 浮点位型相同 ⇒ 哈希值相同 ⇒ 摘除/侵蚀/剥落决策逐位一致。
    用于 raveling 的点摘除与坑槽边缘剥落——这两个基元原本用
    ``rng.choice(cloud_indices)``，其消耗随传入云的点集变化，跨段会在缝
    两侧给出不同的孔洞/剥蚀模式（见设计稿云无关规则）。哈希只依赖坐标
    与盐值，不消耗任何 ``Generator`` 状态。

    Args:
        x, y: 点坐标 (N,)，float64。
        salt: 32/64 位整数盐值（按位与 2⁶⁴-1 截断）。

    Returns:
        (N,) float64，均匀分布于 [0, 1)（splitmix64 尾混）。
    """

    xb = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).view(np.int64)
    yb = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).view(np.int64)
    with np.errstate(over="ignore"):
        h = (
            xb.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            + yb.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            + np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
        )
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


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

    for _o in range(octaves):
        grid_size = 8
        noise_grid = rng.uniform(-1.0, 1.0, (grid_size, grid_size))

        interp = interpolate.RegularGridInterpolator(
            (np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size)),
            noise_grid,
            bounds_error=False,
            fill_value=0.0,
        )

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

    closest = A + t[:, None] * AB
    return np.sqrt(np.sum((points - closest) ** 2, axis=1))


def _point_to_segment_distance_t(
    points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


# 1.1.1 — 路面宏观轮廓生成 (ISO 8608 PSD)


def generate_road_surface(
    width: float,
    length: float,
    grid_res: float,
    pavement_type: str = "asphalt",
    roughness_class: str = "A",
    seed: int | None = None,
    *,
    texture_rms: float = 0.0,
    texture_hurst: float = 0.7,
    crossfall: float = 0.0,
    crossfall_shape: str = "crowned",
    longitudinal_grade: float = 0.0,
    texture_het_sigma_ln: float = 0.0,
    texture_het_corr_len_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a road surface via a piecewise isotropic radial PSD.

    高度场由三部分构成：确定性设计几何 + ISO 8608 宏观粗糙度 + 自仿射
    宏观纹理，随机部分共用一次 FFT 合成。

    **ISO 8608 段** — ISO 定义的是一维纵断面谱
    :math:`G_d(n) = G_d(n_0)(n/n_0)^{-w}` (:math:`n_0=0.1`, :math:`w=2`)。
    各向同性二维表面必须满足边缘化一致性
    :math:`G_{1D}(n_x) = \\int \\Phi(\\sqrt{n_x^2+n_y^2})\\,dn_y`
    (Dodds & Robson 1973; Kamash & Robson 1978; Bogsjö 2008)，对 w=2 解得

    .. math::
        \\Phi_{\\mathrm{ISO}}(\\Omega) = \\frac{G_d(n_0)\\,n_0^2}{4}\\,
        \\Omega^{-3}, \\qquad \\Omega \\le 2.83\\ \\mathrm{c/m}.

    **纹理段** — 自仿射 (fBm 型) 表面谱
    :math:`\\Phi_{\\mathrm{tex}}(\\Omega) \\propto \\Omega^{-(2H+2)}`
    (Persson 2006)，占据 :math:`(2.83, f_{\\mathrm{Nyq}}]`，
    总能量按 Parseval 校准到 ``texture_rms``。

    方差目标为离散 Parseval 和
    :math:`\\sigma^2 = \\sum \\Phi\\,\\Delta f_x \\Delta f_y` (单位 m²)。
    每次实现被精确缩放到目标 RMS —— 这是受控方差的设计选择，便于
    等级间可比性（样本方差的自然涨落被抑制，见项目文档声明）。

    **设计几何** — 排水横坡（双向路拱或单向坡）与纵坡：
    :math:`z_c(x) = -i_c\\,|x - W/2|` (crowned) 或 `-i_c\\,x` (plane)，
    :math:`z_g(y) = i_g\\,y`。法向量从叠加后的最终高度场统一重算，
    因此包含坡度与纹理的贡献。

    **纹理沿程不均匀** — ``texture_het_sigma_ln > 0`` 时（默认 0 = 关闭，
    行为与历史逐位一致），先以 ISO 带边界 :math:`f_{\\mathrm{iso,max}}`
    做高低通分解，仅对短波（纹理段）施加沿里程 y 的均值为 1 对数正态
    幅值调制 :math:`M(y)`；调制场由一维高斯低通白噪声标准化得到，
    相关长度 ``texture_het_corr_len_m``。ISO 长波段不被调制，且
    :math:`M` 按 :math:`\\mathrm{mean}(M^2)=1` 归一，重组后经既有全局
    Parseval 再定标，总方差仍精确等于目标 :math:`\\sigma^2`。

    Args:
        width:  路面宽度 (m)，x 方向。
        length: 路面长度 (m)，y 方向。
        grid_res: 网格分辨率 (m)。
        pavement_type: 路面类型 ('asphalt' | 'concrete')，仅影响标签。
        roughness_class: ISO 8608 粗糙度等级 'A'–'E'。
        seed: 随机种子。
        texture_rms: 宏观纹理位移 RMS (m)。0 表示无纹理段。
        texture_hurst: 纹理段 Hurst 指数 H，谱斜率 -(2H+2)。
        crossfall: 排水横坡坡率 (如 0.02 = 2%)。0 表示水平。
        crossfall_shape: 'crowned' 双向路拱 | 'plane' 单向坡。
        longitudinal_grade: 纵坡坡率。
        texture_het_sigma_ln: 纹理沿程不均匀的对数域调制强度 σ_ln。
            0.0 = 关闭（默认），此时不做分解、不抽取额外随机数，输出
            与合并谱单次 FFT 的历史路径逐位一致。
        texture_het_corr_len_m: 沿里程 y 的高斯低通相关长度 (米)，
            仅在 ``texture_het_sigma_ln > 0`` 时生效。


    Returns:
        points:  点云 (N, 3) = [x, y, z]。
        normals: 单位法向量 (N, 3)，含坡度与纹理贡献。
    """
    rng = np.random.default_rng(seed)

    x = np.arange(0.0, width, grid_res)
    y = np.arange(0.0, length, grid_res)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)

    n0 = 0.1  # 参考空间频率 (cycle/m)
    Gd0 = ISO_ROUGHNESS[roughness_class] * 1e-6  # m³/cycle

    fx = np.fft.fftfreq(nx, d=grid_res)  # (nx,)
    fy = np.fft.fftfreq(ny, d=grid_res)  # (ny,)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    omega = np.sqrt(FX**2 + FY**2)  # 径向波数 (cycle/m)
    nyquist = 0.5 / grid_res
    dfx = 1.0 / (nx * grid_res)
    dfy = 1.0 / (ny * grid_res)

    f_iso_max = min(ISO8608_BAND_MAX_CYCLES_PER_M, nyquist)

    # ISO 8608 段：各向同性径向谱 Φ(Ω) = (Gd0·n0²/4)·Ω⁻³，带限到 2.83 c/m。
    with np.errstate(divide="ignore", invalid="ignore"):
        psd_iso = np.where(
            (omega > 0.0) & (omega <= f_iso_max),
            0.25 * Gd0 * n0**2 * np.where(omega > 0.0, omega, 1.0) ** (-3.0),
            0.0,
        )
    var_iso = float(np.sum(psd_iso) * dfx * dfy)

    # 纹理段：Φ ∝ Ω^-(2H+2)，占据 (2.83, Nyquist]，能量校准到 texture_rms²。
    var_tex = 0.0
    psd_tex = np.zeros_like(psd_iso)
    if texture_rms > 0.0 and nyquist > f_iso_max:
        beta = 2.0 * texture_hurst + 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(
                omega > f_iso_max,
                np.where(omega > 0.0, omega, 1.0) ** (-beta),
                0.0,
            )
        raw_var = float(np.sum(raw) * dfx * dfy)
        if raw_var > 0.0:
            psd_tex = raw * (texture_rms**2 / raw_var)
            var_tex = texture_rms**2

    psd_2d = psd_iso + psd_tex

    white_noise = rng.normal(0.0, 1.0, (nx, ny))
    W_hat = np.fft.fft2(white_noise)
    if texture_het_sigma_ln > 0.0:
        # 纹理沿程不均匀：高低通分解后仅调制短波分量。两分量共用同一
        # W_hat，分解本身确定；调制场 M(y) 为中位数 1 且 mean(M²)=1
        # （二阶矩归一，非算术均值 1：归一后 mean(M)≈exp(-σ²/2)<1），
        # 只沿里程轴 (indexing='ij' 的轴 1) 变化并广播到横向。h_tex 零均值
        # 使乘积均值≈0，方差由 E[M²]=1 决定，故总方差不变。
        h_iso = np.real(np.fft.ifft2(np.sqrt(psd_iso) * W_hat))
        h_tex = np.real(np.fft.ifft2(np.sqrt(psd_tex) * W_hat))
        e = rng.normal(0.0, 1.0, ny)
        g = ndimage.gaussian_filter1d(
            e, sigma=texture_het_corr_len_m / grid_res, mode="reflect"
        )
        z = (g - g.mean()) / max(float(g.std()), 1e-15)
        M = np.exp(texture_het_sigma_ln * z)
        M = M / np.sqrt(np.mean(M**2))
        h_field = h_iso + h_tex * M[None, :]
    else:
        # 关闭态：合并谱单次 FFT，与历史路径逐位一致（不得改写为分解式，
        # sqrt(a)+sqrt(b) ≠ sqrt(a+b) 会破坏逐位一致性）。
        h_field = np.real(np.fft.ifft2(np.sqrt(psd_2d) * W_hat))
    h_field = h_field - np.mean(h_field)

    # Parseval 定标：σ² = ΣΦ·Δfx·Δfy (m²)，实现级精确缩放（受控方差）。
    rms_target = np.sqrt(max(var_iso + var_tex, 0.0))
    rms_current = np.std(h_field)
    if rms_current > 1e-15 and rms_target > 0.0:
        h_field = h_field * (rms_target / rms_current)

    # 确定性设计几何：横坡/路拱 + 纵坡。法向量必须包含坡度贡献，
    # 因此在计算法向之前叠加。
    if crossfall != 0.0:
        if crossfall_shape == "crowned":
            h_field = h_field - crossfall * np.abs(X - width / 2.0)
        else:
            h_field = h_field - crossfall * X
    if longitudinal_grade != 0.0:
        h_field = h_field + longitudinal_grade * Y

    normals_grid = _compute_normals(h_field, grid_res, grid_res)

    points = np.stack([X.ravel(), Y.ravel(), h_field.ravel()], axis=1)  # (N, 3)
    normals = normals_grid.reshape(-1, 3)  # (N, 3)

    return points, normals


# 1.1.2 — 分段条带合成（strip segmentation runtime）


def _iso_band_variance_fine(
    width: float, length: float, grid_res: float, Gd0: float
) -> float:
    """ISO 8608 带（Ω ≤ f_iso_max）在**细网格（单块路径）**离散频率集上的
    精确 Parseval 方差，供分段宏观场定标使用。

    单块 :func:`generate_road_surface` 的 ``var_iso`` 为
    ``Σ Φ·dfx·dfy``，其中 ``fx = fftfreq(nx, grid_res) = k/width``（
    ``nx = ceil(width/grid_res)``）、``fy = l/length``。这里用同一组
    频率（``k/width``、``l/length``）在带内求和：因 Φ 带外为 0，只需
    ``|k| ≤ f_iso_max·width``、``|l| ≤ f_iso_max·length`` 的项，网格规模
    仅 ~O((f·W)·(f·L))（3 m/50 m@2 mm 为 9×142），代价可忽略。粗网格
    ``dfx = 1/(nx_c·h_macro) ≠ 1/width``，直接用粗网格和定标会引入
    ~1.3% 的宏观 RMS 偏差（见 :func:`synthesize_strip_macro_band`）。

    Args:
        width, length: 条带几何 (m)。
        grid_res: 细网格分辨率 (m)，用于 f_iso_max。
        Gd0: ISO 8608 功率谱密度常数 (m³/cycle)，= 等级×1e-6。

    Returns:
        细网格 ISO 带方差 (m²)。
    """

    n0 = 0.1
    f_iso_max = min(ISO8608_BAND_MAX_CYCLES_PER_M, 0.5 / grid_res)
    dfx = 1.0 / width
    dfy = 1.0 / length
    kx = np.arange(
        -int(np.ceil(f_iso_max * width)), int(np.ceil(f_iso_max * width)) + 1
    )
    ky = np.arange(
        -int(np.ceil(f_iso_max * length)), int(np.ceil(f_iso_max * length)) + 1
    )
    FX = (kx * dfx)[:, None]
    FY = (ky * dfy)[None, :]
    omega = np.sqrt(FX**2 + FY**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = np.where(
            (omega > 0.0) & (omega <= f_iso_max),
            0.25 * Gd0 * n0**2 * np.where(omega > 0.0, omega, 1.0) ** (-3.0),
            0.0,
        )
    return float(np.sum(psd) * dfx * dfy)


def synthesize_strip_macro_band(
    width: float,
    length: float,
    grid_res: float,
    roughness_class: str,
    rng_surface: np.random.Generator,
    *,
    macro_grid_factor: float = 4.0,
) -> dict:
    """Synthesize the ISO 8608 macro band once for the whole strip on a
    coarse grid.

    分段体制的宏观长波段**整条带全局一次**合成：粗网格间距
    ``h_macro = 1 / (macro_grid_factor * f_iso_max)``（对 ISO 带上限取
    ``macro_grid_factor`` 倍过采样），PSD 与 :func:`generate_road_surface`
    的 ISO 段逐字相同（Φ = Gd0·n0²/4·Ω⁻³，Ω ≤ f_iso_max）。**Parseval
    定标取细网格（单块路径）的 ISO 带离散和**——粗网格的
    ``dfx = 1/(nx_c·h_macro)`` 与细网格的 ``dfx = 1/width`` 不同（nx_c·h
    ≈ width + h_macro），直接用粗网格和定标会偏 -2.58%（2 mm/3×50 m）或
    +2.36%（5 mm/7×5 m），即宏观 RMS 偏差 ~1.3%；改用细网格频率集上的
    小规模精确求和（ka·dfx、kb·dfy，仅 |k| ≤ f_iso_max·W/L 项，成本可
    忽略）后与单块宏观方差一致。宏观场全局单块 ⇒ 接缝处精确连续，与
    粗糙度等级无关。

    Args:
        width: 条带横向宽度 (m)。
        length: 条带里程向长度 (m)。
        grid_res: 细网格分辨率 (m)，仅用于 f_iso_max。
        roughness_class: ISO 8608 等级。
        rng_surface: 表面随机流（与单块路径同源：由 surface seed 派生），
            宏观白噪声从此抽取。
        macro_grid_factor: 粗网格过采样倍数。

    Returns:
        dict(x=..., y=..., field=(nx_c, ny_c) float64, spacing_m=h_macro)
    """

    n0 = 0.1
    Gd0 = ISO_ROUGHNESS[roughness_class] * 1e-6
    f_iso_max = min(ISO8608_BAND_MAX_CYCLES_PER_M, 0.5 / grid_res)
    h_macro = 1.0 / (macro_grid_factor * f_iso_max)

    x_c = np.arange(0.0, width + h_macro, h_macro)
    y_c = np.arange(0.0, length + h_macro, h_macro)
    nx_c, ny_c = len(x_c), len(y_c)

    fx = np.fft.fftfreq(nx_c, d=h_macro)
    fy = np.fft.fftfreq(ny_c, d=h_macro)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    omega = np.sqrt(FX**2 + FY**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        psd_iso = np.where(
            (omega > 0.0) & (omega <= f_iso_max),
            0.25 * Gd0 * n0**2 * np.where(omega > 0.0, omega, 1.0) ** (-3.0),
            0.0,
        )

    # 细网格（单块口径）ISO 带方差：细网格 fx = k/(nx·grid_res) = k/width，
    # fy = l/length。仅需 |k| ≤ f_iso_max·width、|l| ≤ f_iso_max·length 的
    # 频率（带外 PSD 为 0），网格极小（3m/50m@2mm → 9×142）。
    var_iso = _iso_band_variance_fine(width, length, grid_res, Gd0)

    white = rng_surface.normal(0.0, 1.0, (nx_c, ny_c))
    macro = np.real(np.fft.ifft2(np.sqrt(psd_iso) * np.fft.fft2(white)))
    macro = macro - np.mean(macro)
    rms_current = float(np.std(macro))
    rms_target = math.sqrt(max(var_iso, 0.0))
    if rms_current > 1e-15 and rms_target > 0.0:
        macro = macro * (rms_target / rms_current)

    return {"x": x_c, "y": y_c, "field": macro, "spacing_m": h_macro}


def generate_strip_surface(
    width: float,
    length: float,
    grid_res: float,
    roughness_class: str,
    seed: int | None,
    *,
    segment_bounds: list[tuple[int, int, int, int]],
    blend_band_rows: int,
    texture_rms: float = 0.0,
    texture_hurst: float = 0.7,
    crossfall: float = 0.0,
    crossfall_shape: str = "crowned",
    longitudinal_grade: float = 0.0,
    texture_het_sigma_ln: float = 0.0,
    texture_het_corr_len_m: float = 2.0,
    macro_grid_factor: float = 4.0,
    macro_band: dict | None = None,
):
    """Segmented strip surface synthesis (yields one extended segment at a
    time to keep the working set bounded).

    谱分解（与 :func:`generate_road_surface` 单块路径的谱定义一致）：

    1. **宏观长波**（ISO 8608 段）：整条带全局一次合成于粗网格（见
       :func:`synthesize_strip_macro_band`），粗→细用
       ``RegularGridInterpolator(method="cubic")`` 逐段采样。接缝处
       精确连续。
    2. **短波纹理**（(f_iso_max, Nyquist]）：每段在扩展域（拥有行 +
       两侧 halo 行）上独立 FFT，Φ ∝ Ω^{-(2H+2)}，Parseval 定标到
       ``texture_rms``；段种子从专用子 rng 按段序抽取，与分段数确定
       对应。
    3. **跨段混合**：缝两侧各 2·blend_band_rows 行的重叠带上按
       root-raised-cosine 混合（cos²+sin²=1，独立等方差高斯场混合后
       方差保持）。同一混合带只计算一次，两个段看到**同一份**混合值
       ⇒ 逐位一致。
    4. **texture-het 调制**：M(y) 全局 1-D（白噪声 → gaussian_filter1d
       → 标准化 → exp(σ·z) → mean(M²)=1），混合后统一乘 M 的全局切片
       （单一施加位点）；het 关闭时零新增随机流。
    5. **设计几何**（crossfall/crowned 折角与纵坡）不进粗网格——折角
       不是带限信号——在混合后的细网格上解析叠加（与单块路径同次序），
       法向从最终高度场统一重算。

    每个输出行的最终值只由恰好一个段的"拥有趟"产生；halo 行只喂邻域
    算子，绝不作为拥有值写出（由调用方按拥有行切片落盘）。

    Args:
        segment_bounds: 每段 ``(y0_owned, y1_owned, y0_ext, y1_ext)`` 行
            区间（行 index，y = row * grid_res；ext 区间已按条带边界裁剪）。
        blend_band_rows: 缝两侧混合带半宽（行数，= halo 行数 m_rows）。
        macro_band: 预先合成好的宏观粗网格（:func:`synthesize_strip_macro_band`
            的返回值）；None 时内部按相同流程合成（独立调用/测试用）。

    Yields:
        ``(seg_index, y0_ext, y1_ext, y0_owned, y1_owned, x_grid,
        y_ext_values, z_ext (nx, ext_rows), normals_ext (nx, ext_rows, 3))``
    """

    rng_surface = np.random.default_rng(seed)
    rng_tex = np.random.default_rng(int(rng_surface.integers(0, 2**31)))

    x = np.arange(0.0, width, grid_res)
    y = np.arange(0.0, length, grid_res)
    nx, ny_total = len(x), len(y)
    nyquist = 0.5 / grid_res
    f_iso_max = min(ISO8608_BAND_MAX_CYCLES_PER_M, nyquist)

    if macro_band is None:
        macro_band = synthesize_strip_macro_band(
            width, length, grid_res, roughness_class, rng_surface,
            macro_grid_factor=macro_grid_factor,
        )
    macro_interp = interpolate.RegularGridInterpolator(
        (macro_band["x"], macro_band["y"]),
        macro_band["field"],
        method="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    # texture-het 全局 1-D 调制场（整条 y 轴，天然跨段连续）。
    if texture_het_sigma_ln > 0.0:
        e = rng_tex.normal(0.0, 1.0, ny_total)
        g = ndimage.gaussian_filter1d(
            e, sigma=texture_het_corr_len_m / grid_res, mode="reflect"
        )
        zmod = (g - g.mean()) / max(float(g.std()), 1e-15)
        het_m = np.exp(texture_het_sigma_ln * zmod)
        het_m = het_m / np.sqrt(np.mean(het_m**2))
    else:
        het_m = None

    beta = 2.0 * texture_hurst + 2.0

    def _synth_texture(y0e: int, y1e: int) -> np.ndarray:
        ext_rows = y1e - y0e
        seed_k = int(rng_tex.integers(0, 2**31))
        rng_k = np.random.default_rng(seed_k)
        fx = np.fft.fftfreq(nx, d=grid_res)
        fy = np.fft.fftfreq(ext_rows, d=grid_res)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        omega = np.sqrt(FX**2 + FY**2)
        dfx = 1.0 / (nx * grid_res)
        dfy = 1.0 / (ext_rows * grid_res)
        if texture_rms > 0.0 and nyquist > f_iso_max:
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(
                    omega > f_iso_max,
                    np.where(omega > 0.0, omega, 1.0) ** (-beta),
                    0.0,
                )
            raw_var = float(np.sum(raw) * dfx * dfy)
            if raw_var <= 0.0:
                h = np.zeros((nx, ext_rows))
            else:
                h = np.real(np.fft.ifft2(
                    np.sqrt(raw * (texture_rms**2 / raw_var))
                    * np.fft.fft2(rng_k.normal(0.0, 1.0, (nx, ext_rows)))
                ))
                std = float(np.std(h))
                if std > 1e-15:
                    h = h * (texture_rms / std)
        else:
            h = np.zeros((nx, ext_rows))
        return h

    prev_field: np.ndarray | None = None
    prev_bounds: tuple[int, int, int, int] | None = None

    def _finalize(seg_k: int, bounds: tuple[int, int, int, int], tex: np.ndarray):
        y0o, y1o, y0e, y1e = bounds
        h = tex
        if het_m is not None:
            h = h * het_m[y0e:y1e][None, :]
        Y_e = y[y0e:y1e]
        macro_part = macro_interp(
            np.stack(np.meshgrid(x, Y_e, indexing="ij"), axis=-1).reshape(-1, 2)
        ).reshape(nx, y1e - y0e)
        h_field = macro_part + h
        if crossfall != 0.0:
            X_e = np.meshgrid(x, Y_e, indexing="ij")[0]
            if crossfall_shape == "crowned":
                h_field = h_field - crossfall * np.abs(X_e - width / 2.0)
            else:
                h_field = h_field - crossfall * X_e
        if longitudinal_grade != 0.0:
            h_field = h_field + longitudinal_grade * Y_e[None, :]
        normals_grid = _compute_normals(h_field, grid_res, grid_res)
        return (seg_k, y0e, y1e, y0o, y1o, x, Y_e, h_field, normals_grid)

    for k, bounds in enumerate(segment_bounds):
        cur = _synth_texture(bounds[2], bounds[3])
        if prev_field is not None:
            # 缝在段界 s = prev.y1_owned = cur.y0_owned；混合带
            # [s − m, s + m)（裁剪到条带内）。两段扩展域都覆盖该带
            # （influence/halo 设计保证），同一带只计算一次并写回两个
            # 域 ⇒ 跨段逐位一致。
            s = segment_bounds[k][0]
            m = blend_band_rows
            lo = max(0, s - m)
            hi = min(ny_total, s + m)
            p0o, p1o, p0e, p1e = prev_bounds
            theta = (
                (y[lo:hi] - y[max(0, s - m)])
                / (2.0 * m * grid_res)
                * (np.pi / 2.0)
            )
            cos_t = np.cos(theta)[None, :]
            sin_t = np.sin(theta)[None, :]
            prev_slice = prev_field[:, lo - p0e : hi - p0e]
            cur_slice = cur[:, lo - bounds[2] : hi - bounds[2]]
            blended = cos_t * prev_slice + sin_t * cur_slice
            prev_field[:, lo - p0e : hi - p0e] = blended
            cur[:, lo - bounds[2] : hi - bounds[2]] = blended
            yield _finalize(k - 1, prev_bounds, prev_field)
        prev_field = cur
        prev_bounds = bounds
    yield _finalize(len(segment_bounds) - 1, prev_bounds, prev_field)


# 1.1.3 — 缘石条带 (非路面点)


def generate_curb_strip(
    road_width: float,
    length: float,
    grid_res: float,
    curb_width_m: float,
    curb_height_m: float,
    *,
    crossfall: float = 0.0,
    crossfall_shape: str = "crowned",
    longitudinal_grade: float = 0.0,
    y_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成行车道横向外侧的两条缘石顶面条带（几何确定性，零 rng）。

    该函数存在的目的不是"好看的路缘"，而是让非路面点进入下游的密度与
    法向倾角统计（域差距诊断需要非路面点），因此它只采样**顶面**：
    没有竖直 riser 立面点，台阶棱线处的法向异常由下游 KDTree 统计自然
    呈现，不人为构造立面采样——这是诚实口径，避免把一条假想的立面当作
    实测几何。

    几何：

    - 条带位于行车道外侧两段：``x ∈ [-curb_width_m, 0)`` 与
      ``x ∈ (road_width, road_width + curb_width_m]``，y 与路面同网格
      （``np.arange(0, length, grid_res)``）。x 向格点数
      ``nx_c = max(1, ceil(curb_width_m / grid_res))``，与
      ``RoadSurfaceConfig.surface_grid_shape`` 同一取整规则，因此总点数
      恰为 ``2 * nx_c * ny``（= ``GeneratorConfig.curb_grid_point_count``）。
    - 顶面高程与相邻路面边缘的**设计几何**对齐并抬高 ``curb_height_m``：
      crowned 时两侧边缘高程为 ``-crossfall * road_width / 2``；plane 时
      x<0 侧为 0、x>road_width 侧为 ``-crossfall * road_width``；再叠加
      ``longitudinal_grade * y``。顶面横向水平（不继承路拱斜率）。
    - 顶面法向取 ``(0, 0, 1)``，与 ``generate_road_surface`` 对平坦区的
      法向一致。

    调用方在拼接后初始化标签，因此本函数返回的点自动获得背景标签 0，
    不占用 38 类中任何病害标签；同时因点数增加会**稀释自然 prevalence
    分母**——这是缘石选项的预期效应，不是缺陷。

    Args:
        road_width: 行车道宽度 (m)，条带据此贴边。
        length: 沿里程长度 (m)。
        grid_res: 网格分辨率 (m)，与路面网格一致。
        curb_width_m: 单侧条带横向宽度 (m)，须 > 0（调用方保证）。
        curb_height_m: 顶面相对相邻路面设计边缘的抬高量 (m)。
        crossfall: 排水横坡坡率，用于与路面边缘高程对齐。
        crossfall_shape: 'crowned' 双向路拱 | 'plane' 单向坡。
        longitudinal_grade: 纵坡坡率，叠加到顶面高程。

    Returns:
        points:  (N_c, 3) = [x, y, z]，N_c = 2 * nx_c * ny。
        normals: (N_c, 3)，顶面法向恒为 (0, 0, 1)。
    """

    y = (
        np.asarray(y_values, dtype=np.float64)
        if y_values is not None
        else np.arange(0.0, length, grid_res)
    )
    nx_c = max(1, int(math.ceil(curb_width_m / grid_res)))

    # x<0 侧取 [-w, 0)（含 -w，不含 0）；x>road_width 侧取 (W, W+w]（含 W+w，不含 W）。
    x_neg = -curb_width_m + np.arange(nx_c) * grid_res
    x_pos = road_width + (np.arange(nx_c) + 1) * grid_res
    x_c = np.concatenate([x_neg, x_pos])

    if crossfall_shape == "crowned":
        edge_neg = -crossfall * road_width / 2.0
        edge_pos = -crossfall * road_width / 2.0
    else:  # plane：单向坡，x<0 侧基准 0，x>W 侧 -i_c·W
        edge_neg = 0.0
        edge_pos = -crossfall * road_width
    edge = np.concatenate([np.full(nx_c, edge_neg), np.full(nx_c, edge_pos)])

    Xc, Yc = np.meshgrid(x_c, y, indexing="ij")  # (2*nx_c, ny)
    Zc = edge[:, None] + curb_height_m + longitudinal_grade * Yc

    points = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1)  # (N_c, 3)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (points.shape[0], 1))

    return points, normals


# 1.1.2 — 微观纹理叠加 (fBm)


def resample_to_lidar_pattern(
    points: np.ndarray,
    scan_lines: int = 64,
    vertical_fov_deg: float = 40.0,
    scan_pattern: str = "rotating",
    range_decay: float = 0.3,
    incidence_angle_drop: float = 0.05,
    rng: np.random.Generator | None = None,
    sensor_pose: np.ndarray | None = None,
    line_sigma_m: float = 0.02,
) -> np.ndarray:
    """P1-1: Resample uniform grid points to simulate LiDAR scan line density.

    将规则网格点云重采样为模拟 LiDAR 扫描线模式的非均匀分布
    （对规则网格做概率稀疏化的快速近似，不是射线投射；保留点仍在
    原网格位置上——诚实口径见 TECHNICAL_REPORT 的限制声明）。

    旋转式 LiDAR（车载位姿 ``sensor_pose``，与噪声/强度共用）：
    - 扫描线地面位置 :math:`y_i = y_s + h\\tan\\alpha_i`，仰角
      :math:`\\alpha_i` 由 ``vertical_fov_deg`` 均分，地面线距
      :math:`\\Delta y_i \\approx \\Delta\\alpha \\cdot R_i^2 / h`
      随距离二次增长；
    - 距离衰减与真实入射角 (:math:`\\cos\\theta \\approx h/R`) 丢点。

    Args:
        points: 规则网格点云 (N, 3)。
        scan_lines: 扫描线数量。
        vertical_fov_deg: 垂直视场角 (度)，决定仰角序列。
        scan_pattern: 'rotating' 或 'solid_state'。
        range_decay: 距离衰减系数 α。
        incidence_angle_drop: 入射角丢点系数 β。
        rng: 随机数生成器。
        sensor_pose: 传感器位姿 (x, y, z)。None 时取场景前方车载
            默认 (x 中线, y_min-1, h=2 m)。
        line_sigma_m: 扫描线地面宽度 σ (m)，固定物理量（footprint +
            平台抖动），默认 2 cm。

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

    # 传感器位姿：与噪声/强度/丢点共用同一车载几何。旧实现的
    # "1.5×extent 高空原点" 是航测几何，与噪声阶段的隐含位姿矛盾。
    if sensor_pose is None:
        pose = np.array([(x_min + x_max) / 2.0, y_min - 1.0, 2.0], dtype=np.float64)
    else:
        pose = np.asarray(sensor_pose, dtype=np.float64).reshape(3)
    h = max(float(pose[2]), 1e-3)

    r_dist = np.sqrt(
        (x - pose[0]) ** 2 + (y - pose[1]) ** 2 + h ** 2
    )
    r_max = np.max(r_dist) + 1e-12

    if scan_pattern == "rotating":
        # 扫描线地面位置 y_i = y_s + h·tan(α_i)：仰角在 [α_lo, α_hi]
        # 内均分（受 vertical_fov_deg 限制），地面线距
        # Δy ≈ Δα·R²/h 随距离二次增长——真实旋转式 LiDAR 的
        # 各向异性密度形态。
        alpha_lo = math.atan2(max(y_min - pose[1], 1e-3), h)
        alpha_hi = math.atan2(max(y_max - pose[1], 2e-3), h)
        fov_rad = math.radians(vertical_fov_deg)
        if alpha_hi - alpha_lo > fov_rad:
            alpha_hi = alpha_lo + fov_rad
        alphas = np.linspace(alpha_lo, alpha_hi, scan_lines)
        y_scans = pose[1] + h * np.tan(alphas)

        nearest_scan = np.argmin(np.abs(y[:, None] - y_scans[None, :]), axis=1)
        dist_to_scan = np.abs(y - y_scans[nearest_scan])

        # 到最近扫描线的 Gaussian falloff。线宽是固定的物理量
        # （footprint + 平台抖动，cm 量级），不随线距缩放——否则
        # 远端线间真空会被抹平，密度各向异性消失。
        scan_prob = np.exp(-0.5 * (dist_to_scan / max(line_sigma_m, 1e-4)) ** 2)

        range_prob = 1.0 - range_decay * (r_dist / r_max)

        # 真实入射角：平坦路面 cosθ ≈ h/R。
        cos_theta = np.clip(h / r_dist, 0.0, 1.0)
        incidence_prob = 1.0 - incidence_angle_drop * (1.0 - cos_theta)

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
    octaves: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add spatially correlated self-affine macro-texture to a point set.

    通过 FFT 谱合成生成自仿射纹理场（二维 fBm 型表面，功率谱
    :math:`\\Phi(\\Omega) \\propto \\Omega^{-(2H+2)}`，Mandelbrot & Van Ness
    1968; Saupe 1988），在包围盒网格上合成后按点位双线性采样，
    沿法向位移。位移场是空间相关的：粗糙度结构由谱斜率
    （即 Hurst 指数 H）控制，``amplitude`` 严格等于位移 RMS。

    主生成管线已把纹理段合并进 ``generate_road_surface`` 的分段径向
    谱（那里法向从最终高度场统一重算）；本函数保留给散点输入的
    调用方。**散点路径不更新法向量**——返回的 normals 是输入的副本，
    需要一致法向时应在位移后的点集上重新估计。

    Args:
        points:  点云 (N, 3)。
        normals: 单位法向量 (N, 3)。
        amplitude: 纹理位移 RMS (m)。
        hurst: Hurst 指数 H ∈ (0, 1)，谱斜率 -(2H+2)。
        octaves: 已弃用，仅为向后兼容保留，不再使用。
        seed: 随机种子。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        normals_copy: 输入法向量的副本（未更新，见上）。
    """
    del octaves  # legacy parameter of the removed i.i.d.-noise implementation
    rng = np.random.default_rng(seed)

    pts = points.copy()
    nrm = normals.copy()
    if amplitude <= 0.0 or points.shape[0] < 4:
        return pts, nrm

    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    extent_x = max(x_max - x_min, 1e-6)
    extent_y = max(y_max - y_min, 1e-6)

    # 网格间距取点密度的量级（均匀分布假设），限制网格规模。
    approx_spacing = math.sqrt(extent_x * extent_y / points.shape[0])
    gx = int(np.clip(round(extent_x / approx_spacing), 16, 2048))
    gy = int(np.clip(round(extent_y / approx_spacing), 16, 2048))

    fx = np.fft.fftfreq(gx, d=extent_x / gx)
    fy = np.fft.fftfreq(gy, d=extent_y / gy)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    omega = np.sqrt(FX**2 + FY**2)
    beta = 2.0 * hurst + 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = np.where(omega > 0.0, np.where(omega > 0.0, omega, 1.0) ** (-beta), 0.0)

    field = np.real(np.fft.ifft2(np.sqrt(psd) * np.fft.fft2(rng.normal(0.0, 1.0, (gx, gy)))))
    field = field - field.mean()
    field_std = field.std()
    if field_std > 1e-15:
        field = field * (amplitude / field_std)

    interp = interpolate.RegularGridInterpolator(
        (np.linspace(x_min, x_max, gx), np.linspace(y_min, y_max, gy)),
        field,
        bounds_error=False,
        fill_value=0.0,
    )
    delta = interp(points[:, :2])
    pts += delta[:, None] * nrm

    return pts, nrm


# 1.1.3 — 裂缝生成 (沥青路面)


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
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    # 从映射或直接参数获取标签
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

    # JTG 5210-2018 的线状裂缝轻/重判据是缝宽（3 mm 分界），不是缝深
    # （检测设备通常无法可靠测缝深）。宽度分布按严重度采样并截断到
    # 判据的正确一侧，深度只与宽度弱相关、不作为分级依据。
    if severity == "light":
        width_mean = params.get("width_mean", 0.0018)
        width_std = params.get("width_std", 0.25)
    else:
        width_mean = params.get("width_mean", 0.006)
        width_std = params.get("width_std", 0.3)
    label_width_floor = params.get("label_width_floor", 0.0)

    if crack_type in ("longitudinal", "transverse"):
        x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))

        if "bezier_control_points" in params:
            control_pts = np.array(params["bezier_control_points"], dtype=np.float64)
        else:
            if crack_type == "longitudinal":
                # 纵向裂缝：沿 y 方向。x_center 可由调用方传入（生成器
                # 提供轮迹带边缘先验——真实纵缝多在轮迹带边缘或施工缝）。
                y_vals = np.linspace(y_min + 0.1 * (y_max - y_min),
                                     y_max - 0.1 * (y_max - y_min), 4)
                x_center = float(params.get("x_center", (x_min + x_max) * 0.5))
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

        # 段数由曲线几何决定并设上限——旧的 N//100 使段数随点云规模
        # 线性增长（350k 点 → 3500 段 × 全点云距离 = 58 s/裂缝，5mm
        # 网格下不可用），与几何保真无关。分段体制经
        # params["num_curve_samples"] 固定为 400（clip 的饱和值），
        # 解耦段内点数 N。
        if "num_curve_samples" in params:
            num_curve_samples = int(params["num_curve_samples"])
        else:
            num_curve_samples = int(np.clip(N // 100, 100, 400))
        curve_pts = _cubic_bezier(control_pts, num_curve_samples)

        pert_seed = rng.integers(0, 2**31)
        x_curve = curve_pts[:, 0]
        y_curve = curve_pts[:, 1]
        pert_x = _fractal_perturbation(x_curve, y_curve, octaves=3,
                                       persistence=0.3, scale=0.5, seed=pert_seed)
        pert_y = _fractal_perturbation(x_curve, y_curve, octaves=3,
                                       persistence=0.3, scale=0.5, seed=pert_seed + 1)
        curve_pts[:, 0] += pert_x * 0.02
        curve_pts[:, 1] += pert_y * 0.02

        # 候选点预筛：只有距曲线 ≤ r_cand 的点才可能落进裂缝或其
        # 标签带（缝宽 P99 半宽 ~6mm + label_width_floor），远处点
        # 无需参与逐段距离计算。语义不变，纯性能优化。
        r_cand = max(0.06, label_width_floor)
        point_tree = spatial.cKDTree(xy)
        cand_lists = point_tree.query_ball_point(curve_pts, r_cand)
        # 分段体制：段扩展域可能完全不含曲线（曲线在别的段）——候选
        # 列表为空时直接跳过（历史路径中曲线恒在云内，不会触发）。
        if any(len(c) for c in cand_lists):
            cand_idx = np.unique(np.concatenate([np.asarray(c, dtype=np.int64)
                                                 for c in cand_lists if len(c)]))
        else:
            cand_idx = np.empty(0, dtype=np.int64)
        min_dist = np.full(N, np.inf)
        min_t = np.zeros(N)
        if len(cand_idx) > 0:
            xy_cand = xy[cand_idx]
            cand_dist = np.full(len(cand_idx), np.inf)
            cand_t = np.zeros(len(cand_idx))
            # 用多段线近似，计算到每段距离和局部参数
            for k in range(num_curve_samples - 1):
                t_start = k / (num_curve_samples - 1)
                t_end = (k + 1) / (num_curve_samples - 1)
                dists, t_local = _point_to_segment_distance_t(
                    xy_cand, curve_pts[k], curve_pts[k + 1]
                )
                # 将局部参数映射到全局曲线参数
                t_on_curve_local = t_start + t_local * (t_end - t_start)
                # 仅在当前段更近时更新距离和参数
                better = dists < cand_dist
                cand_dist = np.where(better, dists, cand_dist)
                cand_t = np.where(better, t_on_curve_local, cand_t)
            min_dist[cand_idx] = cand_dist
            min_t[cand_idx] = cand_t

        # 裂缝宽度沿曲线变化 (对数正态采样)，基于空间位置插值。
        # 截断到 JTG 3 mm 判据的正确一侧：轻度 ≤ 3 mm，重度 > 3 mm。
        # 分段体制：params["width_profile"] 由计划阶段预抽（含截断）
        # 传入，替代内部抽取，保证跨段同一条裂缝宽度场逐位一致。
        if "width_profile" in params:
            lognorm_sample = np.asarray(params["width_profile"], dtype=np.float64)
        else:
            lognorm_sample = rng.lognormal(mean=np.log(width_mean), sigma=width_std,
                                           size=num_curve_samples)
            if severity == "light":
                lognorm_sample = np.minimum(lognorm_sample, JTG_CRACK_WIDTH_THRESHOLD_M - 1e-4)
            else:
                lognorm_sample = np.maximum(lognorm_sample, JTG_CRACK_WIDTH_THRESHOLD_M + 2e-4)

        # measure-then-label：标签严重度由实现几何的代表缝宽（中位数）
        # 对照 3 mm 阈值决定，而不是信任输入的 severity 字符串——
        # "标签由几何参数决定" 从声明变成可验证机制。（截断采样使
        # 实现值与请求值一致，此处是防回归的机制化保证。）
        w_rep = float(np.median(lognorm_sample))
        realized_light = w_rep <= JTG_CRACK_WIDTH_THRESHOLD_M
        if crack_type == "longitudinal":
            label_val = 5 if realized_light else 6
        else:
            label_val = 7 if realized_light else 8

        # 使用每个点在曲线上的投影参数 t 进行宽度插值，而非点索引
        half_width = np.interp(
            min_t,
            np.linspace(0, 1, num_curve_samples),
            lognorm_sample,
        ) / 2.0

        # 深度剖面
        # d(depth_ratio) = d_max * exp(-(depth_ratio)^p)
        # λ = half_width/2 = w/4（w 为插值缝宽）：高斯槽在物理带边缘
        # (r=w/2) 处深度 = d_max·e⁻⁴ ≈ 0.018·d_max，比 λ=w/2 的表述窄得多。
        lambda_param = half_width / 2.0  # λ = w/4
        p_param = 2.0  # 高斯槽 (p=2)

        # 裂缝区域掩码
        physical_mask = min_dist <= half_width
        if np.any(physical_mask):
            depth_ratio = min_dist[physical_mask] / np.clip(lambda_param[physical_mask], 1e-12, None)
            depth = d_max * np.exp(-(depth_ratio ** p_param))
            pts[physical_mask, 2] -= depth

        label_half_width = np.maximum(half_width, label_width_floor * 0.5)
        label_mask = min_dist <= label_half_width
        if np.any(label_mask):
            lbl[label_mask] = label_val

    # 龟裂 / 块状裂缝 (Voronoi 图)
    elif crack_type in ("alligator", "block"):
        if crack_type == "alligator":
            block_size = 0.2 if severity == "severe" else 0.5
            width_crack = params.get(
                "width_mean", 0.0015 if severity == "light" else 0.005
            )
        else:
            block_size = 1.0 if severity == "light" else 0.6
            width_crack = params.get(
                "width_mean", 0.002 if severity == "light" else 0.005
            )
        # 分段体制：params["region_bounds"] 由计划阶段按全局条带 bbox
        # 裁剪后传入（patch 最终边界），保证跨段 Voronoi 全链一致；
        # 不传时回退为从传入云 bbox 推导（单块路径历史行为逐位不变）。
        if "region_bounds" in params:
            x_min, x_max, y_min, y_max = (
                float(v) for v in params["region_bounds"]
            )
        else:
            scene_x_min, scene_x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
            scene_y_min, scene_y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))

            # 龟裂是**局部**疲劳损坏（轮迹带内数米的 patch），不是整条路
            # 的裂缝网——旧实现把 Voronoi 种子撒满全场（7×5m 场景 136/140
            # 个 0.5m 瓦片含龟裂标注）。patch 中心可由调用方传入（生成器
            # 给轮迹带先验），尺寸：龟裂 ~1-4m × 0.6-1.2m，块裂更大。
            if crack_type == "alligator":
                region_w = float(params.get("region_width", rng.uniform(0.6, 1.2)))
                region_l = float(params.get("region_length", rng.uniform(1.0, 4.0)))
            else:
                region_w = float(params.get("region_width", rng.uniform(1.5, 3.0)))
                region_l = float(params.get("region_length", rng.uniform(2.0, 4.5)))

            region_center = params.get("region_center")
            if region_center is None:
                region_center = (
                    rng.uniform(scene_x_min, scene_x_max),
                    rng.uniform(scene_y_min, scene_y_max),
                )
            x_min = max(scene_x_min, float(region_center[0]) - region_w / 2.0)
            x_max = min(scene_x_max, float(region_center[0]) + region_w / 2.0)
            y_min = max(scene_y_min, float(region_center[1]) - region_l / 2.0)
            y_max = min(scene_y_max, float(region_center[1]) + region_l / 2.0)
            if x_max - x_min < block_size or y_max - y_min < block_size:
                # patch 被场景边界裁剪得过小：扩到最小可用尺寸。
                x_min = max(scene_x_min, x_max - max(region_w, 2 * block_size))
                x_max = min(scene_x_max, x_min + max(region_w, 2 * block_size))
                y_min = max(scene_y_min, y_max - max(region_l, 2 * block_size))
                y_max = min(scene_y_max, y_min + max(region_l, 2 * block_size))
        area = max((x_max - x_min) * (y_max - y_min), block_size ** 2)

        num_seeds = max(int(area / (block_size ** 2)), 4)

        # M2: Matern-II 带抑制半径的泊松点过程 (替代 rng.uniform 撒点)
        inhibition_radius = block_size * 0.15
        seeds = _generate_alligator_seeds(
            x_min, x_max, y_min, y_max, num_seeds, inhibition_radius, rng
        )

        # Voronoi 图
        vor = spatial.Voronoi(seeds)

        ridge_segments = []
        for v_idx_pair in vor.ridge_vertices:
            v1, v2 = v_idx_pair
            if v1 >= 0 and v2 >= 0:
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

        # 裂缝宽度；形变与标签都限制在 patch 边界（+10cm 边距）内，
        # 防止跨越边缘单元的长脊线把标注拖出局部损坏区。
        in_region = (
            (xy[:, 0] >= x_min - 0.1) & (xy[:, 0] <= x_max + 0.1)
            & (xy[:, 1] >= y_min - 0.1) & (xy[:, 1] <= y_max + 0.1)
        )
        half_width_crack = width_crack / 2.0
        physical_mask = (min_dist_ridge <= half_width_crack) & in_region

        if np.any(physical_mask):
            t = min_dist_ridge[physical_mask] / max(half_width_crack, 1e-12)
            depth = d_max * np.exp(-(t ** 2.0))
            pts[physical_mask, 2] -= depth

        label_half_width = max(half_width_crack, label_width_floor * 0.5)
        label_mask = (min_dist_ridge <= label_half_width) & in_region
        if np.any(label_mask):
            lbl[label_mask] = label_val

    return pts, lbl


# 1.1.4 — 坑槽生成 (Pothole)


def add_pothole(
    points: np.ndarray,
    labels: np.ndarray,
    center: tuple[float, float],
    radius: float,
    depth: float,
    edge_quality: float,
    severity: str,
    seed: int | None = None,
    beta_range: tuple[float, float] | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    分段体制（``params["deterministic_seed"]`` + ``params["pit_seed"]``）：
    边缘剥落改为坐标哈希的逐点确定性决策，微坑参数改由专用 ``pit_seed``
    随机流抽定——历史路径里 ``rng.choice(edge_idx)`` 的消耗随传入云的点集
    变化，会污染其后所有抽取（含 Poisson 微坑），使坑缘剥落/微坑在缝两侧
    不一致（半径大、锚点近缝时扩展域截断剥落环）。坐标哈希 + 专用微坑流
    使任一物理点在任何段里得到相同结果，无需归属段规则（归属段反而会把
    跨缝坑槽在邻段拥有行上截断）。不传 params 时走历史路径，逐位不变。

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        center:  坑槽中心 (cx, cy)。
        radius:  坑槽半径 R (m)。
        depth:   最大深度 d_max (m)。
        edge_quality: 边缘质量因子 [0,1]，1 为完整，0 为严重剥落。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子（历史路径）。
        beta_range: 重度坑槽的 β 采样区间 (lo, hi)；轻度恒为 2（椭球）。
            None 时重度取 [3, 5]（默认口径，与 PotholeConfig.beta_range 一致）。
        params: 可选覆盖：``'deterministic_seed'``（剥落坐标哈希盐）、
            ``'pit_seed'``（微坑专用随机流种子）。

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

    # 超椭圆指数 β：轻度恒为 2（椭球）；重度按 beta_range 采样（平底）。
    if beta_range is not None and not 0.0 < beta_range[0] <= beta_range[1]:
        raise ValueError(f"beta_range must satisfy 0 < lo <= hi, got {beta_range}")
    if severity == "light":
        beta = 2.0
    elif beta_range is not None:
        lo, hi = beta_range
        beta = float(rng.uniform(lo, hi))
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

    det_seed = None if params is None else params.get("deterministic_seed")
    pit_seed = None if params is None else params.get("pit_seed")
    pit_rng = rng if pit_seed is None else np.random.default_rng(int(pit_seed))

    # 边缘剥落 (edge spalling)
    if edge_quality < 1.0:
        edge_mask = (r >= 0.8 * radius) & (r <= 1.2 * radius)
        if np.any(edge_mask):
            edge_idx = np.where(edge_mask)[0]
            if det_seed is not None:
                # 云无关：每个边缘点按坐标哈希以 edge_quality 概率保留。
                u_keep = _coord_hash01(
                    pts[edge_idx, 0], pts[edge_idx, 1], int(det_seed)
                )
                spall_idx = edge_idx[u_keep >= edge_quality]
                if len(spall_idx) > 0:
                    u_depth = _coord_hash01(
                        pts[spall_idx, 0], pts[spall_idx, 1],
                        int(det_seed) ^ 0x27D4EB2F165667C5,
                    )
                    spall_depth = u_depth * (depth * 0.3)
                    pts[spall_idx, 2] -= spall_depth
                    lbl[spall_idx] = label_val
            else:
                num_spall = int(edge_quality * len(edge_idx))
                if num_spall < len(edge_idx):
                    spall_idx = rng.choice(edge_idx, size=len(edge_idx) - num_spall,
                                           replace=False)
                    spall_depth = rng.uniform(0.0, depth * 0.3, size=len(spall_idx))
                    pts[spall_idx, 2] -= spall_depth
                    lbl[spall_idx] = label_val

        # Poisson 微坑 (小尺度次生坑)
        num_pits = pit_rng.poisson(max(1, int(radius * 5)))
        for _ in range(num_pits):
            pit_angle = pit_rng.uniform(0, 2 * np.pi)
            pit_radius = radius * pit_rng.uniform(0.8, 1.3)
            pit_cx = cx + pit_radius * np.cos(pit_angle)
            pit_cy = cy + pit_radius * np.sin(pit_angle)
            pit_r = 0.02 + pit_rng.random() * 0.04  # 微坑半径 2-6cm
            pit_d = pit_rng.uniform(0.001, 0.005)  # 深度 1-5mm

            r_pit = np.sqrt((xy[:, 0] - pit_cx) ** 2 + (xy[:, 1] - pit_cy) ** 2)
            in_pit = r_pit <= pit_r
            if np.any(in_pit):
                r_norm_pit = r_pit[in_pit] / pit_r
                z_pit = -pit_d * (1.0 - r_norm_pit ** 2.0) ** 0.5
                pts[in_pit, 2] += z_pit
                lbl[in_pit] = label_val

    return pts, lbl


# M3: Lévy 极值剥落 — 重尾分布模拟裂缝/坑槽边缘的深层剥落


def add_edge_spalling_heavy_tail(
    points: np.ndarray,
    labels: np.ndarray,
    edge_mask: np.ndarray,
    depth_base: float = 0.005,
    hurst: float = 0.7,
    trigger_prob: float = 0.05,
    label_val: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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


# 1.1.5 — 松散 (Raveling)


def add_raveling(
    points: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray,
    severity: str,
    seed: int | None = None,
    remove_nan: bool = False,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add raveling (松散) — simulate fine aggregate loss.

    模拟细集料脱落：
    1. 在区域内随机移除一定比例的点 (用 ``np.nan`` 标记，调用方过滤)
    2. 对剩余点，高度降低 (高斯噪声模拟表面侵蚀)

    轻 (标签 11)：移除比例 ~5%，深度 ~1-2mm
    重 (标签 12)：移除比例 ~20%，深度 ~3-5mm 点蚀

    可改进点#1: 增加 ``remove_nan`` 参数，为 True 时在返回前自动过滤 NaN
    点并同步裁剪标签数组，减少调用方负担。

    分段体制（``params["deterministic_seed"]``）：摘除与侵蚀改为**坐标
    哈希**的逐点确定性决策，不再用 ``rng.choice(region_idx)``——后者消耗
    随传入云的点集变化，缝两侧取自不同窗口时孔洞/侵蚀模式不一致（松散框
    可占 0.2-0.5·L，必然跨段，且无哪个段的扩展域能完整包含它，故归属段
    规则不适用）。坐标哈希保证同一物理点在任何段的扩展云里决策相同。
    不传 params 时走历史 rng 路径，逐位不变。

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        region_mask: 布尔掩码 (N,)，指示松散发生的区域。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子（历史路径；deterministic 模式下不使用）。
        remove_nan: 是否在返回前自动过滤 NaN 点 (可改进点#1)。
        params: 可选覆盖：``'deterministic_seed'``（int）→ 坐标哈希决策。

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

    det_seed = None if params is None else params.get("deterministic_seed")

    if det_seed is not None:
        # 云无关路径：逐点坐标哈希决定摘除/保留与侵蚀深度。
        salt = int(det_seed)
        u_remove = _coord_hash01(
            pts[region_idx, 0], pts[region_idx, 1], salt
        )
        remove_mask = u_remove < removal_ratio
        remove_idx = region_idx[remove_mask]
        pts[remove_idx] = np.nan
        lbl[remove_idx] = label_val  # 即使移除也标注标签
        remaining_idx = region_idx[~remove_mask]
        if len(remaining_idx) > 0:
            u_erosion = _coord_hash01(
                pts[remaining_idx, 0], pts[remaining_idx, 1],
                salt ^ 0x2545F4914F6CDD1D,
            )
            # 逆变换采样复现历史 N(0.5, 0.3)·depth 的侵蚀分布。
            z_erosion = stats.norm.ppf(
                np.clip(u_erosion, 1e-12, 1.0 - 1e-12)
            )
            erosion = np.clip(
                pitting_depth * 0.5 + pitting_depth * 0.3 * z_erosion,
                0.0,
                pitting_depth * 1.5,
            )
            pts[remaining_idx, 2] -= erosion
            lbl[remaining_idx] = label_val
    else:
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


# 1.1.6 — 沉陷 (Depression)


def add_depression(
    points: np.ndarray,
    labels: np.ndarray,
    center: tuple[float, float],
    radius: float,
    depth: float,
    severity: str,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    depression = -depth * np.exp(-0.5 * r2 / sigma2)

    pts[:, 2] += depression

    # 标注受影响区域 (深度超过 1% 最大深度的点)
    affected = np.abs(depression) > depth * 0.01
    lbl[affected] = label_val

    return pts, lbl


# 1.1.7 — 车辙 (Rutting)


def add_rutting(
    points: np.ndarray,
    labels: np.ndarray,
    center_line: float,
    wheel_separation: float,
    depth: float,
    width: float,
    severity: str,
    seed: int | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    分段体制：``params["modulation_wavelength"]`` 由计划阶段按**全局**
    条带长度固定（Λ = 全局 y 幅 × 0.5）传入；不传时回退为从传入云的
    y 范围推导（单块路径历史行为逐位不变）。

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        center_line: 道路中心线 x 坐标。
        wheel_separation: 左右轮迹间距 (m)。
        depth:  最大深度 (m)。
        width: 单条轮迹宽度 (m)。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子（当前实现为确定性计算，保留接口）。
        params: 可选覆盖：'modulation_wavelength' 纵向调制波长 (m)。

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

    x_left = center_line - wheel_separation / 2.0
    x_right = center_line + wheel_separation / 2.0

    sigma = width / 3.0

    if params is not None and "modulation_wavelength" in params:
        modulation_wavelength = float(params["modulation_wavelength"])
    else:
        y_range = np.max(y) - np.min(y)
        modulation_wavelength = y_range * 0.5
    epsilon = 0.2
    modulation = 1.0 + epsilon * np.sin(2.0 * np.pi * y / modulation_wavelength)

    track_left = np.exp(-0.5 * ((x - x_left) / sigma) ** 2)
    track_right = np.exp(-0.5 * ((x - x_right) / sigma) ** 2)

    rut_depth = depth * (track_left + track_right) * modulation

    pts[:, 2] -= rut_depth

    # 标签边界必须几何可观测：绝对阈值 ≥2mm（旧的 5% 相对阈值在
    # 标注边界处形变仅 ~0.6mm，低于传感器噪声，标签不可学习且把
    # 标注带宽拉到 1m/轮迹，远超 0.6m 物理轮迹宽）。
    affected = rut_depth > max(0.002, depth * 0.05)
    lbl[affected] = label_val

    return pts, lbl


# 1.1.8 — 波浪拥包 (Corrugation)


def add_corrugation(
    points: np.ndarray,
    labels: np.ndarray,
    direction: str,
    wavelength: float,
    amplitude: float,
    severity: str,
    seed: int | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add corrugation (波浪拥包) — sinusoidal height modulation.

    正弦波高度调制：

    .. math::
        z'(x, y) = z(x, y) + A \\cdot \\cos\\left(\\frac{2\\pi}{\\lambda} u\\right)

    其中 :math:`u` 为沿波纹方向的空间坐标：

    - ``direction='longitudinal'``: :math:`u = y`
    - ``direction='transverse'``: :math:`u = x`

    轻 (标签 17)：振幅 10-25mm
    重 (标签 18)：振幅 > 25mm

    波浪拥包是**局部**病害（数米范围的搓板/推挤），不是全路段周期
    起伏：正弦调制乘以高斯空间包络（σ_e ~ 1-3 m），标签只覆盖包络
    支撑域内波幅可观测的点——旧实现无包络时约 94% 的场景点被标注，
    类先验被极端扭曲。注意 amplitude 指单侧波高；若 JTG 分级量按
    峰谷差解读则为 2A，正式实验前需对照标准原文口径（文档声明）。

    分段体制：``params["envelope_center"]``/``params["envelope_sigma"]``
    由计划阶段按**全局**条带 bbox 抽定后传入（不传时回退为从传入云
    的 bbox 抽取——单块路径历史行为逐位不变）。

    Args:
        points:  点云 (N, 3)。
        labels:  标签 (N,)。
        direction: 波纹方向 'longitudinal' | 'transverse'。
        wavelength: 波长 (m)。
        amplitude:  振幅 (m)。
        severity: 严重程度 'light' | 'severe'。
        seed: 随机种子（包络中心与尺寸）。
        params: 可选覆盖：'envelope_center' (cx, cy)、'envelope_sigma' σ_e。

    Returns:
        points_modified:  修改后点云 (N, 3)。
        labels_modified: 修改后标签 (N,)。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    lbl = labels.copy()

    label_val = 17 if severity == "light" else 18

    if direction == "longitudinal":
        u = pts[:, 1]  # y 方向
    elif direction == "transverse":
        u = pts[:, 0]  # x 方向
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # 高斯空间包络：局部搓板区（中心随机、σ_e ∈ [0.7, 1.3] m）。
    if params is not None and "envelope_center" in params:
        cx, cy = (float(v) for v in params["envelope_center"])
        sigma_e = float(params["envelope_sigma"])
    else:
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))
        sigma_e = float(rng.uniform(0.7, 1.3))
    envelope = np.exp(
        -((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) / (2.0 * sigma_e ** 2)
    )

    modulation = amplitude * envelope * np.cos(2.0 * np.pi * u / wavelength)
    pts[:, 2] += modulation

    # 标签 = 包络支撑域内局部波幅可观测（≥2mm 且 ≥20% 名义振幅）的点，
    # 即标注半径 ≤ σ_e·√(2·ln 5) ≈ 1.8σ_e —— 局部病害而非全场。
    local_amp = amplitude * envelope
    affected = local_amp > max(0.002, amplitude * 0.2)
    lbl[affected] = label_val

    return pts, lbl


# 1.1.9 — 泛油 (Bleeding)


def add_bleeding(
    points: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Add bleeding (泛油) — reflectance change only, NO geometry change.

    泛油仅改变路面反射率 (强度值)，不改变几何形状。标签 = 19。

    物理方向：泛油是沥青结合料上泛形成的光滑油膜，在 905/1550 nm
    近红外波段结合料强吸收（反照率低于外露集料），且光滑表面的
    镜面反射使非垂直入射的后向散射进一步减弱——泛油区域在 MLS
    强度影像中通常**偏暗**。强度修正由生成器的辐射模型
    (``_compute_intensity``, 反照率 ×0.55) 统一施加，本函数只贴标签。

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


# 1.1.10 — 水泥路面损坏 (Concrete Damage)


def add_patching(
    points: np.ndarray,
    labels: np.ndarray,
    center: tuple[float, float],
    width: float,
    length: float,
    label: int,
    angle_rad: float = 0.0,
    elevation: float = 0.0,
    edge_width: float = 0.08,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a finite-width repair patch with a smooth geometric transition.

    The patch is a rounded rectangle in the local road tangent plane. Its
    height is blended to a least-squares reference plane with a cubic smoothstep
    over ``edge_width``. This gives a continuous height field instead of a
    label-only rectangle, while preserving nearby road roughness outside the
    repair boundary.

    分段体制：``params["reference_plane"] = (a, b, c)`` 由计划阶段在
    全局宏观 + 设计几何场上拟合后传入（纹理零均值，lstsq 天然平均掉），
    替代对传入云的 lstsq——单块路径不传 params 时行为逐位不变。
    """
    if width <= 0.0 or length <= 0.0:
        raise ValueError("patch width and length must be positive")
    if edge_width <= 0.0:
        raise ValueError("patch edge_width must be positive")

    pts = points.copy()
    lbl = labels.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    cx, cy = center

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    local_x = cos_a * (x - cx) + sin_a * (y - cy)
    local_y = -sin_a * (x - cx) + cos_a * (y - cy)

    half_width = width * 0.5
    half_length = length * 0.5
    corner_radius = min(half_width, half_length, max(edge_width, 0.02))
    qx = np.abs(local_x) - half_width + corner_radius
    qy = np.abs(local_y) - half_length + corner_radius
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    signed_distance = outside + np.minimum(np.maximum(qx, qy), 0.0) - corner_radius

    support = signed_distance <= edge_width
    core = signed_distance <= 0.0
    if not np.any(core):
        return pts, lbl

    design = np.column_stack((x, y, np.ones_like(x)))
    if params is not None and "reference_plane" in params:
        plane = np.asarray(params["reference_plane"], dtype=np.float64)
    else:
        plane, *_ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
    target_height = design @ plane + elevation

    t = np.clip(1.0 - np.maximum(signed_distance, 0.0) / edge_width, 0.0, 1.0)
    blend = t * t * (3.0 - 2.0 * t)
    pts[support, 2] += blend[support] * (target_height[support] - pts[support, 2])
    lbl[core] = label
    return pts, lbl


def add_concrete_damage(
    points: np.ndarray,
    labels: np.ndarray,
    damage_type: str,
    severity: str,
    params: dict,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
            - 'domain_bounds': (x_min, x_max, y_min, y_max) 全局条带 bbox
              （分段体制；替代从传入云推导的 bbox，使接缝线/线中心等
              bbox 派生量跨段一致）
            - 'target_slab': (ix, iy) 计划固定的目标板块（slab_shatter /
              corner_break / blowup）
            - 'slab_bounds': (x0, x1, y0, y1) 计划固定的板块盒（同上三型，
              使碎片种子/拱起中心跨段一致）
            - 'corner_signs': (sx, sy) 计划固定的板角选向（corner_break）
            - 'corner_reach': 计划预计算的角点距离（corner_break，跨缝
              板块的倾斜场跨段一致）
            - 'joint_y': 计划固定的错台接缝 y 坐标（faulting）
            - 'pit_center': (cx, cy) 计划固定的坑洞中心（pitting）
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

    # 解析板块参数
    slab_len = params.get("slab_length", 5.0)
    slab_wid = params.get("slab_width", 4.0)
    joint_w = params.get("joint_width", 0.008)
    # P1-4 修复：支持板块偏移参数
    x_offset = params.get("x_offset", 0.0)
    y_offset = params.get("y_offset", 0.0)

    # 获取标签
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

    # 辅助：计算点到 slab 接缝的距离
    # 接缝位置。分段体制：params["domain_bounds"]（全局条带 bbox）由
    # 计划阶段传入，替代从传入云推导的 bbox——保证跨段的接缝线列表、
    # faulting 的 joint_y、slab_crack 的线中心、pitting 的中心等
    # bbox 派生量逐段一致（单块路径不传时行为逐位不变）。
    if "domain_bounds" in params:
        x_min, x_max, y_min, y_max = (float(v) for v in params["domain_bounds"])
    else:
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

    # 1) 破碎板 (Slab Shatter)
    if damage_type == "slab_shatter":
        # 选择一块或多块板，将板内点分割成 Voronoi 碎片
        # 每个碎片随机垂直错台
        # 分段体制：params["target_slab"]/["slab_bounds"] 由计划阶段按
        # 全局板块列表固定后传入（跨段一致）；不传时回退为从传入云的
        # 板块集合抽取（单块路径历史行为逐位不变）。下同。
        ix, iy = slab_indices(x, y)
        if "target_slab" in params:
            target_slab = np.asarray(params["target_slab"], dtype=int)
        else:
            unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
            target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 3:
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            if "slab_bounds" in params:
                x_s_min, x_s_max, y_s_min, y_s_max = (
                    float(v) for v in params["slab_bounds"]
                )
            else:
                x_s_min, x_s_max = np.min(slab_x), np.max(slab_x)
                y_s_min, y_s_max = np.min(slab_y), np.max(slab_y)

            n_fragments = rng.integers(3, 8)
            seeds = np.column_stack([
                rng.uniform(x_s_min, x_s_max, n_fragments),
                rng.uniform(y_s_min, y_s_max, n_fragments),
            ])

            vor = spatial.Voronoi(seeds)
            tree = spatial.KDTree(seeds)
            _, frag_idx = tree.query(np.column_stack([slab_x, slab_y]))

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

    # 2) 水泥裂缝 (Concrete Crack)
    elif damage_type == "slab_crack":
        # 通过板体的直线裂缝，边缘更尖锐
        angle = rng.uniform(0, np.pi)
        x_center = (x_min + x_max) * 0.5
        y_center = (y_min + y_max) * 0.5
        normal_x, normal_y = np.cos(angle), np.sin(angle)
        dist_to_line = np.abs(normal_x * (x - x_center) + normal_y * (y - y_center))

        crack_width_joint = joint_w * 2
        physical_mask = dist_to_line < crack_width_joint

        if np.any(physical_mask):
            d_max_crack = params.get("d_max", 0.015 if severity == "light" else 0.040)
            t = dist_to_line[physical_mask] / crack_width_joint
            # 更尖锐的剖面 (p=1 近似 V 形)
            depth_crack = d_max_crack * np.exp(-(t ** 1.0))
            pts[physical_mask, 2] -= depth_crack

        label_width_floor = params.get("label_width_floor", 0.0)
        label_mask = dist_to_line < max(crack_width_joint, label_width_floor * 0.5)
        if np.any(label_mask):
            lbl[label_mask] = label_val

    # 3) 板角断裂 (Corner Break)
    elif damage_type == "corner_break":
        # 选择一个板块的一个角
        ix, iy = slab_indices(x, y)
        if "target_slab" in params:
            target_slab = np.asarray(params["target_slab"], dtype=int)
        else:
            unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
            target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 0:
            # 板角断裂的定义（JTG/PCA 口径）：一条弦线分别与横缝和纵缝
            # 相交（交点距角点均小于半板边长），把板角切下一个三角形块，
            # 该块整体断裂并沉降/倾斜——裂缝**不通过角点**。旧实现是
            # "过角点的射线带 + 圆形沉降盘"，形态与真实病害不同。
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            if "slab_bounds" in params:
                x_s_min, x_s_max, y_s_min, y_s_max = (
                    float(v) for v in params["slab_bounds"]
                )
            else:
                x_s_min, x_s_max = np.min(slab_x), np.max(slab_x)
                y_s_min, y_s_max = np.min(slab_y), np.max(slab_y)
            slab_w_local = max(x_s_max - x_s_min, 1e-6)
            slab_l_local = max(y_s_max - y_s_min, 1e-6)

            # 选一个角及指向板内的方向（分段体制可经 params["corner_signs"]
            # 固定，保证跨段选角一致）
            if "corner_signs" in params:
                sign_x, sign_y = (float(v) for v in params["corner_signs"])
                corner_x = x_s_min if sign_x > 0 else x_s_max
                corner_y = y_s_min if sign_y > 0 else y_s_max
                sx = 1.0 if corner_x == x_s_min else -1.0
                sy = 1.0 if corner_y == y_s_min else -1.0
            else:
                corner_x = x_s_min if rng.random() < 0.5 else x_s_max
                corner_y = y_s_min if rng.random() < 0.5 else y_s_max
                sx = 1.0 if corner_x == x_s_min else -1.0
                sy = 1.0 if corner_y == y_s_min else -1.0

            # 弦线端点：沿两条相邻接缝各取一个交点（0.2-0.5 倍边长，
            # 测试可通过 params 固定）。
            frac_a = float(params.get("chord_frac_x", rng.uniform(0.2, 0.5)))
            frac_b = float(params.get("chord_frac_y", rng.uniform(0.2, 0.5)))
            p_a = np.array([corner_x + sx * frac_a * slab_w_local, corner_y])
            p_b = np.array([corner_x, corner_y + sy * frac_b * slab_l_local])

            # 三角形内部 = 弦线的角点侧半平面（限于本板）。
            chord = p_b - p_a
            rel_pts = np.stack([slab_x - p_a[0], slab_y - p_a[1]], axis=1)
            cross_pts = chord[0] * rel_pts[:, 1] - chord[1] * rel_pts[:, 0]
            cross_corner = chord[0] * (corner_y - p_a[1]) - chord[1] * (corner_x - p_a[0])
            triangle = cross_pts * np.sign(cross_corner) >= 0.0

            chord_len = float(np.linalg.norm(chord))
            dist_chord = np.abs(cross_pts) / max(chord_len, 1e-9)

            d_max_corner = params.get("d_max", 0.006 if severity == "light" else 0.015)
            if np.any(triangle):
                # 三角块整体沉降 + 向角点方向的线性倾斜（断裂块转动）。
                dist_to_corner = np.sqrt(
                    (slab_x - corner_x) ** 2 + (slab_y - corner_y) ** 2
                )
                # 分段体制：params["corner_reach"] 由计划阶段在全局网格
                # 窗口上预计算（板块跨缝时各段只看到部分三角），保证倾斜
                # 场跨段一致。
                if "corner_reach" in params:
                    reach = float(params["corner_reach"])
                else:
                    reach = float(dist_to_corner[triangle].max()) or 1.0
                tilt = 0.5 + 0.5 * (1.0 - dist_to_corner[triangle] / reach)
                pts[slab_idx[triangle], 2] -= d_max_corner * tilt
                lbl[slab_idx[triangle]] = label_val

            # 沿弦线刻 V 槽（裂缝本体），标签同类。
            groove = (dist_chord < joint_w * 3) & (
                cross_pts * np.sign(cross_corner) >= -joint_w * 3 * chord_len
            )
            if np.any(groove):
                pts[slab_idx[groove], 2] -= 0.5 * d_max_corner
                lbl[slab_idx[groove]] = label_val

    # 4) 错台 (Faulting)
    elif damage_type == "faulting":
        # 接缝两侧高度差（分段体制：params["joint_y"] 由计划阶段在全局
        # 接缝列表上抽定传入；domain_bounds 为全局时回退路径也逐段一致）
        if "joint_y" in params:
            joint_y = float(params["joint_y"])
        else:
            interior_trans = trans_joints[(trans_joints > y_min) & (trans_joints < y_max)]
            if len(interior_trans):
                joint_y = float(rng.choice(interior_trans))
            else:
                joint_y = (y_min + y_max) * 0.5

        fault_offset = params.get("d_max", 0.005 if severity == "light" else 0.015)

        # 错台是接缝两侧**整板**的相对竖向位移（基层唧泥/塌陷），
        # z 场为逐板分片常量、在接缝处一次跳变——旧实现只位移接缝
        # 两侧 ±40mm 窄条带，制造出两个物理上不存在的反向台阶。
        # （恰在接缝线上的点归属单侧，避免出现第三个中间高程级。）
        side = np.where(y >= joint_y, 1.0, -1.0)
        pts[:, 2] += side * fault_offset * 0.5
        # JTG 在接缝处量测错台，标签仍限于接缝邻域条带。
        near_joint = np.abs(y - joint_y) < joint_w * 5
        if np.any(near_joint):
            lbl[near_joint] = label_val

    # 5) 唧泥 (Pumping)
    elif damage_type == "pumping":
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

    # 6) 边角剥落 (Edge Spall)
    elif damage_type == "edge_spall":
        # 板块边缘破损
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

    # 7) 接缝料损坏 (Joint Damage)
    elif damage_type == "joint_damage":
        # 接缝处填充物缺失 → 形成沟槽
        near_joint = joint_dist < joint_w

        if np.any(near_joint):
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

    # 8) 坑洞 (Pitting) — 类似坑槽但更深
    elif damage_type == "pitting":
        # 分段体制：params["pit_center"] 由计划阶段按全局 bbox 抽定传入
        if "pit_center" in params:
            center_x, center_y = (float(v) for v in params["pit_center"])
        else:
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

    # 9) 拱起 (Blowup)
    elif damage_type == "blowup":
        # 板体向上拱起 (余弦拱形)
        ix, iy = slab_indices(x, y)
        if "target_slab" in params:
            target_slab = np.asarray(params["target_slab"], dtype=int)
        else:
            unique_slabs = np.unique(np.stack([ix, iy], axis=1), axis=0)
            target_slab = unique_slabs[rng.integers(len(unique_slabs))]

        slab_mask = (ix == target_slab[0]) & (iy == target_slab[1])
        slab_idx = np.where(slab_mask)[0]

        if len(slab_idx) > 0:
            slab_x = x[slab_idx]
            slab_y = y[slab_idx]
            if "slab_bounds" in params:
                x_s_min, x_s_max, y_s_min, y_s_max = (
                    float(v) for v in params["slab_bounds"]
                )
            else:
                x_s_min, x_s_max = np.min(slab_x), np.max(slab_x)
                y_s_min, y_s_max = np.min(slab_y), np.max(slab_y)
            x_s_center = (x_s_min + x_s_max) / 2.0
            y_s_center = (y_s_min + y_s_max) / 2.0

            blowup_height = params.get("d_max", rng.uniform(0.02, 0.06))

            dx_blow = (slab_x - x_s_center) / (slab_wid * 0.5)
            dy_blow = (slab_y - y_s_center) / (slab_len * 0.5)
            r_blow = np.sqrt(dx_blow ** 2 + dy_blow ** 2)
            r_blow = np.clip(r_blow, 0, 1.5)

            uplift = blowup_height * (np.cos(np.pi * r_blow * 0.5) ** 2)
            uplift = np.clip(uplift, 0, blowup_height)

            pts[slab_idx, 2] += uplift
            uplifted = uplift > blowup_height * 0.1
            lbl[slab_idx[uplifted]] = label_val

    # 10) 露骨 (Exposed Aggregate)
    elif damage_type == "exposed_aggregate":
        # 表面纹理缺失 → 增加粗糙度 + 降低强度
        region_idx = np.where(
            (x > x_min + 0.1 * (x_max - x_min)) &
            (x < x_max - 0.1 * (x_max - x_min)) &
            (y > y_min + 0.1 * (y_max - y_min)) &
            (y < y_max - 0.1 * (y_max - y_min))
        )[0]

        if len(region_idx) > 3:
            n_exposed = max(1, int(len(region_idx) * rng.uniform(0.1, 0.4)))
            exposed_idx = rng.choice(region_idx, size=n_exposed, replace=False)

            roughness = rng.normal(0, params.get("d_max", 0.003), size=n_exposed)
            pts[exposed_idx, 2] += roughness
            lbl[exposed_idx] = label_val

    else:
        raise ValueError(f"Unknown concrete damage type: {damage_type}")

    return pts, lbl


# 1.1.10b — 局部凹陷深度与几何遮挡


def local_depression_depth(points: np.ndarray, cell: float = 0.08) -> np.ndarray:
    """每点相对局部参考面（0.08 m 粗网格中位数）的凹陷深度 (≥0, m)。

    供强度遮蔽（凹陷内回波弱）与几何遮挡共用同一个"低于周边路表
    多少"的量。
    """
    n = points.shape[0]
    x_min = points[:, 0].min()
    y_min = points[:, 1].min()
    ix = np.clip(((points[:, 0] - x_min) / cell).astype(np.int64), 0, None)
    iy = np.clip(((points[:, 1] - y_min) / cell).astype(np.int64), 0, None)
    cell_id = ix * (iy.max() + 1) + iy
    order = np.argsort(cell_id, kind="stable")
    sorted_ids = cell_id[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    z_ref = np.empty(n, dtype=np.float64)
    for group in np.split(order, boundaries):
        z_ref[group] = np.median(points[group, 2])
    return np.maximum(z_ref - points[:, 2], 0.0)


def geometric_occlusion_keep_mask(
    points: np.ndarray,
    sensor_origin: np.ndarray,
    depth: np.ndarray,
    rng: np.random.Generator,
    min_depth: float = 0.004,
    transition: float = 0.003,
) -> np.ndarray:
    """深窄凹陷内部点的几何遮挡掩码（True = 保留）。

    对光束入射角 θ（自竖直），深度 d 处的点可见当且仅当凹陷开口
    半宽 w_half 满足 :math:`d \\le d_{vis} = w_{half}/\\tan\\theta`
    （审计 F5.8：深 30 mm、开口 5 mm 的裂缝物理上不可能被完整
    回波——真实数据中窄裂缝表现为点缺失的线，而非被完整采样的
    深沟）。w_half 用点到最近非凹陷 (rim) 点的水平距离近似：
    裂缝内 w_half ≈ 半缝宽 → 深处被遮挡；宽坑槽 w_half 可达半径
    → 几乎全可见。遮挡概率随 (d - d_vis) 平滑上升，上限 0.95。
    """
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    deep = depth > min_depth
    if not np.any(deep) or np.all(deep):
        return keep

    rim_xy = points[~deep][:, :2]
    tree = spatial.cKDTree(rim_xy)
    w_half, _ = tree.query(points[deep][:, :2], k=1)

    origin = np.asarray(sensor_origin, dtype=np.float64).reshape(3)
    d_vec = points[deep] - origin[None, :]
    horiz = np.hypot(d_vec[:, 0], d_vec[:, 1])
    vert = np.maximum(origin[2] - points[deep, 2], 1e-3)
    # tanθ = 水平距离 / 垂直高差；d_vis = w_half / tanθ
    d_vis = w_half * vert / np.maximum(horiz, 1e-6)

    p_occ = np.clip((depth[deep] - d_vis) / max(transition, 1e-6), 0.0, 0.95)
    keep[deep] = rng.random(int(deep.sum())) > p_occ
    return keep


# 1.1.11 — LiDAR 噪声仿真


def simulate_lidar_noise(
    points: np.ndarray,
    distance_noise_std: float,
    dropout_rate: float,
    angular_jitter_deg: float,
    seed: int | None = None,
    enable_edge_mixing: bool = True,
    mixed_pixel_prob: float = 0.01,
    curvature: np.ndarray | None = None,
    curvature_threshold: float = 0.5,
    sensor_origin: np.ndarray | None = None,
    sigma_r: np.ndarray | None = None,
    drop_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate LiDAR measurement noise on point cloud.

    1. **球坐标噪声（传感器坐标系）**：
       以 ``sensor_origin`` 为球心把点云转换为 :math:`(r, \\theta, \\phi)`，
       对距离加高斯噪声、角度加高斯抖动，再转换回笛卡尔坐标。测距
       误差沿真实波束方向 :math:`\\mathbf d = (\\mathbf p - \\mathbf p_s)/R`
       作用，其垂直分量 :math:`n_r |d_z|` 是病害深度测量误差的主体
       —— 这要求传感器位于路面上方（车载 h≈2 m），而不是路面角点。

    2. **测距噪声异方差**：``sigma_r`` 给定时使用逐点标准差
       :math:`\\sigma_{r,i}`。物理依据：:math:`\\sigma_r \\propto
       1/\\sqrt{\\mathrm{SNR}}`，暗、掠射、远处的点噪声更大
       (Wujanz et al. 2017; Soudarissanane et al. 2011)。

    3. **Dropout（辐射耦合）**：``drop_weight`` 给定时（接收功率代理
       :math:`\\propto \\rho\\cos\\theta/R^2`），丢点概率随功率下降
       单调上升 :math:`P_i = \\mathrm{clip}(p_d \\cdot \\tilde w / w_i,
       0, 0.95)`（:math:`\\tilde w` 为中位功率），低强度 ↔ 高丢点
       的真实耦合得以保留；未提供时退回旧的距离比例模型。

    4. **边缘混合效应**：仅对曲率相对阈值筛出的病害边缘点做邻域
       均值混合（模拟混合像元）。

    Args:
        points:  点云 (N, 3)。
        distance_noise_std: 距离噪声标准差 (m)，``sigma_r`` 为 None 时使用。
        dropout_rate: 基础点丢失概率 [0, 1)。
        angular_jitter_deg: 角度抖动标准差 (度)。
        seed: 随机种子。
        enable_edge_mixing: 是否启用边缘混合。
        mixed_pixel_prob: 混合像素点比例。
        curvature: 曲率数组 (N,) 用于筛选边缘点。
        curvature_threshold: 相对曲率阈值（乘以 std(|curvature|)）。
        sensor_origin: 传感器位姿 (3,)。None 时退回旧的原点球心
            （仅为向后兼容；正式管线必须传入车载位姿）。
        sigma_r: 逐点测距噪声标准差 (N,)，None 时用常数。
        drop_weight: 逐点接收功率代理 (N,)，用于辐射耦合丢点。

    Returns:
        噪声点云 (N', 3)，其中 N' ≤ N。NaN 点已被移除。
    """
    rng = np.random.default_rng(seed)
    pts = points.copy()
    N = pts.shape[0]

    if sensor_origin is None:
        origin = np.zeros(3, dtype=np.float64)
    else:
        origin = np.asarray(sensor_origin, dtype=np.float64).reshape(3)

    # 1. 平移到传感器坐标系后做笛卡尔 → 球坐标
    q = pts - origin[None, :]
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
    theta = np.arctan2(y, x)
    phi = np.arcsin(np.clip(z / r, -1.0, 1.0))

    # 2. 加噪声（sigma_r 提供时逐点异方差）
    angular_jitter_rad = np.deg2rad(angular_jitter_deg)

    if sigma_r is not None:
        sigma = np.asarray(sigma_r, dtype=np.float64).reshape(N)
        r_noisy = r + rng.normal(0.0, 1.0, size=N) * sigma
    else:
        r_noisy = r + rng.normal(0.0, distance_noise_std, size=N)
    r_noisy = np.maximum(r_noisy, 0.0)
    theta_noisy = theta + rng.normal(0.0, angular_jitter_rad, size=N)
    phi_noisy = phi + rng.normal(0.0, angular_jitter_rad, size=N)
    phi_noisy = np.clip(phi_noisy, -np.pi / 2.0, np.pi / 2.0)

    # 3. 球坐标 → 笛卡尔，再平移回路面坐标系
    pts_noisy = np.empty_like(pts)
    pts_noisy[:, 0] = r_noisy * np.cos(theta_noisy) * np.cos(phi_noisy)
    pts_noisy[:, 1] = r_noisy * np.sin(theta_noisy) * np.cos(phi_noisy)
    pts_noisy[:, 2] = r_noisy * np.sin(phi_noisy)
    pts_noisy += origin[None, :]

    # 4. Dropout
    if dropout_rate > 0:
        if drop_weight is not None:
            # 辐射耦合：P(drop) 随接收功率下降单调上升，中位功率处
            # 恰为基础丢失率。
            w = np.asarray(drop_weight, dtype=np.float64).reshape(N)
            w = np.maximum(w, 1e-12)
            w_ref = float(np.median(w))
            dropout_prob = np.clip(dropout_rate * (w_ref / w), 0.0, 0.95)
        else:
            # 旧行为（无功率信息的调用方）：距离比例附加项。
            r_max = np.max(r_noisy)
            dist_factor = r / max(r_max, 1e-12)
            dropout_prob = dropout_rate + (1.0 - dropout_rate) * dist_factor * 0.2
            dropout_prob = np.clip(dropout_prob, 0.0, 0.99)

        keep_mask = rng.random(N) > dropout_prob
        pts_noisy = pts_noisy[keep_mask]
    else:
        keep_mask = np.ones(N, dtype=bool)

    # 5. 边缘混合效应 (Edge Mixing) — P2-3 修复
    if enable_edge_mixing and len(pts_noisy) > 10:
        n_mixed = max(1, int(len(pts_noisy) * mixed_pixel_prob))

        # P2-3: 如果提供了曲率，仅在高曲率区域混合（病害边缘）
        if curvature is not None and len(curvature) > 0:
            # 曲率与 keep_mask 对齐：需要从原始 curvature 对应到保留后的点
            # 假设 curvature 在调用前已与输入 points 对齐
            cur = curvature[keep_mask] if len(curvature) == N else curvature
            # 高曲率区域的索引
            cur_abs = np.abs(cur)
            edge_candidates = np.where(cur_abs > curvature_threshold * np.std(cur_abs))[0]
            if len(edge_candidates) > n_mixed:
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


# __main__ — 自检脚本 (测试所有 11 个函数)

if __name__ == "__main__":
    seed = 42
    passed = 0
    total = 11

    test_width = 0.5
    test_length = 0.5
    test_grid_res = 0.02
    expected_n = int(test_width / test_grid_res) * int(test_length / test_grid_res)

    print("\n[1/11] generate_road_surface ...", end=" ")
    try:
        pts, nrm = generate_road_surface(
            width=test_width, length=test_length, grid_res=test_grid_res,
            pavement_type="asphalt", roughness_class="A", seed=seed,
        )
        N_actual = pts.shape[0]
        assert pts.shape == (N_actual, 3), f"points shape: {pts.shape}"
        assert nrm.shape == (N_actual, 3), f"normals shape: {nrm.shape}"
        normal_norms = np.linalg.norm(nrm, axis=1)
        assert np.allclose(normal_norms, 1.0, atol=1e-6), (
            f"Normals not unit: max dev={np.max(np.abs(normal_norms - 1.0))}"
        )
        assert np.all(np.isfinite(pts)), "Non-finite points detected"
        print(f"OK (N={N_actual})")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[2/11] add_micro_texture ...", end=" ")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        pts2, nrm2 = add_micro_texture(
            pts, nrm, amplitude=0.001, hurst=0.7, octaves=4, seed=seed,
        )
        assert pts2.shape == pts.shape, f"shape mismatch: {pts2.shape} vs {pts.shape}"
        assert nrm2.shape == nrm.shape, "normal shape mismatch"
        assert np.all(np.isfinite(pts2)), "Non-finite points"
        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[3/11] add_crack ...", end=" ")
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
                unique_labels = np.unique(lbl_c)
                for ul in unique_labels:
                    if ul > 0:
                        assert 1 <= ul <= 8, f"Label {ul} out of range [1, 8]"

        print("OK (4 types × 2 severities)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[4/11] add_pothole ...", end=" ")
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
        assert 9 in np.unique(lbl_p) or np.all(lbl_p == 0), (
            f"Expected label 9, got {np.unique(lbl_p)}"
        )
        z_diff = pts_p[:, 2] - pts[:, 2]
        assert np.all(z_diff <= 0 + 1e-10), "Pothole should only lower points"

        pts_p2, lbl_p2 = add_pothole(
            pts, lbl, center=(cx, cy), radius=0.08, depth=0.04,
            edge_quality=0.5, severity="severe", seed=seed,
        )
        assert 10 in np.unique(lbl_p2) or np.all(lbl_p2 == 0)

        print("OK (light=9, severe=10)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[5/11] add_raveling ...", end=" ")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        region_mask = pts[:, 0] > test_width / 2.0

        pts_r, lbl_r = add_raveling(
            pts, lbl, region_mask=region_mask, severity="light", seed=seed,
        )
        assert pts_r.shape == pts.shape
        assert lbl_r.shape == lbl.shape
        unique_lbl = np.unique(lbl_r)
        assert 0 in unique_lbl, "Background points should remain label 0"
        assert 11 in unique_lbl or np.any(lbl_r[region_mask] == 11), (
            "Raveling region should have label 11"
        )

        n_nan = np.sum(np.any(np.isnan(pts_r), axis=1))
        print(f"OK ({n_nan} NaN points)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[6/11] add_depression ...", end=" ")
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
        center_d = np.argmin((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        assert z_diff[center_d] < -0.01, (
            f"Center depression too shallow: {z_diff[center_d]}"
        )
        assert np.max(np.abs(z_diff)) <= 0.02 + 1e-6

        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[7/11] add_rutting ...", end=" ")
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
        assert np.max(np.abs(z_diff)) <= 0.015 * 1.5 + 1e-6

        pts_ru2, lbl_ru2 = add_rutting(
            pts, lbl, center_line=test_width / 2.0,
            wheel_separation=0.2, depth=0.03, width=0.08,
            severity="severe", seed=seed,
        )
        z_diff2 = pts_ru2[:, 2] - pts[:, 2]
        assert np.any(z_diff2 < -0.015)

        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[8/11] add_corrugation ...", end=" ")
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
        assert np.max(np.abs(z_diff)) <= 0.015 + 1e-6

        pts_c2, lbl_c2 = add_corrugation(
            pts, lbl, direction="transverse",
            wavelength=0.1, amplitude=0.02, severity="severe", seed=seed,
        )
        z_diff2 = pts_c2[:, 2] - pts[:, 2]
        assert np.any(np.abs(z_diff2) > 0.015)

        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[9/11] add_bleeding ...", end=" ")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        lbl = np.zeros(pts.shape[0], dtype=np.int64)

        region = pts[:, 0] > test_width / 2.0
        lbl_b = add_bleeding(pts, lbl, region_mask=region, seed=seed)
        assert lbl_b.shape == lbl.shape
        assert np.all(lbl_b[region] == 19), "Bleeding region should be label 19"
        assert np.all(lbl_b[~region] == 0), "Background should remain label 0"

        print("OK")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[10/11] add_concrete_damage ...", end=" ")
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

        print("OK (10 types)")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[11/11] simulate_lidar_noise ...", end=" ")
    try:
        pts, nrm = generate_road_surface(test_width, test_length, test_grid_res,
                                         seed=seed)
        pts_noisy = simulate_lidar_noise(
            pts, distance_noise_std=0.01, dropout_rate=0.05,
            angular_jitter_deg=0.01, seed=seed,
        )
        assert pts_noisy.shape[1] == 3
        assert pts_noisy.shape[0] <= pts.shape[0], (
            f"Noisy points {pts_noisy.shape[0]} > original {pts.shape[0]}"
        )
        assert np.all(np.isfinite(pts_noisy)), "Non-finite points in noisy cloud"
        assert pts_noisy.shape[0] >= pts.shape[0] * 0.8, (
            f"Too many points dropped: {pts_noisy.shape[0]}/{pts.shape[0]}"
        )

        pts_noisy2 = simulate_lidar_noise(
            pts, distance_noise_std=0.005, dropout_rate=0.0,
            angular_jitter_deg=0.005, seed=seed,
        )
        assert pts_noisy2.shape[0] == pts.shape[0], (
            f"Without dropout, shape should match: {pts_noisy2.shape[0]} vs {pts.shape[0]}"
        )

        print(f"OK (dropped {pts.shape[0] - pts_noisy.shape[0]}/{pts.shape[0]})")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")

    # 汇总
    print("\n" + "=" * 72)
    print(f"  结果: {passed}/{total} 测试通过")
    if passed == total:
        print("  [OK] 全部通过 — primitives.py 实现完成")
    else:
        print(f"  [FAIL] {total - passed} 个测试失败")
    print("=" * 72)
