"""Height-field line-profiler resampling with an uncalibrated noise model.

The implementation bilinearly resamples a single-valued road grid at fixed
transverse/longitudinal spacing. Missing nearest source cells remain missing;
labels use nearest XY support and normals interpolate the latent surface.
This is not calibrated laser/camera ray tracing: there is no beam footprint,
self-occlusion calculation, optical triangulation or motion reconstruction.
``laser_incidence_deg`` affects the heuristic noise multiplier, not sample XY.

The stored production intensity still comes from the generator's radiometric
proxy; the per-profile brightness here drives speckle, noise and dropout only.
The Gamma brightness factor, angular scaling, height-noise standard deviation,
flyers and dropout probabilities are explicit engineering hypotheses. Published
instrument precision cannot by itself be interpreted as a Gaussian sigma.
Csencsics et al. (2022), DOI 10.1109/I2MTC48687.2022.9806610, motivates treating
speckle as a measurement uncertainty; the formula here is not their validated
optical simulation. ISO 13473-3:2002 concerns instrument classification, not
calibration of these defaults.

Off-grid points such as curbs are excluded from the regular road lattice.
Output labels/normals are authoritative at sampled XY in the production
line-profiler path. They are synthetic support/latent-normal annotations,
not normals estimated from a noisy real point cloud. The returned audit records
sampling bandwidth and the unmodelled optical effects without implying that
sampling intervals prove sensor accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import LineProfilerConfig

__all__ = ["LineProfilerConfig", "LineProfilerResult", "simulate_line_profiler"]

#: 格网回填时允许的坐标往返误差（米）。arange 网格上的浮点往返误差在
#: 1e-16 相对量级，1e-6·grid_res 的绝对容差对任何实际 grid_res 都足够
#: 宽松，同时仍能把缘石等真正异位的点排除在外。
_LATTICE_TOL_REL = 1e-6
_LATTICE_TOL_ABS = 1e-12


@dataclass
class LineProfilerResult:
    """线激光剖面仪仿真的输出点云与审计字段。

    Attributes:
        points: 输出点云 (M, 3) = [x, y, z_meas]。
        normals: 采样点的潜在表面法向量 (M, 3)，按四角插值并归一。
        labels: 采样点标签 (M,)（剖面内最近格网标签，距离阈值
            ``2·lateral_res_m`` 之外或无格网标签时为 -1 = 未知）。
        profile_index: 每点所属剖面序号 (M,)。
        flyer_mask: 飞点标记 (M,) bool。
        dropout_count: 丢点计数（低亮度丢点 + 源格网空洞导致的不可测
            样本；被丢弃点不在输出中）。
    """

    points: np.ndarray
    normals: np.ndarray
    labels: np.ndarray
    profile_index: np.ndarray
    flyer_mask: np.ndarray
    dropout_count: int
    sampling_audit: dict | None = None


def _rebuild_surface_lattice(
    points: np.ndarray,
    nx: int,
    ny: int,
    grid_res: float,
    x0: float,
    y0: float,
) -> tuple[np.ndarray, int]:
    """把点云按格网坐标回填成 ``(nx, ny)`` 索引阵（-1 = 该格网无点）。

    输入云不保证恰好是 ``nx*ny`` 行的完整规则网格：松散过滤会删行、
    缘石并入会追加异位点。本函数对每个点计算其格网坐标
    ``(i, j) = round((x-x0)/grid_res)``，坐标往返误差在容差内且落在
    ``[0, nx-1]×[0, ny-1]`` 的点才回填；其余点（缘石/非格网点）不计入
    采样源，也不计丢点（它们本不在被采样的路面足迹内）。

    Args:
        points: (N, 3) 点云。
        nx, ny: 路面合成网格形状。
        grid_res: 网格间距 (米)。
        x0, y0: 网格原点 (米)。

    Returns:
        (idx_grid, n_off_lattice)：``(nx, ny)`` int32 索引阵（-1 = 无点）
        与落在格网外的点数。

    Raises:
        ValueError: 网格过小（< 2×2）、无任何点落在格网上、或同一格网
            单元被多个点占用（输入不是单值表面）。
    """
    if nx < 2 or ny < 2:
        raise ValueError(
            f"line-profiler needs a surface grid of at least 2x2, got {(nx, ny)}"
        )
    n_points = int(points.shape[0])
    if n_points >= 2**31:
        raise ValueError(
            f"line-profiler lattice indexing is int32; got {n_points:,} points"
        )

    fi = (points[:, 0] - x0) / grid_res
    fj = (points[:, 1] - y0) / grid_res
    i_raw = np.rint(fi)
    j_raw = np.rint(fj)
    tol = _LATTICE_TOL_REL * grid_res + _LATTICE_TOL_ABS
    on_lattice = (
        np.isfinite(fi)
        & np.isfinite(fj)
        & (i_raw >= 0.0)
        & (i_raw <= nx - 1.0)
        & (j_raw >= 0.0)
        & (j_raw <= ny - 1.0)
        & (np.abs(fi - i_raw) * grid_res <= tol)
        & (np.abs(fj - j_raw) * grid_res <= tol)
    )
    idx = np.nonzero(on_lattice)[0]
    n_off = n_points - int(idx.shape[0])
    if idx.shape[0] == 0:
        raise ValueError(
            "no input point lies on the road lattice "
            f"({n_points:,} points, grid_res={grid_res:g} m, "
            f"origin=({x0:g}, {y0:g})) — line-profiler samples the "
            "synthesised road grid, so a cloud with no grid-aligned point "
            "cannot be profiled."
        )

    cell = (
        i_raw[idx].astype(np.int64) * np.int64(ny) + j_raw[idx].astype(np.int64)
    )
    if cell.shape[0] > 1:
        ordered = np.sort(cell)
        if bool(np.any(ordered[1:] == ordered[:-1])):
            raise ValueError(
                "multiple input points map to the same surface grid cell — "
                "line-profiler expects a single-valued grid surface (one "
                "point per lattice cell)."
            )

    flat = np.full(nx * ny, -1, dtype=np.int32)
    flat[cell] = idx.astype(np.int32)
    return flat.reshape(nx, ny), n_off


def _gather_axis0(values: np.ndarray, col: np.ndarray) -> np.ndarray:
    """按索引列取值，``col < 0``（格网空洞）处填 NaN。"""
    pos = np.maximum(col, 0)
    out = values[pos]
    missing = col < 0
    if bool(np.any(missing)):
        out = out.copy()
        out[missing] = np.nan
    return out


def _gather_labels(values: np.ndarray, col: np.ndarray) -> np.ndarray:
    """按索引列取标签，``col < 0``（格网空洞）处填 -1（未知）。"""
    pos = np.maximum(col, 0)
    return np.where(col >= 0, values[pos], -1)


def _corner_weights(tx: np.ndarray, ty: float) -> np.ndarray:
    """Four-corner weights, ordered left/a, right/a, left/b, right/b."""
    return np.stack(
        [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
         (1.0 - tx) * ty, tx * ty], axis=1
    )


def _corner_values(a: np.ndarray, b: np.ndarray, i0: np.ndarray) -> np.ndarray:
    return np.stack([a[i0], a[i0 + 1], b[i0], b[i0 + 1]], axis=1)


def _bilinear(
    z_a: np.ndarray,
    z_b: np.ndarray,
    i0: np.ndarray,
    tx: np.ndarray,
    ty: float,
) -> np.ndarray:
    """四角双线性高程插值；仅有限、正权重支撑参与重归一。

    零权重 NaN 不传播。最近源格网缺失的丢点策略由调用方单独应用，
    因而邻角缺失不会扩大空洞范围。无有效正权重支撑时返回 NaN。
    """
    values = _corner_values(z_a, z_b, i0)
    weights = _corner_weights(tx, ty)
    valid = np.isfinite(values) & (weights > 0.0)
    weights = np.where(valid, weights, 0.0)
    wsum = weights.sum(axis=1)
    num = np.sum(weights * np.where(valid, values, 0.0), axis=1)
    return np.where(wsum > 0.0, num / np.maximum(wsum, 1e-300), np.nan)


def _bilinear_vec(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    i0: np.ndarray,
    tx: np.ndarray,
    ty: float,
) -> np.ndarray:
    """四角法向先逐个单位化，再加权并单位化；不翻转输入方向。

    有限非零且权重为正的法向才参与。退化和回退至最近有效支撑，距离
    相同优先较大 x，再优先较大 y（与最近源格网的半格取上侧一致）。
    无有效支撑返回 NaN，由调用方只对可测样本报错。
    """
    values = _corner_values(vec_a, vec_b, i0)
    weights = _corner_weights(tx, ty)
    finite = np.isfinite(values).all(axis=2)
    clean = np.where(finite[:, :, None], values, 0.0)
    # Scale first so finite extreme magnitudes neither overflow nor underflow.
    scale = np.max(np.abs(clean), axis=2)
    valid = finite & (scale > 0.0) & (weights > 0.0)
    scaled = clean / np.where(scale > 0.0, scale, 1.0)[:, :, None]
    lengths = np.linalg.norm(scaled, axis=2)
    unit = scaled / np.where(lengths > 0.0, lengths, 1.0)[:, :, None]
    weights = np.where(valid, weights, 0.0)
    total = np.sum(weights[:, :, None] * unit, axis=1)
    total_length = np.linalg.norm(total, axis=1)
    good = total_length > 1e-12 * weights.sum(axis=1)
    out = np.full((len(tx), 3), np.nan)
    out[good] = total[good] / total_length[good, None]

    fallback = ~good & valid.any(axis=1)
    if bool(np.any(fallback)):
        distances = np.stack(
            [tx**2 + ty**2, (1.0-tx)**2 + ty**2,
             tx**2 + (1.0-ty)**2, (1.0-tx)**2 + (1.0-ty)**2], axis=1
        )
        # argmin's first-match rule gives deterministic ties in this order.
        order = np.array([3, 1, 2, 0])
        ranked = np.where(valid, distances, np.inf)[:, order]
        nearest = order[np.argmin(ranked, axis=1)]
        rows = np.flatnonzero(fallback)
        out[rows] = unit[rows, nearest[rows]]
    return out


def simulate_line_profiler(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    grid_res: float,
    origin_xy: tuple[float, float],
    cfg: LineProfilerConfig,
    rng: np.random.Generator,
    noise_enabled: bool = True,
) -> LineProfilerResult:
    """Simulate a line-laser profiler scan over a grid-ordered surface cloud.

    几何链：纵向步进 ``y_k = k·step_m``；每条剖面在相邻两行合成网格上
    做双线性插值得到沿 ``y = y_k`` 的地表剖面，再按 ``lateral_res_m``
    横向线性重采样。高程噪声按 Csencsics 2022 散斑参数化
    ``σ_z = σ_z0·(1+tan²θ_inc)/√b_rel`` 施加（θ_inc 为光平面公称入射角，
    b_rel 为逐剖面几何亮度代理 × Gamma 散斑相对中位数之比）；飞点按
    ``flyer_prob`` 逐点离群；低亮度点按归一化亮度反比概率丢点。

    输入云不保证是完整规则网格（松散 NaN 过滤删行、缘石并入追加异位
    点），模块按格网坐标反查重建表面格网；源格网空洞（松散移除的点）
    在该样本上产生丢点而不伪造高程。详见模块 docstring。

    Args:
        points: 网格规则点云 (N, 3)，行优先展平（index = i·ny + j，
            ``generate_road_surface`` 的 ravel 口径）。病害之后的云允许
            是完整网格的子序列（松散过滤）并在云尾附带非路面点（缘石）。
        normals: 单位法向量 (N, 3)。
        labels: 标签 (N,)（病害后）。
        grid_shape: (nx, ny) 合成网格形状。
        grid_res: 网格间距 (米)。
        origin_xy: 网格原点 (米)。
        cfg: 传感器配置（``config.LineProfilerConfig``，唯一真源）。
        rng: 随机流。

    Returns:
        LineProfilerResult。

    Raises:
        ValueError: 点云与网格不兼容（无格网点、格网单元重复占用、
            网格 < 2×2），或可测样本无有限非零的正权重法向支撑。
    """
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    x0, y0 = float(origin_xy[0]), float(origin_xy[1])
    width = (nx - 1) * grid_res
    length = (ny - 1) * grid_res

    idx_grid, _n_off_lattice = _rebuild_surface_lattice(
        points, nx, ny, grid_res, x0, y0
    )
    z_all = points[:, 2]
    n_all = normals
    lbl_all = labels

    # 源格网全空洞时的线激光头参考高程回退值（逐剖面 nanmax 失效时用）。
    finite_z = z_all[idx_grid[idx_grid >= 0]]
    finite_z = finite_z[np.isfinite(finite_z)]
    z_ref_fallback = (
        float(np.median(finite_z)) if finite_z.size else 0.0
    )

    # 横向采样位置（整条带共享）。
    n_lat = max(2, int(math.floor(width / cfg.lateral_res_m)) + 1)
    x_s = x0 + np.arange(n_lat) * cfg.lateral_res_m
    x_s = x_s[x_s <= x0 + width + 1e-12]
    n_lat = int(x_s.shape[0])
    fx = (x_s - x0) / grid_res
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    tx = (fx - i0).astype(np.float64)
    # 最近格网列的横向距离（标签传递阈值判定用）。
    dx_nn = np.minimum(tx, 1.0 - tx) * grid_res

    # 纵向剖面位置。
    n_prof = max(1, int(math.floor(length / cfg.step_m)) + 1)
    y_k = y0 + np.arange(n_prof) * cfg.step_m
    y_k = y_k[y_k <= y0 + length + 1e-12]
    n_prof = int(y_k.shape[0])

    tan_inc = math.tan(math.radians(cfg.laser_incidence_deg))
    ang_factor = 1.0 + tan_inc * tan_inc
    label_max_dist = 2.0 * cfg.lateral_res_m

    chunks_pts: list[np.ndarray] = []
    chunks_nrm: list[np.ndarray] = []
    chunks_lbl: list[np.ndarray] = []
    chunks_prof: list[np.ndarray] = []
    chunks_fly: list[np.ndarray] = []
    dropout_count = 0

    for k in range(n_prof):
        fy = (y_k[k] - y0) / grid_res
        j0 = min(max(int(math.floor(fy)), 0), ny - 2)
        ty = fy - j0
        dy_nn = min(ty, 1.0 - ty) * grid_res
        # 光平面×网格求交（规则网格 ⇒ 相邻两行线性插值）。列索引阵中的
        # -1（松散移除的格网单元）经 _gather_* 变为 NaN/-1；插值**只在该
        # 样本自身格网可用时**进行（权重重归一），因此单个空缺单元不会
        # 经邻位权重把相邻样本一起变 NaN。样本自身格网缺失由 hole_mask
        # 单独判定为丢点——不伪造高程。
        col_a = idx_grid[:, j0]
        col_b = idx_grid[:, j0 + 1]
        z_a = _gather_axis0(z_all, col_a)
        z_b = _gather_axis0(z_all, col_b)
        # 横向重采样（线性，权重在可用单元上重归一）。
        z_prof = _bilinear(z_a, z_b, i0, tx, ty)

        n_a = _gather_axis0(n_all, col_a)
        n_b = _gather_axis0(n_all, col_b)
        n_prof_row = _bilinear_vec(n_a, n_b, i0, tx, ty)

        # 标签：剖面内最近格网单元，距离阈值 2·lateral_res_m。
        i_near = np.where(tx < 0.5, i0, i0 + 1)
        j_near = j0 if ty < 0.5 else j0 + 1

        # 空洞掩码：源格网缺失（松散移除）⇒ 该横向样本不可测。判据只取
        # **该样本自身**的最近格网单元（横向 i_near、纵向 j_near）是否
        # 空缺，而不看邻位权重是否产生 NaN——后者会让一个空缺单元连带
        # 污染相邻样本，把局部稀疏放大约 2 倍；自身口径下每个空缺单元
        # 只损失它自己支撑的样本，与源点云稀疏一致（不夸大、不编造）。
        hole_mask = idx_grid[i_near, j_near] < 0
        finite_mask = ~hole_mask & np.isfinite(z_prof)
        invalid_normal = finite_mask & ~np.isfinite(n_prof_row).all(axis=1)
        if bool(np.any(invalid_normal)):
            first = int(np.flatnonzero(invalid_normal)[0])
            raise ValueError(
                "line-profiler measurable sample has no finite nonzero "
                "positive-weight normal support "
                f"(profile={k}, lateral_index={first})"
            )
        # 空洞点不进入输出或亮度中位数，但也必须保持其亮度代理有限。
        n_prof_row[~finite_mask] = np.array([0.0, 0.0, 1.0])

        lbl_prof = _gather_labels(lbl_all, idx_grid[i_near, j_near])
        nn_dist = np.sqrt(dx_nn * dx_nn + dy_nn * dy_nn)
        lbl_prof = np.where(nn_dist <= label_max_dist, lbl_prof, -1)

        # 逐剖面几何亮度代理（无标签/材料先验）：cos θ_laser / R²，
        # 激光头位于 (x_mid, y_k, z_head)，z_head 取该剖面最大高程 +
        # sensor_height_m（全空洞剖面回退到全场中位高程）。
        if bool(np.any(finite_mask)):
            z_head = float(np.max(z_prof[finite_mask])) + cfg.sensor_height_m
        else:
            z_head = z_ref_fallback + cfg.sensor_height_m
        head = np.array([x0 + 0.5 * width, y_k[k], z_head])
        # 亮度代理在**可测样本**上计算：空洞样本的 z 无定义，会让 R² /
        # cos_l 变 NaN，进而污染整条剖面的中位数归一（NaN 中位数 →
        # b_rel 全 NaN → p_drop 全 0.95，把局部稀疏放大成整条丢点）。
        # 这些样本最终被 hole_mask 丢弃，故先把它们的 z 置为剖面参考
        # 高程（仅用于亮度代理的有限性，不进入输出）。
        z_for_brightness = np.where(finite_mask, z_prof, z_head)
        d = head[None, :] - np.stack(
            [x_s, np.full(n_lat, y_k[k]), z_for_brightness], axis=1
        )
        R2 = np.sum(d * d, axis=1) + 1e-12
        d_hat = d / np.sqrt(R2)[:, None]
        cos_l = np.clip(np.sum(n_prof_row * d_hat, axis=1), 0.03, 1.0)
        brightness_geom = cos_l / R2
        # 散斑实现：Gamma(shape=k, scale=1/k)，均值 1、CV=1/√k。这是
        # cfg.speckle_shape 的唯一消费点，同时耦合进 σ_z 与低亮度丢点。
        if noise_enabled:
            speckle = rng.gamma(
                shape=cfg.speckle_shape, scale=1.0 / cfg.speckle_shape, size=n_lat
            )
        else:
            speckle = np.ones(n_lat, dtype=np.float64)
        brightness = brightness_geom * speckle
        # 中位数同样只在可测样本上统计（空洞样本的代理值不参与归一）。
        b_med = (
            float(np.median(brightness[finite_mask]))
            if bool(np.any(finite_mask))
            else float(np.median(brightness))
        )
        b_rel = brightness / max(b_med, 1e-12)

        # Csencsics 2022 散斑参数化高程噪声（θ_inc = 光平面公称入射角）。
        sigma_z = cfg.sigma_z0_m * ang_factor / np.sqrt(np.maximum(b_rel, 1e-6))
        z_meas = (
            z_prof + rng.normal(0.0, 1.0, n_lat) * sigma_z
            if noise_enabled
            else z_prof.copy()
        )

        # 飞点启发式（± 两侧等概，幅度 flyer_sigma_mult·σ_z）。
        fly = (rng.random(n_lat) < cfg.flyer_prob) if noise_enabled else np.zeros(n_lat, dtype=bool)
        if bool(np.any(fly)):
            sign = np.where(rng.random(n_lat) < 0.5, -1.0, 1.0)
            z_meas = z_meas.copy()
            z_meas[fly] += sign[fly] * cfg.flyer_sigma_mult * sigma_z[fly]

        # 丢点：低亮度（辐射耦合，与 simulate_lidar_noise 同型）+
        # 源格网空洞（松散移除 → 该样本无表面可测）。
        if noise_enabled:
            p_drop = np.clip(
                cfg.brightness_dropout_ref * b_med / np.maximum(brightness, 1e-12),
                0.0,
                0.95,
            )
            keep = (rng.random(n_lat) >= p_drop) & finite_mask
        else:
            keep = finite_mask.copy()
        dropout_count += int(np.count_nonzero(~keep))

        pts_k = np.stack(
            [x_s[keep], np.full(int(keep.sum()), y_k[k]), z_meas[keep]],
            axis=1,
        )
        chunks_pts.append(pts_k)
        chunks_nrm.append(n_prof_row[keep])
        chunks_lbl.append(lbl_prof[keep])
        chunks_prof.append(np.full(int(keep.sum()), k, dtype=np.int64))
        chunks_fly.append(fly[keep])

    if chunks_pts:
        points_out = np.vstack(chunks_pts)
        normals_out = np.vstack(chunks_nrm)
        labels_out = np.concatenate(chunks_lbl).astype(np.int64)
        prof_out = np.concatenate(chunks_prof)
        fly_out = np.concatenate(chunks_fly)
    else:
        points_out = np.zeros((0, 3))
        normals_out = np.zeros((0, 3))
        labels_out = np.zeros(0, dtype=np.int64)
        prof_out = np.zeros(0, dtype=np.int64)
        fly_out = np.zeros(0, dtype=bool)

    return LineProfilerResult(
        points=points_out,
        normals=normals_out,
        labels=labels_out,
        profile_index=prof_out,
        flyer_mask=fly_out,
        dropout_count=dropout_count,
        sampling_audit={
            "geometry_sampler": "bilinear_valid_positive_weight_corners",
            "source_spacing_m": float(grid_res),
            "source_axis_nyquist_cycles_m": 0.5 / float(grid_res),
            "output_spacing_m": {"x": float(cfg.lateral_res_m), "y": float(cfg.step_m)},
            "output_axis_nyquist_cycles_m": {"x": 0.5 / cfg.lateral_res_m, "y": 0.5 / cfg.step_m},
            "label_support": "nearest_source_grid_cell_at_xy; unknown beyond 2*lateral_res_m",
            "normal_semantics": "latent_surface_normal_interpolated_at_xy",
            "geometric_occlusion": "not_modelled; source holes remain missing",
            "beam_footprint": "point_sample; not_calibrated",
            "anti_alias_filter": "none; finer output cannot recover missing source bandwidth",
            "noise_enabled": bool(noise_enabled),
            "noise_parameters": "engineering_hypotheses; not calibrated from precision specifications",
            "real_sensor_validated": False,
        },
    )
