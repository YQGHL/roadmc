"""线激光剖面仪（line profiler）传感器仿真 —— ``--observation-model line-profiler``。

**设计方裁定（2026-10 修复批，取代设计稿两处字面表述）**：

1. **采样位置**：本模块在生成管线**步骤 8 位置**被调用（替换扫描式
   ``simulate_lidar_noise``），对**病害施加后**的最终几何采样——设计稿
   sensor_contract 早先写"在步骤 2.5 位置跑"与同一合同要求的
   ``labels`` 入参 / ``flyer_ratio`` 元数据自相矛盾（步骤 2.5 时标签恒为
   全零），按"对病害后几何采样"裁定并在此定稿。步骤 2.5 的旋转 LiDAR
   扫描重采样在本档位下被显式跳过（元数据记
   ``scan_override='skipped_under_line_profiler'``）。
2. **配置单一真源**：``LineProfilerConfig`` 唯一定义在
   ``roadmc.data.synthetic.config``（本模块 ``from .config import`` 引用，
   **不再自带同名 dataclass**）——消除双份定义的字段漂移风险。
3. **存储强度通道**：场景存储的 ``intensity`` 沿用管线统一辐射模型
   ``_compute_intensity``（单一位姿 σ_r/丢点/强度三用一致性的前提；
   生成器步骤 9 的最近邻传递会把强度带到本模块输出的点上）。本模块
   内部另算**逐剖面线激光头几何亮度代理**（cos θ_laser / R²，无标签、
   无材料先验）仅供散斑 σ_z 与低亮度丢点使用——设计稿"强度走线激光
   头位姿逐剖面重算"一句据此作废，避免与步骤 9 的 NN 覆盖冲突。

**几何链**（参照 laser-scanner-toolkit 的标定-重建链：光平面标定
（交比不变）→ 运动轴步进 → 射线-平面求交重建；本地镜像 README 见
调研材料 tk.md，其 MIT 许可声明本次未核验）：纵向步进
``y_k = k·step_m``；每条剖面取光平面（横向 x–竖向 z，自竖直倾角
``laser_incidence_deg``）与合成网格表面的求交——**规则网格上退化为
相邻两行网格高度场的线性插值**（ray-plane 链在网格面上的文档化简化），
再按 ``lateral_res_m`` 横向线性重采样得到一条剖面 (x_m, y_k, z)。

**高程噪声（Csencsics 2022 散斑参数化，DOI
10.1109/I2MTC48687.2022.9806610，已核；"Csencsics 2023" 期刊扩展本次
未检索到，需人工核对）**：

.. math::

    \\sigma_z(\\theta, b) = \\sigma_{z0}\\,
    \\frac{1 + \\tan^2\\theta_{\\mathrm{inc}}}{\\sqrt{b_{\\mathrm{rel}}}}

其中 ``θ_inc = laser_incidence_deg`` 取**光平面公称入射角**（设计稿
"入射角 30° 为工程设定"的字面口径：被倾斜的是光平面本身，故平坦面上
沿整条剖面线的入射角恒等于该公称值，不随沿线位置变化），``b_rel`` 为
逐剖面几何亮度代理（``cos θ_laser / R²``，线激光头位姿见下）乘
**Gamma(shape=speckle_shape, scale=1/shape) 散斑实现**后相对该剖面中
位数的比值——Gamma 因子均值 1、变异系数 ``1/√speckle_shape``，是
``LineProfilerConfig.speckle_shape`` 唯一的消费点（早期版本把它申报进
元数据却完全不使用，属配置静默失效，本版修正）。σ_z0 = 0.25 mm 与
横向分辨率 1 mm 为 LSP 实测规格（DOI
10.1109/ACCESS.2019.2956758——**导入而非自校准**）。

**飞点启发式**：逐点以 ``flyer_prob`` 概率加 ±flyer_sigma_mult·σ_z 的
离群位移（保留在输出云中，``flyer_mask`` 记录；± 两侧等概，"外向"按
"偏离表面"理解，不限定单一符号）；低亮度点按
``brightness_dropout_ref`` 归一化亮度反比概率丢点（与
``simulate_lidar_noise`` 的辐射耦合丢点同型）。

**输入云契约的放松（必要偏差，见模块注释）**：设计稿写"网格规则点云
(N, 3)，行优先展平"，但钩子位于病害之后——松散（raveling）的 NaN 过滤
会按点删行、缘石（curb）并入会在云尾追加非路面点，两者都使严格
reshape 契约不可满足（实测 5 mm 网格、默认病害配置下约 8% 场景直接
抛 ``ValueError``）。本版改为**按格网坐标反查重建表面格网**：点云中
落在路面格网 ``(i, j)`` 上的点按 ``round((x-x0)/grid_res)`` 回填到
``(nx, ny)`` 索引阵，格网外的点（缘石）不参与采样，**空格网单元**
（松散移除的点）在该横向/纵向样本上产生丢点而非伪造高程——这与管线
其余部分对松散的口径一致（LiDAR 档同样让松散点从输出中消失），也避免
在信息缺失区编造几何。缘石条带因此在本档位下不被采样（位于路面足迹
之外），这是单块网格路径的既定限制，已在此显式记录。

特征契约：本模块不产出任何标签派生通道；下游 feats 只由最终点与强度
计算（features.py 口径）。返回的 ``labels`` 仅为接口完整性保留——生成
器步骤 9 的最近邻传递是最终生效口径。
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
        normals: 采样点法向量 (M, 3)（同一网格插值口径，已重归一；
            生成器步骤 9 仍会按最近邻传递覆盖，此处为接口完整性保留）。
        labels: 采样点标签 (M,)（剖面内最近格网标签，距离阈值
            ``2·lateral_res_m`` 之外或无格网标签时为 -1 = 未知；生成器
            步骤 9 的最近邻传递为最终生效口径）。
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


def _bilinear(
    z_a: np.ndarray,
    z_b: np.ndarray,
    i0: np.ndarray,
    tx: np.ndarray,
    ty: float,
) -> np.ndarray:
    """相邻两行 × 相邻两列的双线性高程插值，权重在可用单元上重归一。

    ``z_a`` / ``z_b`` 是相邻两条整列（长度 nx），空洞单元为 NaN。对每个
    横向样本参与插值的四个单元中，只有**自身支持该样本的单元**（按其
    i_near 列）缺失时才产生 NaN；邻位缺失时其权重被重新分配给可用邻
    位，因此单个松散空洞不会连带污染相邻横向样本。纵向权重 ``ty``
    在自身单元缺失时同样重归一（全空 → NaN，由调用方按空洞丢点）。

    Args:
        z_a: 近行整列 (nx,)，值为 NaN 处对该列所有样本不可用。
        z_b: 远行整列 (nx,)。
        i0: 每样本左列索引 (n_lat,)。
        tx: 每样专栏内比例 (n_lat,)。
        ty: 行间比例 (0..1)。

    Returns:
        (n_lat,) 插值高程，自身无可用支撑时为 NaN。
    """
    # 自身列（i_near）：i0 或 i0+1 中较近者。
    use_left = tx < 0.5
    i_self = np.where(use_left, i0, i0 + 1)
    z_self_row_a = z_a[i_self]
    z_self_row_b = z_b[i_self]
    ok_a = np.isfinite(z_self_row_a)
    ok_b = np.isfinite(z_self_row_b)

    wa = (1.0 - ty) * ok_a.astype(np.float64)
    wb = ty * ok_b.astype(np.float64)
    wsum = wa + wb
    num = wa * np.where(ok_a, z_self_row_a, 0.0) + wb * np.where(
        ok_b, z_self_row_b, 0.0
    )
    return np.where(wsum > 0.0, num / np.maximum(wsum, 1e-300), np.nan)


def _bilinear_vec(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    i0: np.ndarray,
    tx: np.ndarray,
    ty: float,
) -> np.ndarray:
    """:func:`_bilinear` 的向量版本（法向量 (N, 3)），同一重归一口径。

    任一分量为 NaN 即视为该单元对该样本不可用（法向量要求三个分量同时
    有效），权重在可用行间重归一。
    """
    use_left = tx < 0.5
    i_self = np.where(use_left, i0, i0 + 1)
    v_a = vec_a[i_self]
    v_b = vec_b[i_self]
    ok_a = np.isfinite(v_a).all(axis=1)
    ok_b = np.isfinite(v_b).all(axis=1)

    wa = (1.0 - ty) * ok_a.astype(np.float64)
    wb = ty * ok_b.astype(np.float64)
    wsum = wa + wb
    num = wa[:, None] * np.where(ok_a[:, None], v_a, 0.0) + wb[:, None] * (
        np.where(ok_b[:, None], v_b, 0.0)
    )
    return np.where(wsum[:, None] > 0.0, num / np.maximum(wsum, 1e-300)[:, None], np.nan)


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
            网格 < 2×2）。
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
        # 法向重归一。
        nlen = np.linalg.norm(n_prof_row, axis=1, keepdims=True)
        n_prof_row = n_prof_row / np.maximum(nlen, 1e-12)

        # 标签：剖面内最近格网单元，距离阈值 2·lateral_res_m。
        i_near = np.where(tx < 0.5, i0, i0 + 1)
        j_near = j0 if ty < 0.5 else j0 + 1

        # 空洞掩码：源格网缺失（松散移除）⇒ 该横向样本不可测。判据只取
        # **该样本自身**的最近格网单元（横向 i_near、纵向 j_near）是否
        # 空缺，而不看邻位权重是否产生 NaN——后者会让一个空缺单元连带
        # 污染相邻样本，把局部稀疏放大约 2 倍；自身口径下每个空缺单元
        # 只损失它自己支撑的样本，与源点云稀疏一致（不夸大、不编造）。
        hole_mask = idx_grid[i_near, j_near] < 0

        lbl_prof = _gather_labels(lbl_all, idx_grid[i_near, j_near])
        nn_dist = np.sqrt(dx_nn * dx_nn + dy_nn * dy_nn)
        lbl_prof = np.where(nn_dist <= label_max_dist, lbl_prof, -1)

        # 逐剖面几何亮度代理（无标签/材料先验）：cos θ_laser / R²，
        # 激光头位于 (x_mid, y_k, z_head)，z_head 取该剖面最大高程 +
        # sensor_height_m（全空洞剖面回退到全场中位高程）。
        finite_mask = ~hole_mask
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
        speckle = rng.gamma(
            shape=cfg.speckle_shape, scale=1.0 / cfg.speckle_shape, size=n_lat
        )
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
        z_meas = z_prof + rng.normal(0.0, 1.0, n_lat) * sigma_z

        # 飞点启发式（± 两侧等概，幅度 flyer_sigma_mult·σ_z）。
        fly = rng.random(n_lat) < cfg.flyer_prob
        if bool(np.any(fly)):
            sign = np.where(rng.random(n_lat) < 0.5, -1.0, 1.0)
            z_meas = z_meas.copy()
            z_meas[fly] += sign[fly] * cfg.flyer_sigma_mult * sigma_z[fly]

        # 丢点：低亮度（辐射耦合，与 simulate_lidar_noise 同型）+
        # 源格网空洞（松散移除 → 该样本无表面可测）。
        p_drop = np.clip(
            cfg.brightness_dropout_ref * b_med / np.maximum(brightness, 1e-12),
            0.0,
            0.95,
        )
        keep = (rng.random(n_lat) >= p_drop) & finite_mask
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
    )
