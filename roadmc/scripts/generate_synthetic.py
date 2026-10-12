"""Batch generator for synthetic road point cloud datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roadmc.data.synthetic.config import (
    DEFAULT_MAX_SURFACE_MEMORY_MIB,
    DEFAULT_MAX_SURFACE_POINTS,
    LABEL_MAP,
    NUM_CLASSES,
    DiseaseConfig,
    GeneratorConfig,
    RoadSurfaceConfig,
)

_WORKER_DATASET = None
_WORKER_SPLIT_DIR = None


def _init_worker(config: GeneratorConfig, split_dir_str: str) -> None:
    global _WORKER_DATASET, _WORKER_SPLIT_DIR
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    _WORKER_DATASET = SyntheticRoadDataset(config=config, dataset_size=0)
    _WORKER_SPLIT_DIR = Path(split_dir_str)


def _save_one_scene(dataset, split_dir: Path, scene_id: int) -> dict:
    result = {"scene_id": scene_id, "ok": False, "npoints": 0, "labels": []}
    try:
        scene = dataset.generate_scene(scene_id)
        pts = scene["points"]
        lbls = scene["labels"]
        feats = scene["feats"]
        normals = scene["normals"]
        pavement_type = scene["pavement_type"]

        valid = ~np.any(np.isnan(pts), axis=1)
        pts = pts[valid]
        lbls = lbls[valid]
        feats = feats[valid]
        normals = normals[valid]
        if len(pts) == 0:
            return result

        np.savez_compressed(
            split_dir / f"scene_{scene_id:04d}.npz",
            points=pts.astype(np.float32),
            labels=lbls.astype(np.int32),
            feats=feats.astype(np.float32),
            normals=normals.astype(np.float32),
            pavement_type=pavement_type,
            roughness_class=scene["roughness_class"],
            scene_id=scene_id,
            feature_schema=scene["feature_schema"],
            feature_names=scene["feature_names"],
            feature_k_neighbors=scene["feature_k_neighbors"],
            coordinate_center=scene["coordinate_center"],
            coordinate_scale=scene["coordinate_scale"],
            coordinates_normalized=scene["coordinates_normalized"],
            resolution_metadata_json=np.asarray(
                json.dumps(
                    scene["resolution_metadata"],
                    ensure_ascii=True,
                    sort_keys=True,
                )
            ),
            resolution_contract=scene["resolution_contract"],
            surface_grid_spacing_m=scene["surface_grid_spacing_m"],
            surface_grid_shape=scene["surface_grid_shape"],
            surface_grid_point_count=scene["surface_grid_point_count"],
            sensor_output_point_count=scene["sensor_output_point_count"],
            observation_model=scene["observation_model"],
            segment_count=scene["segment_count"],
            segment_length_m=scene["segment_length_m"],
            segment_overlap_m=scene["segment_overlap_m"],
            model_target_points=scene["model_target_points"],
        )
        result["ok"] = True
        result["npoints"] = int(len(pts))
        result["labels"] = list(np.unique(lbls).astype(int))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Scene {scene_id} failed: {exc}", stacklevel=2)
    return result


def _worker_save(scene_id: int) -> dict:
    return _save_one_scene(_WORKER_DATASET, _WORKER_SPLIT_DIR, scene_id)


def generate_dataset(
    count: int,
    config: GeneratorConfig,
    split: str,
    output_dir: Path,
    sampling_mode: str = "bernoulli",
    num_workers: int = 1,
) -> dict:
    from roadmc.data.synthetic.generator import SyntheticRoadDataset

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    scene_config = config
    if split == "val" and config.seed is not None:
        scene_config = replace(config, seed=config.seed + 100000)
    scene_config = replace(
        scene_config,
        disease=replace(scene_config.disease, sampling_mode=sampling_mode),
    )

    class_counts = dict.fromkeys(range(NUM_CLASSES), 0)
    point_counts: list[int] = []
    failed = 0
    scene_ids = list(range(count))

    if num_workers <= 1:
        dataset = SyntheticRoadDataset(config=scene_config, dataset_size=count)
        try:
            from tqdm import tqdm

            iterator = tqdm(scene_ids, desc=f"生成 {split} 集")
        except ImportError:
            print(f"生成 {split} 集 ({count} 个场景)...")
            iterator = scene_ids

        for sid in iterator:
            res = _save_one_scene(dataset, split_dir, sid)
            if res["ok"]:
                point_counts.append(res["npoints"])
                for lbl in res["labels"]:
                    if lbl in class_counts:
                        class_counts[lbl] += 1
            else:
                failed += 1
    else:
        print(f"并行生成 {split} 集 ({count} 个场景, {num_workers} workers)...")
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(scene_config, str(split_dir))) as pool:
            try:
                from tqdm import tqdm

                iterator = tqdm(pool.imap_unordered(_worker_save, scene_ids), total=count, desc=f"生成 {split} 集")
            except ImportError:
                iterator = pool.imap_unordered(_worker_save, scene_ids)

            for res in iterator:
                if res["ok"]:
                    point_counts.append(res["npoints"])
                    for lbl in res["labels"]:
                        if lbl in class_counts:
                            class_counts[lbl] += 1
                else:
                    failed += 1

    return {
        "split": split,
        "requested": count,
        "generated": count - failed,
        "failed": failed,
        "class_distribution": class_counts,
        "point_stats": {
            "total": int(sum(point_counts)) if point_counts else 0,
            "mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "min": int(np.min(point_counts)) if point_counts else 0,
            "max": int(np.max(point_counts)) if point_counts else 0,
        },
    }


def verify_class_distribution(output_dir: Path, split: str) -> dict[int, int]:
    split_dir = output_dir / split
    if not split_dir.exists():
        print(f"[WARN] {split} 目录不存在: {split_dir}")
        return {}

    class_counts = dict.fromkeys(range(NUM_CLASSES), 0)
    total_scenes = 0

    print(f"\n=== {split.upper()} 类别分布 ===")
    print(f"{'标签':<6} {'类型':<12} {'严重度':<6} {'场景数':<8} {'占比':<8}")
    print("-" * 50)

    for npz_file in sorted(split_dir.glob("scene_*.npz")):
        try:
            data = np.load(npz_file, allow_pickle=True)
            labels = data["labels"]
            for lbl in np.unique(labels):
                if int(lbl) in class_counts:
                    class_counts[int(lbl)] += 1
            total_scenes += 1
        except Exception:
            continue

    for label_id in range(NUM_CLASSES):
        info = LABEL_MAP.get(label_id, {})
        dtype = info.get("type", "未知")
        severity = info.get("severity", "-")
        count = class_counts[label_id]
        pct = (count / total_scenes * 100) if total_scenes else 0
        print(f"{label_id:<6} {dtype:<12} {severity:<6} {count:<8} {pct:.1f}%")

    print(f"\n总计: {total_scenes} 个场景")
    return class_counts


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI while retaining all legacy option spellings."""

    parser = argparse.ArgumentParser(description="RoadMC 合成点云数据集批量生成")
    parser.add_argument("--train-count", type=int, default=2000, help="训练集场景数 (默认 2000)")
    parser.add_argument("--val-count", type=int, default=500, help="验证集场景数 (默认 500)")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--surface-grid-spacing",
        "--grid-res",
        dest="surface_grid_spacing",
        type=float,
        default=0.005,
        help="物理表面高度场网格间距 (米)，默认 0.005 与 GeneratorConfig 一致；--grid-res 为兼容旧名",
    )
    parser.add_argument(
        "--sensor-output-points",
        "--num-points",
        dest="sensor_output_points",
        type=int,
        default=65536,
        help="存盘传感器点云目标点数；--num-points 为兼容旧名",
    )
    parser.add_argument(
        "--sensor-output-density",
        "--target-density",
        dest="sensor_output_density",
        type=float,
        default=None,
        help="存盘传感器点密度 (点/平方米)，大于 0 时优先于固定点数",
    )
    parser.add_argument(
        "--model-target-points",
        type=int,
        default=None,
        help="下游模型目标点数，仅写入契约；生成器不会执行该采样",
    )
    parser.add_argument(
        "--max-surface-points",
        type=int,
        default=DEFAULT_MAX_SURFACE_POINTS,
        help="单场景传感器采样前允许的最大表面网格点数",
    )
    parser.add_argument(
        "--max-surface-memory-mib",
        type=float,
        default=DEFAULT_MAX_SURFACE_MEMORY_MIB,
        help="单 worker 表面合成估算峰值内存上限 (MiB)",
    )
    parser.add_argument(
        "--max-parallel-memory-mib",
        type=float,
        default=8192.0,
        help="所有生成 worker 的表面合成估算内存总上限 (MiB)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=7.0,
        help="路面宽度 (米)，默认 7.0（双车道口径）；硬下限 1.0 m",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=5.0,
        help="路面长度（里程向）(米)，默认 5.0",
    )
    parser.add_argument(
        "--observation-model",
        choices=["on", "off", "line-profiler"],
        default="off",
        help=(
            "观测模型体制：off（默认）直接输出全部表面点，跳过扫描抽稀与"
            "噪声丢点；on 复现完整传感器观测管线（与历史版本逐位一致）；"
            "line-profiler 为线激光剖面仪观测模型（传感器模块未就绪前"
            "报'未安装'类清晰错误）"
        ),
    )
    parser.add_argument("--pavement", type=str, default="mixed", choices=["asphalt", "concrete", "mixed"], help="路面类型")
    parser.add_argument("--roughness", type=str, default="B", choices=["A", "B", "C", "D", "E"], help="ISO 8608 粗糙度等级")
    parser.add_argument("--max-diseases", type=int, default=3, help="每场景最多病害数（0 = 仅无病害背景场景）")
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="（已弃用）--disease-sampling bernoulli 的旧别名：伯努利语义现已是默认；保留仅为兼容现有脚本",
    )
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    _add_realism_flags(parser)
    return parser


def _add_realism_flags(parser: argparse.ArgumentParser) -> None:
    """Register the 2026-10 generator behaviour-fix rollback flags.

    默认值选择修正后的行为；每个 flag 恢复一项具体的旧行为，供回归
    门按 rollback_recipe 逐项复现旧输出（默认全部不传 = 全新行为）。
    """
    g = parser.add_argument_group("生成器行为修正回退（2026-10）")
    g.add_argument(
        "--macro-spectrum",
        choices=["iso-marginal", "legacy", "two-segment"],
        default="two-segment",
        help=(
            "宏观谱权重口径：two-segment（默认，ISO 段 + VT 2015 两区间幂律"
            "纹理段，H 数值导入而非自校准）| iso-marginal（单段纹理谱 ISO "
            "边际重构，回退档）| legacy（历史点采样权重 + std 重定标）"
        ),
    )
    g.add_argument(
        "--disease-sampling",
        choices=["bernoulli", "legacy"],
        default=None,
        help="病害选择口径：bernoulli（默认，disease_probs 即出现概率，允许 0 病害场景）| legacy（概率归一化加权恒定计数）",
    )
    g.add_argument(
        "--pothole-beta-legacy",
        action="store_true",
        help="坑槽 β 恢复历史口径（轻=2.0 字面量、重~U(3,5)，beta_range 同步复位）",
    )
    g.add_argument(
        "--pothole-micropits-legacy",
        action="store_true",
        help="微坑恢复历史口径（Poisson(1) 且仅在 edge_quality<1.0 时生成）",
    )
    g.add_argument(
        "--pothole-aspect-legacy",
        action="store_true",
        help="取消坑槽 d/R 纵横比先验（恢复 R、d 独立采样）",
    )
    g.add_argument(
        "--pothole-spall-band",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="坑缘剥落环半径带 (lo·R, hi·R)，默认 0.8 1.0；历史口径传 0.8 1.2",
    )
    g.add_argument(
        "--agc-mode",
        choices=["fixed", "scene-p99"],
        default="fixed",
        help="强度 AGC 口径：fixed（默认，固定绝对标定）| scene-p99（历史逐场景 99 分位归一）",
    )
    g.add_argument(
        "--rutting-modulation-legacy",
        action="store_true",
        help="取消车辙名义深度 1/(1+ε) 归一化（历史口径）",
    )
    g.add_argument(
        "--rutting-center-legacy",
        action="store_true",
        help="车辙中心恢复恒位于道路中线（不走轮迹带先验）",
    )
    g.add_argument(
        "--crack-perturb-legacy",
        action="store_true",
        help="恢复裂缝走向扰动的 %%1 周期化环绕（历史口径）",
    )
    g.add_argument(
        "--crack-width-trunc",
        choices=["resample", "clip"],
        default="resample",
        help="缝宽判据侧截断：resample（默认，拒绝重采）| clip（历史钳制到边界）",
    )
    g.add_argument(
        "--areacrack-lambda-legacy",
        action="store_true",
        help="恢复面状裂缝槽形 λ=w/2（历史口径）",
    )
    g.add_argument(
        "--areacrack-label-legacy",
        action="store_true",
        help="面状裂缝标签恢复直接采信 severity 字符串（历史口径）",
    )
    g.add_argument(
        "--concrete-legacy",
        action="store_true",
        help="一次性恢复水泥病害四项旧行为（板内裂缝/错台全条带/边角剥落无界指数/破碎板脊线累加）",
    )
    g.add_argument(
        "--depression-center-legacy",
        action="store_true",
        help="沉陷中心恢复固定内缩 0.5 m（历史口径）",
    )
    g.add_argument(
        "--depression-label-legacy",
        action="store_true",
        help="沉陷标签恢复 depth·1%% 隐式外扩（历史口径）",
    )
    g.add_argument(
        "--bleeding-geometry-legacy",
        action="store_true",
        help="泛油恢复恒定 40%% 面积、贯穿全路长条带（历史口径）",
    )
    g.add_argument(
        "--lambert-abs-legacy",
        action="store_true",
        help="恢复 cosθ 取 |n·(−d̂)| 的历史口径（背向面按朝向计亮）",
    )
    g.add_argument(
        "--albedo-clip-hi",
        type=float,
        default=0.98,
        help="反照率截断上限（默认 0.98；历史口径传 0.95）",
    )
    g.add_argument(
        "--micro-texture-grid-legacy",
        action="store_true",
        help="恢复纹理谱 fftfreq(d=extent/gx) 历史频率网格口径",
    )
    g.add_argument(
        "--het-field-1d",
        action="store_true",
        help="纹理 het 调制恢复沿里程 y 一维广播（历史口径；默认二维调制场）",
    )
    g.add_argument(
        "--het-corr-len-m",
        type=float,
        default=None,
        help=(
            "纹理 het 调制场高斯相关长度 (m)。默认 0.01（不传时用库内"
            "默认）；历史口径传 2.0。与 --texture-het-corr 为同一字段"
            "的别名（类预算脚本既有名），显式传任一者即覆盖"
        ),
    )
    g.add_argument(
        "--texture-het-corr",
        type=float,
        default=None,
        help=(
            "--het-corr-len-m 的别名（类预算脚本历史名）；两者都显式"
            "传且值不同时报错"
        ),
    )

    g = parser.add_argument_group("two-segment 谱形（2026-10 结构批）")
    g.add_argument(
        "--texture-h-macro",
        type=float,
        default=0.34,
        help="two-segment 宏观纹理段 Hurst 指数（导入而非自校准，需人工核对 VT 原页）",
    )
    g.add_argument(
        "--texture-h-micro",
        type=float,
        default=0.75,
        help="two-segment 微观纹理段 Hurst 指数（导入而非自校准，需人工核对 VT 原页）",
    )
    g.add_argument(
        "--texture-break-wavelength",
        type=float,
        default=0.010,
        help="two-segment 断裂波长 λ_b (m)；VT 2015 Table 4-2 实测 0.50–2.30 cm（导入）",
    )
    g.add_argument(
        "--mpd-rms-ratio",
        type=float,
        default=0.81,
        help=(
            "MPD/纹理带 RMS 比值标定（默认 0.81 = two-segment 谱形下"
            "的重新标定值；历史单段谱形口径传 0.52）"
        ),
    )

    g = parser.add_argument_group("AM 慢变包络（默认关闭）")
    g.add_argument(
        "--am-envelope",
        choices=["on", "off"],
        default="off",
        help="AM 慢变包络开关（默认 off = 旧行为逐位不变）",
    )
    g.add_argument(
        "--am-wcut",
        type=float,
        default=2.83,
        help="AM 快/慢波分界频率 (cycle/m)，默认 2.83 = ISO 带边界",
    )
    g.add_argument(
        "--am-target-skew",
        type=float,
        default=0.0,
        help="AM CDF 变换目标偏度（导入而非自校准）",
    )
    g.add_argument(
        "--am-target-kurt",
        type=float,
        default=4.5,
        help="AM CDF 变换目标峰度（导入而非自校准）",
    )

    g = parser.add_argument_group("过渡带剖面（默认值 = 旧行为）")
    g.add_argument(
        "--crack-profile-exponent",
        type=float,
        default=2.0,
        help="缝深剖面指数 p（2.0 = 现行高斯槽；p→1 近线性 V 形）",
    )
    g.add_argument(
        "--crack-edge-band-width",
        type=float,
        default=0.0,
        help="缝缘崩碎过渡带宽度 (m)，0 = 关闭（现行行为）；工程假设",
    )
    g.add_argument(
        "--pothole-spall-profile",
        choices=["uniform", "linear-decay"],
        default="uniform",
        help="坑缘剥落深度剖面：uniform（默认，现行行为）| linear-decay（工程假设）",
    )
    g.add_argument(
        "--patch-edge-profile",
        choices=["smoothstep", "step-fillet"],
        default="smoothstep",
        help="修补边缘剖面：smoothstep（默认，现行行为）| step-fillet（错台实测口径，导入）",
    )
    g.add_argument(
        "--fault-fillet-m",
        type=float,
        default=0.0,
        help="水泥错台阶跃平滑宽度 (m)，0 = 现行纯阶跃",
    )

    g = parser.add_argument_group("基层条件 / 病害因果链（默认关闭）")
    g.add_argument(
        "--causality",
        choices=["on", "off"],
        default="off",
        help="病害因果链开关（SNP/龄期/YE4 抽取 + ICA 硬规则 + TPM 调制）；off（默认）= 旧行为",
    )
    g.add_argument(
        "--snp-range",
        nargs=2,
        type=float,
        default=(3.0, 5.5),
        metavar=("LO", "HI"),
        help="平均结构数 SNP 抽样区间（--causality on 时生效）",
    )
    g.add_argument(
        "--age-range",
        nargs=2,
        type=float,
        default=(0.0, 15.0),
        metavar=("LO", "HI"),
        help="场景龄期 (年) 抽样区间（--causality on 时生效）",
    )
    g.add_argument(
        "--tpm-path",
        type=str,
        default=None,
        help="Markov 状态转移概率矩阵 JSON 路径（可选；enable 时 disease_probs 被 π 调制）",
    )


def _resolve_disease_sampling(args: argparse.Namespace) -> str:
    """Resolve --disease-sampling / deprecated --no-stratified into one mode."""
    if args.no_stratified:
        if args.disease_sampling is not None and args.disease_sampling != "bernoulli":
            raise ValueError(
                "--no-stratified (deprecated alias for --disease-sampling "
                "bernoulli) conflicts with --disease-sampling "
                f"{args.disease_sampling!r}"
            )
        warnings.warn(
            "--no-stratified is deprecated: it maps to --disease-sampling "
            "bernoulli, which is now the default. Use --disease-sampling "
            "legacy to restore the weighted constant-count behaviour.",
            stacklevel=2,
        )
        return "bernoulli"
    return args.disease_sampling or "bernoulli"


def _resolve_het_corr_len_m(args: argparse.Namespace) -> float | None:
    """归并 --het-corr-len-m 与 --texture-het-corr（同一字段的别名）.

    两者都为 None → 返回 None（用库内默认 0.01，不做覆盖）；只传其
    一 → 用传入值；都传且值不同 → ValueError（避免静默丢一个 flag
    的值——历史版本里类预算脚本的 --texture-het-corr 会被
    --het-corr-len-m 的默认值无声覆盖）。
    """
    a, b = args.het_corr_len_m, args.texture_het_corr
    if a is None and b is None:
        return None
    if a is not None and b is not None and not np.isclose(a, b):
        raise ValueError(
            f"--het-corr-len-m ({a:g}) and --texture-het-corr ({b:g}) set "
            "the same field (micro_texture.het_corr_len_m) to different "
            "values; pass only one of them."
        )
    return a if a is not None else b


def _apply_realism_overrides(
    args: argparse.Namespace,
    config: GeneratorConfig,
) -> GeneratorConfig:
    """Translate the realism rollback flags into GeneratorConfig fields."""
    config = replace(
        config,
        pothole=replace(
            config.pothole,
            beta_mode="legacy" if args.pothole_beta_legacy else config.pothole.beta_mode,
            beta_range=(3.0, 5.0) if args.pothole_beta_legacy else config.pothole.beta_range,
            micropit_mode=(
                "legacy" if args.pothole_micropits_legacy else config.pothole.micropit_mode
            ),
            aspect_ratio_range=(
                None if args.pothole_aspect_legacy else config.pothole.aspect_ratio_range
            ),
            spall_band=(
                tuple(args.pothole_spall_band)
                if args.pothole_spall_band is not None
                else config.pothole.spall_band
            ),
            spall_profile=args.pothole_spall_profile,
        ),
        rutting=replace(
            config.rutting,
            peak_normalize=not args.rutting_modulation_legacy,
            center_mode=(
                "fixed" if args.rutting_center_legacy else config.rutting.center_mode
            ),
        ),
        crack=replace(
            config.crack,
            perturb_mode=(
                "legacy" if args.crack_perturb_legacy else config.crack.perturb_mode
            ),
            width_truncation=args.crack_width_trunc,
            area_lambda_mode=(
                "half" if args.areacrack_lambda_legacy else config.crack.area_lambda_mode
            ),
            area_label_mode=(
                "legacy" if args.areacrack_label_legacy else config.crack.area_label_mode
            ),
            profile_exponent=args.crack_profile_exponent,
            edge_band_width_m=args.crack_edge_band_width,
        ),
        concrete_damage=(
            replace(
                config.concrete_damage,
                crack_in_slab=False,
                fault_scope="legacy-strip",
                edge_spall_mode="legacy",
                shatter_mode="legacy",
            )
            if args.concrete_legacy
            else replace(
                config.concrete_damage,
                fault_fillet_m=args.fault_fillet_m,
            )
        ),
        patching=replace(
            config.patching,
            edge_profile=args.patch_edge_profile,
        ),
        depression=replace(
            config.depression,
            center_inset_mode=(
                "legacy" if args.depression_center_legacy else config.depression.center_inset_mode
            ),
            label_radius_mode=(
                "legacy-1pct"
                if args.depression_label_legacy
                else config.depression.label_radius_mode
            ),
        ),
        bleeding=replace(
            config.bleeding,
            geometry_mode=(
                "legacy-strip" if args.bleeding_geometry_legacy else config.bleeding.geometry_mode
            ),
        ),
        radiometric=replace(
            config.radiometric,
            agc_mode=args.agc_mode,
            cos_abs=True if args.lambert_abs_legacy else config.radiometric.cos_abs,
            albedo_clip_hi=args.albedo_clip_hi,
        ),
        micro_texture=replace(
            config.micro_texture,
            grid_convention=(
                "legacy"
                if args.micro_texture_grid_legacy
                else config.micro_texture.grid_convention
            ),
            het_field="1d" if args.het_field_1d else config.micro_texture.het_field,
            het_corr_len_m=(
                _resolve_het_corr_len_m(args)
                if _resolve_het_corr_len_m(args) is not None
                else config.micro_texture.het_corr_len_m
            ),
            h_macro=args.texture_h_macro,
            h_micro=args.texture_h_micro,
            break_wavelength_m=args.texture_break_wavelength,
            mpd_rms_ratio=args.mpd_rms_ratio,
        ),
        am_envelope=replace(
            config.am_envelope,
            enable=args.am_envelope == "on",
            wcut_cycles_per_m=args.am_wcut,
            target_skew=args.am_target_skew,
            target_kurt=args.am_target_kurt,
        ),
        base_condition=replace(
            config.base_condition,
            enable=args.causality == "on",
            snp_range=tuple(args.snp_range),
            age_years_range=tuple(args.age_range),
            tpm_path=args.tpm_path,
        ),
    )
    return config


def _config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    """Translate CLI names into the backward-compatible config fields."""

    config = GeneratorConfig(
        road=RoadSurfaceConfig(
            width=args.width,
            length=args.length,
            grid_res=args.surface_grid_spacing,
            roughness_class=args.roughness,
            pavement_type=args.pavement,
            macro_spectrum_mode=args.macro_spectrum,
        ),
        disease=DiseaseConfig(
            max_diseases_per_scene=args.max_diseases,
            use_stratified=not args.no_stratified,
            sampling_mode=_resolve_disease_sampling(args),
        ),
        seed=args.seed,
        num_points=args.sensor_output_points,
        target_density=args.sensor_output_density,
        model_target_points=args.model_target_points,
        max_surface_points=args.max_surface_points,
        max_surface_memory_mib=args.max_surface_memory_mib,
        # CLI 默认 off（冻结决策）；GeneratorConfig 的库内默认仍是 "on"。
        observation_model=args.observation_model,
    )
    return _apply_realism_overrides(args, config)


def _validate_request(
    args: argparse.Namespace,
    config: GeneratorConfig,
) -> tuple[int, float]:
    """Validate counts and aggregate worker memory before creating outputs."""

    if args.train_count < 0 or args.val_count < 0:
        raise ValueError("--train-count and --val-count must be >= 0")
    if args.max_diseases < 0:
        raise ValueError("--max-diseases must be >= 0 (0 = disease-free background scenes)")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0; zero retains legacy single-worker behavior")

    # 毫米域校验：宽度硬下限 1.0 m——add_depression/add_pothole 在
    # [0.5, width-0.5] 上抽中心（generator.py），width < 1 时该区间
    # 为空（lo >= hi），numpy 会直接抛错；在请求入口给出可读报错。
    if args.width < 1.0:
        raise ValueError(
            f"--width must be >= 1.0 m, got {args.width}: disease centers are "
            "sampled in [0.5, width - 0.5] (add_depression / add_pothole), which "
            "is empty below 1.0 m"
        )
    # crowned 路拱是 7 m 双车道口径（±30 mm 中央落差）：3 m 窄路中央
    # 路拱语义上是双车道剖面，建议 plane。不强制——crowned 几何本身合法。
    if config.road.crossfall_shape == "crowned" and args.width <= 3.5:
        warnings.warn(
            f"--width {args.width:g} m <= 3.5 m with crossfall_shape='crowned': "
            "the crowned profile is calibrated for the 7 m two-lane "
            "cross-section; 'plane' is usually the right semantics on narrow "
            "roads. Kept as configured (warning only).",
            stacklevel=2,
        )
    # 轮迹带横向占位 = wheel_separation + rut_width（默认 2.4 m）：更窄的
    # 路面会让车辙沟槽溢出路缘（add_rutting 的 x_L/x_R 落到路外）。
    rutting_span = config.rutting.wheel_separation + config.rutting.rut_width
    if args.width < rutting_span:
        warnings.warn(
            f"--width {args.width:g} m is narrower than the rutting wheel-track "
            f"span ({rutting_span:g} m = wheel_separation + rut_width): rut "
            "grooves would extend beyond the carriageway edge.",
            stacklevel=2,
        )

    # 谱形前置检查：two-segment 需要 primitives 侧两段谱批次（本树未
    # 合入时构造合法但逐场景运行会失败）——入口清晰报错并提示回退，
    # 而不是让批量跑逐场景 warn 失败、退出码 0、产出空数据集。
    if config.road.macro_spectrum_mode == "two-segment":
        from roadmc.data.synthetic.generator import _primitive_supports_mode_value
        from roadmc.data.synthetic.primitives import generate_road_surface

        if not _primitive_supports_mode_value(
            generate_road_surface, "macro_spectrum_mode", "two-segment"
        ):
            raise ValueError(
                "--macro-spectrum two-segment requires the primitives-side "
                "two-segment PSD batch, which is not merged into "
                "primitives.py in this build yet. Re-run with "
                "--macro-spectrum iso-marginal or legacy."
            )

    # 分段条带计划在 on 体制下不可执行（观测管线未分段）；line-profiler
    # 亦仅单块路径。前置硬报错，而不是让批量跑逐场景 warn 失败、退出码
    # 0、产出空数据集。
    estimate = config.estimate_surface_generation()
    if estimate.segmented and config.observation_model != "off":
        raise ValueError(
            f"Strip of {estimate.point_count:,} points requires the segmented "
            f"regime ({estimate.segment_count} segments × "
            f"{estimate.segment_length_m:.3f} m), which currently supports "
            "observation_model='off' only (the sensor pipeline is not "
            "segmented in this build, and line-profiler is single-block "
            "only). Re-run with --observation-model off (the default), or "
            "raise --max-surface-points / --max-surface-memory-mib to hold "
            "the whole strip in one block."
        )

    effective_workers = max(1, args.workers)
    largest_split = max(args.train_count, args.val_count)
    active_workers = min(effective_workers, largest_split) if largest_split else 0
    estimated_parallel_mib = 0.0
    if active_workers:
        estimated_parallel_mib = config.validate_parallel_budget(
            active_workers,
            args.max_parallel_memory_mib,
        )
    return effective_workers, estimated_parallel_mib


def _print_resolution_summary(
    config: GeneratorConfig,
    workers: int,
    estimated_parallel_mib: float,
) -> None:
    estimate = config.estimate_surface_generation()
    model_points = (
        "由训练配置决定"
        if config.model_target_points is None
        else str(config.model_target_points)
    )
    sensor_desc = (
        "全表面直出 (点数/密度请求被忽略)"
        if config.observation_model == "off"
        else f"{config.sensor_output_mode}/约 {config.sensor_output_target_points:,} 点"
    )
    print(
        "分辨率契约: "
        f"表面={config.road.surface_grid_spacing_m * 1000.0:.2f} mm "
        f"({config.road.width:g}m×{config.road.length:g}m, "
        f"{estimate.point_count:,} 点), "
        f"观测模型={config.observation_model}, "
        f"传感器输出={sensor_desc}, "
        f"模型目标={model_points}"
    )
    if estimate.segmented:
        print(
            f"分段条带: {estimate.segment_count} 段 × {estimate.segment_length_m:.3f} m "
            f"(重叠 {estimate.overlap_m:.3f} m, 单段扩展点数 "
            f"{estimate.segment_point_count_ext:,}, worker 峰值 "
            f"{estimate.worker_peak_mib:.0f} MiB)"
        )
    per_scene_gib = estimate.point_count * 48 / 1024 ** 3
    print(
        f"体量估算: 每场景落盘数组约 {per_scene_gib:.2f} GiB"
        "（f32 坐标/法向/特征 + i32 标签，压缩后更小）；"
        "大规模生成前请确认磁盘预算"
    )
    if estimated_parallel_mib:
        print(
            f"安全估算: {workers} worker(s) 表面合成峰值约 "
            f"{estimated_parallel_mib:.1f} MiB"
        )
    print("注意: 表面网格间距不会在传感器输出或模型采样后保持不变。")


def main() -> None:
    args = _build_parser().parse_args()
    config = _config_from_args(args)
    effective_sampling = config.disease.sampling_mode
    workers, estimated_parallel_mib = _validate_request(args, config)

    output_dir = Path(args.output_dir)
    print("\n开始生成数据集...")
    print(f"配置: 路面类型={args.pavement}, 粗糙度={args.roughness}, 每场景最多病害={args.max_diseases}")
    print(f"随机种子: {args.seed}")
    _print_resolution_summary(config, workers, estimated_parallel_mib)

    stats = {
        "train": generate_dataset(args.train_count, config, "train", output_dir, sampling_mode=effective_sampling, num_workers=workers),
        "val": generate_dataset(args.val_count, config, "val", output_dir, sampling_mode=effective_sampling, num_workers=workers),
    }

    print("\n" + "=" * 60)
    class_distribution = {split: verify_class_distribution(output_dir, split) for split in ("train", "val")}

    metadata = {
        "config": {
            "surface_grid_spacing_m": args.surface_grid_spacing,
            "grid_res": args.surface_grid_spacing,
            "pavement": args.pavement,
            "sensor_output_points": args.sensor_output_points,
            "num_points": args.sensor_output_points,
            "sensor_output_density": args.sensor_output_density,
            "target_density": args.sensor_output_density,
            "model_target_points": args.model_target_points,
            "roughness": args.roughness,
            "seed": args.seed,
            "max_diseases": args.max_diseases,
            "num_classes": NUM_CLASSES,
            "sampling_mode": effective_sampling,
            "use_stratified": not args.no_stratified,
            "macro_spectrum": args.macro_spectrum,
            "agc_mode": args.agc_mode,
            "max_surface_points": args.max_surface_points,
            "max_surface_memory_mib": args.max_surface_memory_mib,
            "max_parallel_memory_mib": args.max_parallel_memory_mib,
            "width_m": args.width,
            "length_m": args.length,
            "observation_model": args.observation_model,
        },
        "resolution_contract": config.resolution_metadata(),
        "parallel_surface_estimated_peak_memory_mib": estimated_parallel_mib,
        "generated_at": datetime.now().isoformat(),
        "train_count": args.train_count,
        "val_count": args.val_count,
        "statistics": {
            split: {
                "requested": s["requested"],
                "generated": s["generated"],
                "failed": s["failed"],
                "point_stats": s["point_stats"],
            }
            for split, s in stats.items()
        },
        "class_distribution": class_distribution,
    }

    metadata_file = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== 生成完成 ===")
    print(f"训练集: {stats['train']['generated']} 个场景 ({stats['train']['failed']} 个失败)")
    print(f"验证集: {stats['val']['generated']} 个场景 ({stats['val']['failed']} 个失败)")
    print(f"输出目录: {output_dir}")
    print(f"元数据: {metadata_file}")


if __name__ == "__main__":
    main()
