# RoadMC 交接说明

## 2026-07-16: Observable-Feature Contract and Medium-Scale Evidence

### Status

The privileged synthetic crack-boundary input has been removed from the active
RoadMC path.  New synthesis, legacy-data loading, real-point-cloud loading,
training checkpoints, evaluation, and domain diagnostics now use or enforce
the same observable input contract:

```text
roadmc.observable_features.v1
[normalized_intensity, pca_curvature, signed_local_height_residual]
```

For a local covariance matrix `C`, curvature is `lambda_min / trace(C)`.  The
third feature is the signed orthogonal residual to a local PCA tangent plane,
divided by the kNN support radius.  It is dimensionless, translation/scale
invariant, and does not accept labels as an argument.

### Credibility Repairs Completed

- Removed `crack_boundary_dist` from `SyntheticRoadDataset.generate_scene`.
  The previous channel was derived from synthetic labels and is no longer a
  model input.
- `RealRoadDataset` computes the same geometry features from real XYZ and
  normalized intensity rather than padding curvature/residual with zeros.
- Legacy `.npz` scenes without the new schema are automatically recomputed
  from stored XYZ and intensity.  New class-budget generation writes schema,
  feature names, and kNN count into every scene.
- Replaced the label-conditioned `EdgeLoss` gate with a supervised, differentiable
  BEV damage-boundary loss over all valid points.  Labels are targets only.
- Validation/test sampling is deterministic uniform sampling.  Disease-aware
  sampling remains training-only, so validation prevalence and calibration are
  not artificially rebalanced.
- Checkpoints store the feature schema and training `max_points`; evaluation
  rejects a different point count unless explicitly requested as a diagnostic.
- Corrected class-budget accounting: an independent quota now counts only a
  scene explicitly forced for that target label, not accidental co-occurrence.
- Added bounded effective-number class weights, synthetic dataset validation,
  class-weight generation, and domain diagnostics for the exact model input
  geometry descriptor.

### Medium-Scale Dataset

`data/credibility_v1_5k/` is intentionally untracked and was generated with
`roadmc/scripts/generate_class_budget.py` using 16 processes, 2 cm grid
spacing, 2048 final points per scene, mixed pavement, ISO roughness B, seed
`20260716`.

- Train: 4,144 scenes, 112 intentionally forced scenes per each label 1..37.
- Validation: 851 scenes, 23 intentionally forced scenes per each label 1..37.
- Train target-point floor actually achieved: 8,925 per class minimum.
- Validation target-point floor actually achieved: 1,606 per class minimum.
- Eight failed individual generation attempts were automatically replaced; the
  final class-budget manifest is complete and has zero feature-contract errors.
- Full validation report:
  `output/credibility_v1_5k/data_validation.json`.
  It checked all 4,995 files and reconstructed observable features for 64
  deterministic scenes in each split, with zero errors/warnings.
- Binary training point counts are background 7,711,430 and disease 775,482.
  Effective-number weights are `[0.701096, 1.298904]` at beta `0.999999`.

### GPU Evidence

Device: RTX 5060 Laptop GPU, 8 GB VRAM, PyTorch 2.11.0+cu128.

The selected practical configuration is Swin3D + mHC + hybrid Muon/AdamW,
`embed_dim=48`, depths `(1,1,2,1)`, heads `(3,3,6,6)`, window 32, batch 4,
2048 points, AMP, Muon LR `1e-2`, weight decay `1e-2`.

Important optimizer finding: Muon at `2e-4` produced a background-only first
epoch.  Fixed-scene overfit probes established that `1e-2` to `2e-2` is the
correct order of magnitude for this implementation; 100 steps at `1e-2`
reached 0.768 IoU on the probe.  The low-LR run is diagnostic evidence only,
not a reported benchmark.

The 2048-point binary checkpoint is:

```text
output/credibility_v1_5k/binary_fullpoints_mhc_muon_lr1e2/
lightning_logs/version_2/checkpoints/baseline-epoch=2-val_mIoU=0.719.ckpt
```

Training-epoch global validation at threshold 0.5:

| Epoch | Disease IoU | Precision | Recall | ECE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5916 | 0.8249 | 0.6766 | 0.0079 |
| 1 | 0.6415 | 0.9100 | 0.6850 | 0.0023 |
| 2 | 0.7193 | 0.8614 | 0.8134 | 0.0018 |

Independent threshold-selection evaluation is the publishable synthetic result:

- 170 validation scenes selected the threshold (0.57); the remaining 681
  scenes were never used for threshold selection.
- Held-out disease IoU / supported mIoU: **0.7235**.
- Precision / recall: **0.8874 / 0.7966**.
- ECE / Brier / NLL: **0.0020 / 0.0442 / 0.0840**.
- Scene-block bootstrap 95% CI for IoU: **[0.7070, 0.7391]**, 1,000 draws.
- Evidence: `output/credibility_v1_5k/evaluation/binary_epoch3_independent.json`.

This validates binary segmentation only on independently generated synthetic
scenes.  It is not evidence of real-domain semantic accuracy or 38-class
accuracy.

The safe binary checkpoint was also loaded into four/eight/full38 models under
the current feature schema.  For each stage, 120 non-head tensors loaded, only
the two classifier-head tensors were reset, and a GPU forward/backward/update
returned finite loss.  Evidence:
`output/credibility_v1_5k/curriculum_transfer_smoke.json`.  This is a transfer
smoke test, not a multi-class benchmark.

### Real-Domain Bridge

`output/credibility_v1_5k/m2s_ground_domain_gap.json` compares 64 new
synthetic validation scenes with three public unlabeled M2S-RoAD PCD frames
after a RANSAC ground-plane proxy on the real frames.

- Joint RBF-MMD: 0.6179.
- Model-input signed residual/radius RBF-MMD: 0.0100.
- Large remaining mismatches: density MMD 0.8847, intensity MMD 1.0782,
  normal-tilt MMD 0.8032.

Interpretation: local geometry residuals now align substantially better than
sensor density and intensity.  Prioritize sensor-aware density/intensity
randomization or real-scan calibration before any real-data fine-tuning.
The three PCD frames are unlabeled and must not be used to claim real mIoU.

### Verification

- `python -m compileall -q roadmc` passed.
- `.venv\\Scripts\\python.exe -m unittest discover -s roadmc -p "test_*.py" -v`
  passed: 34 tests.
- The test suite covers feature invariance, label-leak regression, edge-loss
  gradients, legacy-file isolation, real-loader features, deterministic
  validation sampling, physical 38-class reachability, budget accounting,
  class weighting, domain diagnostics, metrics, and dataset validation.

### Next Work

1. Freeze the binary checkpoint and threshold before making a true held-out
   synthetic test split or comparing architecture ablations.
2. Use the safe binary checkpoint only as a curriculum initializer for
   four/eight/full38 stages; report those stages separately and do not infer
   38-class quality from the binary result.
3. Acquire licensed, labeled real road-damage scenes with verified units and
   a JTG-compatible mapping.  Then apply the same fixed feature schema,
   source/target domain report, and held-out evaluation protocol.
4. Address the dominant real-domain gaps with calibrated intensity/density
   augmentation before attempting unsupervised adaptation or RL.

## 2026-07-15: Weeks 1-3 Credibility Repair Update

### Active Objective

Complete a credible path from physics-based synthetic point clouds to real-scan
diagnostics: global scene-level evaluation, controlled 38-class coverage,
binary-to-38 curriculum training, and a measurable real-data bridge.

### Completed This Session

- Added global validation/evaluation metrics in `roadmc/metrics.py`:
  full-split confusion matrix, supported-class mIoU, foreground IoU,
  ECE/Brier/NLL, scene-block bootstrap confidence intervals, and binary
  threshold scanning.
- Fixed the offline GPU evaluator so targets are moved to the model device and
  so `--bootstrap-samples 0` does not crash report rendering.
- Made every JTG non-background label `1..37` physically reachable through
  `roadmc/data/synthetic/labels.py` and
  `SyntheticRoadDataset.generate_scene(..., target_label=...)`.
- Added the missing asphalt/concrete patching primitive for labels `20` and
  `37`. Its geometry is a rounded repair region blended to a local least-
  squares tangent plane with a cubic smoothstep boundary.
- Fixed two generator failure modes:
  narrow cracks can vanish under coarse sampling, and zero-noise spherical
  round-trip error could clear all crack labels. Label support now respects the
  sampling footprint, while physical crack width remains unchanged.
- Added target-label-aware final resampling, so rare forced classes survive
  fixed-point-count generation.
- Added `roadmc/scripts/generate_class_budget.py`. It is resume-safe and only
  declares completion once every selected class reaches both its independent
  scene quota and effective point quota. Its manifest records scene count,
  forced instance count, and target point count per class.
- Added curriculum mappings `binary -> four -> eight -> full38` in
  `roadmc/data/curriculum.py`, and connected them to training and evaluation.
  The classifier head is reset while backbone/MHC weights are reused through
  `--pretrained_checkpoint`.
- Verified one-epoch GPU smoke runs with `RTX 5060 Laptop`, AMP, MHC, and
  Muon for binary, four-class, eight-class, and 38-class stages. These used
  only five synthetic scenes per split and are pipeline checks, not benchmarks.
- Added a real-data bridge:
  `roadmc/data/real/dataset.py` now supports `.npy`, `.ply`, `.pcd`, `.las`,
  and `.laz`; `roadmc/data/real/metadata.py` defines JSON sidecars for sensor,
  coordinate units, intensity scale, segment, and provenance.
- Added `roadmc/domain_gap.py` and
  `roadmc/scripts/diagnose_domain_gap.py`. It compares density, intensity,
  normal tilt, PCA curvature, and local height residual using Wasserstein-1,
  Energy distance, per-feature RBF-MMD, and joint RBF-MMD.

### Verified Evidence

- `python -m compileall -q roadmc` passed.
- `.venv\Scripts\python.exe -m unittest discover -s roadmc -p "test_*.py" -v`
  passed: 22 tests.
- Controlled synthesis test confirms all labels `1..37` occur in final scenes.
- Class-budget multi-process smoke generation completed train/val with zero
  failures after target-aware resampling.
- GPU binary smoke result: `val_mIoU=0.145` on five validation scenes. It only
  proves the training/checkpoint/global-evaluation path works.
- GPU four/eight/full38 stage transfers all loaded 100 backbone tensors and
  correctly reinitialized the two classifier-head tensors.
- Synthetic-to-synthetic domain diagnostics ran successfully.
- Three public M2S-RoAD PCD frames were downloaded only for unlabeled domain
  diagnostics at `data/real/m2s_road_sample/` and are intentionally untracked.
  Ground-plane filtering retained about 39% of each frame. Reports are at:
  `output/m2s_road_sample_domain_gap.json` and
  `output/m2s_road_sample_ground_domain_gap.json`.

### Public Data Status

- M2S-RoAD dataset card license: `CC-BY-NC-ND-4.0`.
- Full archives are too large for this machine: approximately 114 GB tarball
  and 271 GB zip. Do not download the complete archive blindly.
- Individual public GitHub PCD frames are about 6.5 MB and are suitable for
  loader/domain-gap smoke tests only. They are unlabeled point clouds, so they
  cannot validate RoadMC 38-class segmentation accuracy.
- Local PCD sidecars state that units are assumed to be meters and intensity
  is uint8; this unit assumption must be confirmed before publishing a metric
  real-data claim.

### Important Remaining Risks

- Resolved for `roadmc.observable_features.v1`: the active third feature is a
  geometry-only local residual.  Old datasets/checkpoints remain legacy and
  are intentionally blocked from credible new experiments unless their inputs
  are recomputed under the current schema.
- The M2S-RoAD full-frame comparison is a sensor-level diagnostic. The optional
  RANSAC ground-plane filter is a geometric road proxy, not semantic road
  segmentation.
- No real labeled 38-class benchmark has been acquired. Current real evidence
  measures domain mismatch, not real semantic segmentation quality.
- Previous small-data scores in this repository are not publishable evidence;
  retain the distinction between smoke validation and a held-out large-scale
  experiment.

### Recommended Next Actions

1. Generate a real class-budgeted synthetic train/val set with a documented
   per-class point floor, then run the binary curriculum for enough epochs to
   establish a stable baseline and confidence interval.
2. Replace the privileged crack-boundary feature with an observable local
   height residual before any real-data fine-tuning.
3. Download a licensed, labeled real road-damage subset or create a small
   manually verified annotation set; run the same domain-gap script before and
   after calibrated domain randomization.
4. Use the curriculum checkpoints in order: binary -> four -> eight -> full38,
   reporting global mIoU, per-class support, calibration, and bootstrap CIs.

更新时间：2026-06-18

## 当前目标

先修复核心 bug，再根据最新论文寻找可落地的优化方向。

## 已完成

### 核心 bug 修复

- 修复了 `DiceLoss` 在 `valid_mask` 分支下的维度错误。
- 将 `EdgeLoss` 从 `argmax` 硬预测改成了可反传的 soft 版本。
- 将验证 `val_mIoU` 的口径改为与评估一致的“跳过背景类”。
- 修复了注意力模块，当前不是全局 `N x N` attention，而是分窗局部 attention。
- 修复了合成数据生成里的 `mixed` / `use_stratified` 逻辑。
- 修复了真实数据集入口，当前以结构化 `.npy` 为主。

### 优化器

- `RoadMCSegModel` 已支持 `Muon` 和 `AdamW` 切换。
- `train.py` 已增加 `--optimizer` 参数。
- 手写训练分支（GAN / end2end）已经共用同一套优化器构造逻辑。

### Backbone / 实验入口

- `Swin3D` 已增加 `use_mhc` 开关。
- `RoadMCSegModel` 已增加 `backbone_name`。
- `train.py` 已增加 `--backbone swin3d|pointmamba`。
- 新增了 `roadmc/models/backbone/pointmamba.py`，这是一个 PointMamba-inspired 的可跑实验骨干。

## 关键文件状态

- `roadmc/models/model_pl.py`
  - 当前包含损失修复、`Muon` / `AdamW` 切换、`backbone_name` 切换。
- `roadmc/train.py`
  - 当前支持 `--optimizer`、`--backbone`、`--use_mhc`、`--no_mhc`。
- `roadmc/models/backbone/swin3d.py`
  - 当前支持 `use_mhc`。
- `roadmc/models/backbone/pointmamba.py`
  - 新增骨干，采用 Morton 排序 + scan mixer，已通过最小前向自检。
- `roadmc/models/attention/window_attention.py`
  - 当前为真正的局部窗口 attention，不再构建全局注意力矩阵。

## 运行方式

### Baseline

```bash
python roadmc/train.py baseline --data_dir ./data/synthetic_output --backbone swin3d --optimizer muon
```

### 对照实验

```bash
python roadmc/train.py baseline --backbone swin3d
python roadmc/train.py baseline --backbone pointmamba
python roadmc/train.py baseline --backbone pointmamba --no_mhc
python roadmc/train.py baseline --optimizer adamw
```

### GAN / end2end

```bash
python roadmc/train.py gan_enhanced --backbone pointmamba
python roadmc/train.py end2end --backbone pointmamba
```

## 已知限制 / 注意点

- `PointMambaBackbone` 是 PointMamba-inspired 实现，不依赖 `mamba_ssm` 官方包。
- 当前环境里 `mamba_ssm` 未安装。
- 代码里还有一些与本文任务无关的本地脏文件，不要误删或回滚。
- 目前只做了语法检查和最小前向自检，没有完整跑大训练。

## 推荐下一步

1. 跑小规模 benchmark，比较：
   - `Swin3D + MHC`
   - `Swin3D + no MHC`
   - `PointMamba + MHC`
   - `PointMamba + no MHC`
2. 记录 `val_mIoU`、`val_loss`、训练速度、显存占用。
3. 如果 `PointMamba` 跑得稳，再继续做：
   - 更接近原论文的扫描顺序
   - 更强的局部-全局混合块
   - 进一步压缩 attention / scan 的开销

## 建议不要碰的东西

- 不要随手改动根目录里那些与本任务无关的脏文件。
- 不要把当前实验入口重新收回到单一 backbone / 单一优化器。
- 不要先上更大的结构重写，先用对照实验验证收益。

## 最新测试记录

### 小规模闭环测试 1

配置：
- `B=2`
- `N=64`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `num_heads=(2,2,4,4)`
- `window_size=16`
- `use_checkpoint=False`
- `optimizer=muon`

结果：
- `swin3d + use_mhc=False`: `loss=4.3495`, `time=0.0581s`, `grad_ok=True`
- `swin3d + use_mhc=True`: `loss=4.2927`, `time=0.1330s`, `grad_ok=True`
- `pointmamba + use_mhc=False`: `loss=4.3983`, `time=0.0308s`, `grad_ok=True`
- `pointmamba + use_mhc=True`: `loss=4.3324`, `time=0.0471s`, `grad_ok=True`

### 小规模闭环测试 2

同样配置下，验证了 `training_step -> backward -> step -> scheduler.step` 的完整闭环。

结果：
- `swin3d + use_mhc=False`: `loss=4.3495`, `time=0.0581s`, `grad_ok=True`, `opt=HybridMuonAdamW`
- `swin3d + use_mhc=True`: `loss=4.2927`, `time=0.1330s`, `grad_ok=True`, `opt=HybridMuonAdamW`
- `pointmamba + use_mhc=False`: `loss=4.3983`, `time=0.0308s`, `grad_ok=True`, `opt=HybridMuonAdamW`
- `pointmamba + use_mhc=True`: `loss=4.3324`, `time=0.0471s`, `grad_ok=True`, `opt=HybridMuonAdamW`

结论：
- 混合优化器可用。
- `MHC` 可用。
- `PointMamba` 这条线在小规模测试中明显更快。
- 当前所有组合都能完成前向、反向、参数更新和调度器步进。

### 小规模正式测试 3

配置：
- `B=2`
- `N=128`
- `embed_dim=24`
- `depths=(1,1,1,1)`
- `num_heads=(2,2,4,4)`
- `window_size=16`
- `optimizer=muon`

结果：
- `swin3d + use_mhc=False`: `loss=4.4161`, `time=0.1366s`, `opt=HybridMuonAdamW`
- `swin3d + use_mhc=True`: `loss=4.3765`, `time=0.0703s`, `opt=HybridMuonAdamW`
- `pointmamba + use_mhc=False`: `loss=4.5474`, `time=0.0390s`, `opt=HybridMuonAdamW`
- `pointmamba + use_mhc=True`: `loss=6.0346`, `time=0.0386s`, `opt=HybridMuonAdamW`

结论：
- `PointMamba + MHC` 在这个超小配置下 loss 偏高，说明当前实现还需要进一步调参。
- 但四个组合都能稳定完成训练闭环，没有出现梯度或优化器错误。

### 小规模延长训练 4

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=256`
- 训练步数：5 step

配置：
- `swin3d + use_mhc=True`
- `pointmamba + use_mhc=True`
- `embed_dim=24`
- `depths=(1,1,1,1)`
- `num_heads=(2,2,4,4)`
- `window_size=16`
- `optimizer=muon`

结果：
- `swin3d + MHC`: `avg_loss=4.4995`, `elapsed=0.92s`, `steps=5`
- `pointmamba + MHC`: `avg_loss=4.3705`, `elapsed=0.32s`, `steps=5`

逐步 loss：
- `swin3d + MHC`: `4.6096 -> 4.6140 -> 4.1779 -> 4.6165 -> 4.4794`
- `pointmamba + MHC`: `4.0246 -> 4.5712 -> 4.8113 -> 3.7362 -> 4.7090`

结论：
- 在这 5 step 的小规模观察里，`PointMamba + MHC` 明显更快。
- 两者 loss 都有波动，没有出现发散。
- `PointMamba + MHC` 还看不出稳定优势，但已经是可继续调参的状态。

### 小规模长跑观察 5

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=512`
- 训练步数：200 step

配置：
- `swin3d + use_mhc=True`
- `pointmamba + use_mhc=True`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `num_heads=(2,2,4,4)`
- `window_size=16`
- `optimizer=muon`

结果：
- `swin3d + MHC`: `steps=200`, `avg_loss=4.2300`, `first10=4.1914`, `last10=4.2687`, `elapsed=44.46s`
- `pointmamba + MHC`: `steps=200`, `avg_loss=7.0392`, `first10=8.2374`, `last10=6.9469`, `elapsed=16.36s`

抽样 loss：
- `swin3d + MHC`: `4.5077 -> 4.1159 -> 4.1246 -> 4.3816 -> 4.2001 -> 3.9501 -> 4.4252`
- `pointmamba + MHC`: `8.4594 -> 8.3643 -> 8.2954 -> 8.5027 -> 7.5402 -> 6.4728 -> 8.3827`

结论：
- `PointMamba + MHC` 的速度仍然明显更好。
- 但在 200 step 的观察里，它的 loss 明显高于 `Swin3D + MHC`，且波动更大。
- 这说明 `PointMamba` 路线值得继续，但当前实现还需要进一步调参或改扫描/块设计，不能直接替换掉 Swin3D。

### PointMamba 扫描修正后复测 6

修改：
- 将 `PointMambaBlock` 的 scan 从近似形式改成显式递推状态更新。
- 将 `MHC` 的 residual 接口改为使用 `x_scan` 作为残差来源，而不是原始输入残差。

结果：
- `pointmamba + MHC`: `steps=200`, `avg_loss=4.5358`, `first10=4.5631`, `last10=4.5173`, `elapsed=44.59s`
- 抽样 loss: `4.5841 -> 4.6547 -> 4.6972 -> 4.6233 -> 4.3683 -> 4.4735 -> 4.3175`

结论：
- 修正后 `PointMamba + MHC` 的 loss 回到了和 `Swin3D` 接近的量级，波动也显著收敛。
- 速度优势仍然存在。
- 这个版本已经比前一版更适合继续做调参和结构微调。

### PointMamba MHC 隔离测试 7

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=512`
- 训练步数：200 step

配置：
- `backbone=pointmamba`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `optimizer=muon`

结果：
- `pointmamba + use_mhc=False`: `avg_loss=4.3781`, `first10=4.4344`, `last10=4.3530`, `elapsed=45.89s`
- `pointmamba + use_mhc=True`: `avg_loss=4.1801`, `first10=4.2990`, `last10=4.1210`, `elapsed=50.81s`

结论：
- scan 修正后，`MHC` 对 PointMamba 是有帮助的。
- 当前主要问题已经不是 loss 发散，而是显式 scan 递推导致速度下降。

### PointMamba 学习率小扫 8

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=512`
- 训练步数：100 step

配置：
- `backbone=pointmamba`
- `use_mhc=True`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `optimizer=muon`

结果：
- `lr=5e-5`: `avg_loss=4.5427`, `first10=4.5506`, `last10=4.5544`, `elapsed=20.95s`
- `lr=1e-4`: `avg_loss=4.5328`, `first10=4.5493`, `last10=4.5379`, `elapsed=22.05s`
- `lr=3e-4`: `avg_loss=4.4933`, `first10=4.5440`, `last10=4.4723`, `elapsed=21.60s`

结论：
- `PointMamba + MHC` 对更高学习率更友好。
- 当前建议优先用 `lr=3e-4` 继续做更长训练观察。

### PointMamba 长跑观察 9

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=512`
- 训练步数：500 step

配置：
- `backbone=pointmamba`
- `use_mhc=True`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `optimizer=muon`
- `lr=3e-4`
- `t_max=500`

结果：
- `steps=500`
- `avg_loss=4.0206`
- `first10=4.5629`
- `last10=3.6743`
- `elapsed=103.25s`

窗口均值：
- `0-9`: `4.5629`
- `90-99`: `4.3244`
- `190-199`: `4.0352`
- `290-299`: `3.8815`
- `390-399`: `3.7416`
- `490-499`: `3.6743`

结论：
- `PointMamba + MHC + lr=3e-4 + t_max=500` 呈现稳定下降趋势。
- 这组配置已经从“可跑”进入“值得继续扩大实验”的状态。

### Swin3D / PointMamba 公平对照 10

数据：
- `data/synthetic_output`
- `batch_size=2`
- `max_points=512`
- 训练步数：500 step

共同配置：
- `use_mhc=True`
- `embed_dim=32`
- `depths=(1,1,1,1)`
- `optimizer=muon`
- `lr=3e-4`
- `t_max=500`

结果：
- `swin3d + MHC`: `avg_loss=3.7884`, `first10=4.4731`, `last10=3.9103`, `elapsed=115.47s`
- `pointmamba + MHC`: `avg_loss=4.0206`, `first10=4.5629`, `last10=3.6743`, `elapsed=103.25s`

窗口均值：
- `swin3d`: `0-9=4.4731`, `90-99=4.2056`, `190-199=4.1468`, `290-299=3.2185`, `390-399=3.4042`, `490-499=3.9103`
- `pointmamba`: `0-9=4.5629`, `90-99=4.3244`, `190-199=4.0352`, `290-299=3.8815`, `390-399=3.7416`, `490-499=3.6743`

结论：
- `Swin3D` 的总体平均 loss 更低。
- `PointMamba` 的最后窗口 loss 更低，下降趋势更平滑，耗时也更短。
- 这说明不能只看平均 loss；下一步应该加入验证集 mIoU 和更长训练，判断谁的泛化更好。

### 验证集诊断 11

验证批次标签分布：
- `labels` 主要由背景类 `0` 构成，同时只有少量病害类 `3`、`13`

短训后预测分布：
- `PointMamba` 预测主要落在类别 `26`、`36`
- 验证集 `mIoU = 0.0`

结论：
- 当前 38 类设置下，短训阶段 mIoU 很容易被压成 0。
- 这更像是类别不平衡 + 训练步数不够的问题，不是单一 backbone 的问题。
- 下一步优先做二分类或少类 sanity check，再考虑继续拉长 38 类训练。

### 二分类 sanity check 12

数据：
- `data/synthetic_output`
- `binary=True`
- `batch_size=2`
- `max_points=512`
- 训练步数：300 step

配置：
- `use_mhc=True`
- `optimizer=muon`
- `lr=3e-4`
- `t_max=300`
- `num_classes=2`

结果：
- `pointmamba + MHC`: `avg_loss=0.8027`, `first25=0.8005`, `last25=0.8129`, `final_val_mIoU=0.0`
- `swin3d + MHC`: `avg_loss=0.8045`, `first25=0.9098`, `last25=0.8181`, `final_val_mIoU=0.0`

预测分布诊断：
- 验证 batch 标签分布：`background=512`, `disease=512`
- `pointmamba` 预测分布：全部预测为 `background`
- `swin3d` 预测分布：全部预测为 `background`
- `pointmamba` 平均概率约为 `[0.5296, 0.4704]`
- `swin3d` 平均概率约为 `[0.5419, 0.4581]`

结论：
- 两个模型都不是结构性坏掉，而是二分类短训后决策塌到背景类。
- 下一步应优先加 disease class weight 或调整二分类阈值，而不是继续盲目拉长训练。

### 二分类加权验证 13

数据：
- `data/synthetic_output`
- `binary=True`
- `batch_size=2`
- `max_points=512`
- 训练步数：200 step

配置：
- `backbone=pointmamba`
- `use_mhc=True`
- `optimizer=muon`
- `lr=3e-4`
- `class_weights=[1.0, 2.5]`

结果：
- `final_val_mIoU=0.5671`
- `avg_loss=0.8949`
- `first25=0.9127`
- `last25=0.8912`

预测分布：
- 训练后验证 batch 预测约为 `background=95`, `disease=929`
- 平均概率约为 `[0.4712, 0.5288]`

结论：
- 加 disease class weight 后，二分类 mIoU 从 0 明显恢复到约 0.56。
- 当前主要问题是类别推动不足，不是 backbone 完全不可用。
- 下一步建议把二分类权重做成训练参数，并继续寻找更稳的权重区间。
### 2026-06-18 当前目标与最新诊断 14

当前执行目标：
- 设计并执行从“二分类加权预训练”过渡到“38 类训练”的后续实验路线。
- 先补齐训练参数与快速诊断能力，再做可控小规模验证。

新增/修改：
- `roadmc/train.py`
  - 新增 `--binary_class_weights`，格式为 `bg,disease`，例如 `1.0,2.5`。
  - 二分类时该参数优先于 `--class_weights`，用于显式控制背景/病害权重。
- `roadmc/scripts/quick_diagnose.py`
  - 新增轻量诊断脚本，不依赖真实 Lightning 环境。
  - 支持 `--binary`、`--binary_class_weights`、`--class_weights`、`--backbone`、`--no_mhc`。
  - 输出训练 loss、验证 mIoU、预测类别分布、标签分布、平均概率。

已验证命令：
- `python -m py_compile roadmc\train.py roadmc\scripts\quick_diagnose.py roadmc\models\model_pl.py`

二分类加权短诊断：
- 命令：`python roadmc\scripts\quick_diagnose.py --binary --binary_class_weights 1.0,2.5 --backbone pointmamba --steps 80 --eval_batches 2 --max_points 512 --batch_size 2 --lr 3e-4 --seed 303`
- 结果：`avg_loss=0.9074`, `first25=0.9256`, `last25=0.9150`, `val_mIoU=0.6264`
- 分布：`pred_counts={0: 316, 1: 1732}`, `label_counts={0: 682, 1: 1366}`, `mean_prob=[0.4786, 0.5214]`
- 结论：二分类加权有效，80 step 即恢复到非零 mIoU；预测仍偏向 disease，`1.0,2.5` 可能略重，但可作为预训练起点。

38 类权重短诊断：
- 命令：`python roadmc\scripts\quick_diagnose.py --class_weights data\synthetic_output\class_weights.pt --backbone pointmamba --steps 100 --eval_batches 2 --max_points 512 --batch_size 2 --lr 3e-4 --seed 303`
- 结果：`avg_loss=1.5683`, `first25=1.5873`, `last25=1.5037`, `val_mIoU=0.0000`
- 分布：`pred_counts={23: 238, 34: 1807, 9: 3}`, `label_counts={0: 682, 3: 376, 13: 478, 1: 512}`
- 结论：38 类从零短训仍不可靠，预测落在错误类别，无法命中验证标签。

推荐下一阶段路线：
1. 增加 checkpoint 保存/加载能力，先保存二分类加权预训练 checkpoint。
2. 增加二分类到 38 类迁移加载逻辑：复用 backbone、MHC、neck 参数，跳过分类 head。
3. 做 38 类小规模微调对照：A = 38 类从零 + class weights；B = 二分类预训练迁移 + 38 类 class weights。
4. 统一输出 loss、val mIoU、pred_counts、label_counts、mean_prob/top predicted classes。

当前判断：
- 问题主线不是 PointMamba/MHC 是否能训练，而是类别不平衡和 38 类冷启动太硬。
- 二分类加权已经证明模型能学到前景/背景边界。
- 下一步目标应聚焦“迁移初始化 + 38 类细分”，不要继续只盲目拉长 38 类从零训练。

### 2026-06-19 迁移对照结果 15

新增能力：
- `roadmc/scripts/quick_diagnose.py`
  - 新增 `--pretrained_binary`，可直接把二分类 checkpoint 加载到 38 类模型中。
  - 新增 `--save_checkpoint`，可把当前模型权重落盘，便于二阶段微调。

二分类预训练快照：
- 命令：
  - `python roadmc\scripts\quick_diagnose.py --binary --binary_class_weights 1.0,2.5 --backbone pointmamba --steps 60 --eval_batches 2 --max_points 512 --batch_size 2 --lr 3e-4 --seed 303 --save_checkpoint data\synthetic_output\binary_pretrain_pointmamba_mhc.ckpt`
- 结果：
  - `val_mIoU=0.6175`
  - `avg_loss=0.9082`
  - `saved_checkpoint=data\synthetic_output\binary_pretrain_pointmamba_mhc.ckpt`

38 类对照：
- 零初始化：
  - `val_mIoU=0.0000`
  - `avg_loss=1.6122`
- 加载二分类预训练：
  - `val_mIoU=0.0000`
  - `avg_loss=1.6128`
  - `loaded=96`, `missing=3`, `unexpected=0`

结论：
- 迁移加载接口是通的，backbone 权重确实成功加载。
- 但在当前 60 step、`max_points=512` 的小规模诊断里，二分类预训练并没有立刻把 38 类 mIoU 拉起来。
- 这说明下一步不能只靠“直接加载后继续同配方训练”，更像是需要：
  1. 冻结 backbone 先训分类头；
  2. 或做更长的 38 类微调；
3. 或先做“少类到多类”的分阶段扩展。

### 2026-06-19 冻结骨干实验 16

实验配置：
- `backbone=pointmamba`
- `binary=False`
- `class_weights=data\synthetic_output\class_weights.pt`
- `pretrained_binary=data\synthetic_output\binary_pretrain_pointmamba_mhc.ckpt`
- `freeze_backbone=True`
- `steps=80`
- `max_points=512`
- `batch_size=2`
- `lr=3e-4`

结果：
- `loaded=96`
- `missing=3`
- `unexpected=0`
- `frozen params=1238976`
- `trainable params=1254`
- `val_mIoU=0.0000`
- `avg_loss=1.5873`

结论：
- 冻结骨干后只训 38 类 head，当前短跑仍然没起来。
- 说明问题不只是 backbone 没迁移到位，分类头本身也很难在当前数据和步数下直接对上。
- 下一步更合理的试验不是继续盲训，而是：
  1. 先把 38 类里高频少数类做子集实验；
  2. 或把 backbone 解冻后做更长的 38 类微调；
  3. 或先加一个中间阶段，比如二分类 -> 4/8 类 -> 38 类。

### 2026-06-19 GPU 二分类与数据生成控制 17

数据生成：
- 新增 `roadmc/scripts/generate_synthetic.py` 参数：
  - `--num-points`
  - `--target-density`
- 目的：避免默认 65536 点/场景导致生成阶段过慢、CPU 多 worker 看起来像“卡死”。
- 实测：
  - `data\synthetic_output_big` 之前多 worker 生成被中断，留下 `train=242`, `val=0`。
  - `data\synthetic_binary_control` 使用 `train=300`, `val=80`, `num_points=8192`, `workers=4` 仍在 15 分钟内未完成，只生成 `train=164`, `val=0`。
- 结论：
  - 当前瓶颈主要在合成数据生成器，不是 GPU 训练。
  - 后续要扩大数据集，建议先优化生成器或改成小批量可恢复生成，不要一次性长时间阻塞。

GPU 二分类正式训练：
- 数据：`data\synthetic_output`，完整 `train=600`, `val=150`。
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`。
- 配置：
  - `binary=True`
  - `binary_class_weights=1.0,2.5`
  - `backbone=pointmamba`
  - `optimizer=muon`
  - `use_mhc=True`
  - `batch_size=4`
  - `max_points=512`
  - `num_workers=4`
  - `lr=3e-4`
- 结果：
  - 未加权正式 GPU 训练：`val_mIoU=0.000`
  - 加权 epoch 0：`val_mIoU=0.284`, `val_loss=0.875`
  - 从 epoch 0 checkpoint 续训到 epoch 1：`val_mIoU=0.338`, `val_loss=0.865`
- 最新 checkpoint：
  - `lightning_logs\version_2\checkpoints\baseline-epoch=1-val_mIoU=0.338.ckpt`

结论：
- 二分类 mIoU 低的主因是正式训练没有使用显式二分类权重。
- 加 `--binary_class_weights 1.0,2.5` 后，正式 GPU 训练从 0 恢复到 0.284，并在第二个 epoch 到 0.338。
- 下一步建议优先继续二分类加权长训，或者调 `binary_class_weights`，比如 `1.0,2.0`、`1.0,3.0` 做小网格；数据生成扩容应先修成可恢复/分批生成。

### 2026-06-19 Swin3D 二分类冲刺与阈值校准 18

本轮目标：
- 在二分类设置下继续把验证 mIoU 往 `0.7-0.9` 推。
- 先不切回 38 类，保持 `binary=True`、`use_mhc=True`、`optimizer=muon`。

关键代码改动：
- `roadmc/models/model_pl.py`
  - 新增 `binary_threshold` 超参。
  - 二分类验证时可用 disease 概率阈值代替固定 `argmax`。
- `roadmc/train.py`
  - 新增 `--binary_threshold` 参数。
  - 训练时传入 `RoadMCSegModel(binary_threshold=...)`。

当前最佳模型路线：
- 数据：`data\synthetic_output`
  - `train=600`
  - `val=150`
- 配置：
  - `backbone=swin3d`
  - `binary=True`
  - `binary_class_weights=1.0,2.5`
  - `optimizer=muon`
  - `use_mhc=True`
  - `batch_size=4`
  - `max_points=512`
  - `num_workers=4`
  - `lr=3e-4`
  - `lambda_edge=0.5`
  - `precision=16-mixed`

训练结果：
- `lightning_logs\version_7`
  - epoch 0：`val_mIoU=0.545`
  - epoch 1：`val_mIoU=0.545` 文件名对应 checkpoint，metrics 中 disease IoU `0.575`
- 续训到 `lightning_logs\version_8`
  - epoch 2：`val_mIoU=0.6499`, `val_IoU_1=0.7275`, `precision_1=0.9389`, `recall_1=0.7735`
  - epoch 3：`val_mIoU=0.5627`
  - epoch 4：`val_mIoU=0.6253`
- 当前最佳 checkpoint：
  - `lightning_logs\version_8\checkpoints\baseline-epoch=2-val_mIoU=0.650.ckpt`

阈值校准：
- 对 `version_8/epoch=2` checkpoint 做阈值扫描。
- 按全验证集点级混淆统计：
  - 最优阈值约 `0.44`
  - `mIoU≈0.7319`
  - disease IoU `≈0.7052`
  - precision `≈0.9039`
  - recall `≈0.7624`
- 按 Lightning 当前 batch 均值口径：
  - 最优阈值约 `0.43`
  - `val_mIoU≈0.6819`
- 结论：
  - 模型已经具备接近/超过 `0.7` 的全局点级二分类能力。
  - Lightning 日志口径与全局点级口径不同，后续需要统一评估函数，否则“0.7”会有口径差异。

失败/不推荐分支：
- 从 `version_8/epoch=2` 用 `binary_class_weights=1.0,3.0`、`binary_threshold=0.43`、`lr=1e-4` 微调：
  - `lightning_logs\version_9`
  - epoch 0：`val_mIoU=0.6144`
  - epoch 1：`val_mIoU=0.6383`
  - 没有超过 `version_8/epoch=2`
- 继续同配置训练到 epoch 3/4 也没有稳定提升，存在过拟合或调度不合适迹象。

下一步建议：
- 不要先盲目继续加 epoch。
- 优先统一评估口径：
  - 增加全验证集累计混淆矩阵评估脚本/正式 evaluate binary 模式。
  - 同时输出 batch mean mIoU 和 global mIoU。
- 然后围绕当前最佳 checkpoint 做小网格：
  - `binary_threshold`: `0.40-0.50`
  - `binary_class_weights`: `1.0,2.3`、`1.0,2.5`、`1.0,2.7`
  - `lambda_edge`: `0.0`、`0.25`、`0.5`
  - `lr`: `3e-5`、`1e-4`
- 若目标是稳定日志口径 `0.7+`，更可能需要：
  - 更好的数据生成/扩容；
  - 或验证集级别 metric 聚合修正；
  - 或在最佳 checkpoint 上做低学习率、短周期、早停式微调。
