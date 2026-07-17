<div align="center">

# RoadMC

**物理仿真点云生成 + 路面病害点云分割**

点云生成理论 · Swin3D / PointMamba · mHC · Muon / AdamW · 二分类到 38 类细分

> [English](README.en.md)

</div>

---

## 项目概览

RoadMC 包含两个同等重要的部分：

1. **点云生成模块**：用路面粗糙度谱、微纹理、病害几何基元和 LiDAR 观测模型生成带标签的路面点云。
2. **模型训练模块**：用 Swin3D 或 PointMamba 作为点云分割骨干，结合 mHC、Focal/Dice/Edge 损失和 Muon/AdamW 优化器训练逐点病害分割模型。

当前实验路线是先做二分类：背景 `0` 与病害 `1`。二分类 mIoU 稳定后，再用二分类 checkpoint 初始化 backbone，切回 38 类细分。

---

## 最近更新：2026-07-17

### 已完成

- 移除由合成标签派生的 `crack_boundary_dist`，统一合成数据、旧 `.npz`、真实点云和模型 checkpoint 的可观测特征契约：`normalized_intensity`、`pca_curvature`、`signed_local_height_residual`。
- 修复验证/测试集的抽样偏置，训练集保留病害分层抽样，验证与测试改为可复现的均匀抽样。
- 新增受控类别预算生成、数据集验证、自动类别权重、二分类到 `4 -> 8 -> 38` 类的迁移接口，以及合成到真实域的密度/强度/几何差异诊断。
- 在 RTX 5060 Laptop 8 GB 上完成 Swin3D + mHC + Muon/AdamW 的二分类中等规模验证：4,995 个场景、每场景 2,048 点，独立阈值评估 Disease IoU / 支持类 mIoU 为 `0.7235`，bootstrap 95% CI 为 `[0.7070, 0.7391]`。
- 完成 34 项自动化测试、代码编译检查，以及二分类 checkpoint 向 4/8/38 类模型的 GPU 前向/反向迁移 smoke test。

### 待完成

- 固定二分类 checkpoint 和阈值后，建立真正独立的合成测试集，并进行 Swin3D、PointMamba、mHC 和优化器的可重复消融实验。
- 以当前二分类 checkpoint 为初始化，逐阶段完成 4 类、8 类和 38 类训练；多分类结果必须单独报告，不能由二分类指标推断。
- 获取带可靠标签、单位和 JTG 映射的真实道路点云，完成真实域 mIoU 验证。目前下载的 M2S-RoAD 点云只有无标签 PCD，只能用于域差诊断。
- 针对当前主要域差异校准 LiDAR 强度、点密度和法向倾角，再评估域随机化、无监督适配和 GAN；在真实标签和独立测试协议建立前，不把域适配结果宣称为真实域性能。

---

## 方法总览

<p align="center">
  <img src="readmeimage/model_architecture.png" alt="RoadMC end-to-end method overview" width="96%"/>
</p>

图中左侧是点云生成模块，右侧是模型训练模块。生成模块不是简单的数据预处理，而是项目的方法核心之一：它决定了几何形态、标签分布、稀有病害覆盖率和训练数据的上限质量。

---

## 安装

```powershell
pip install -e .
```

或使用 `uv`：

```powershell
uv sync
```

---

## 点云生成：理论与实现

RoadMC 的合成数据不是随机噪声点云，而是从“路面形貌 + 病害形变 + 传感器观测”的角度构造监督数据。

### 理论组成

| 组成 | 作用 | 当前实现 |
| --- | --- | --- |
| 路面基底 | 给出道路整体高程与粗糙度 | ISO 8608 PSD 粗糙度谱，支持 A-E 等级 |
| 微纹理 | 叠加局部细节，避免表面过于光滑 | fBm fractal surface、局部曲率、法向量 |
| 病害基元 | 在几何上生成裂缝、坑槽、车辙、剥落等形态 | `roadmc/data/synthetic/primitives.py` |
| LiDAR 观测 | 让点云更接近扫描数据而不是规则网格 | scan-line resampling、测距噪声、角度扰动、密度控制 |
| 标签传播 | 把病害区域转成逐点监督信号 | 38 类 JTG 风格标签，训练时可折叠为二分类 |
| 场景输出 | 形成可训练的 `.npz` 数据集 | `points`、`labels`、`feats`、`normals`、`pavement_type` |

### 生成新数据集

```powershell
python roadmc/scripts/generate_synthetic.py `
  --train-count 2000 `
  --val-count 500 `
  --output-dir ./data/synthetic_output `
  --pavement mixed `
  --roughness B `
  --workers 8
```

### 扩展已有数据集

扩展脚本适合中断后续跑，会检查已有 `scene_*.npz`，只补足缺失数量。

```powershell
python roadmc/scripts/expand_synthetic_dataset.py `
  --output-dir ./data/synthetic_output `
  --target-total 5000 `
  --workers 16 `
  --num-points 8192 `
  --pavement mixed `
  --roughness B
```

当前设备上建议优先使用 `16` workers。此前 `32` workers 容易触发 Windows 页面文件或内存压力。

### 受控类别预算与数据契约

正式实验应使用类别预算生成，而不是只按文件数扩展数据集。脚本会分别约束每类的强制场景数和有效目标点数，并为每个场景写入 `roadmc.observable_features.v1` 特征契约。

```powershell
python roadmc/scripts/generate_class_budget.py `
  --output-dir ./data/credibility_v1_5k `
  --split both `
  --target-scenes-per-class 112 `
  --min-points-per-class 4000 `
  --grid-res 0.02 `
  --num-points 2048 `
  --workers 16 `
  --pavement mixed `
  --roughness B

python roadmc/scripts/validate_synthetic_dataset.py `
  --data-dir ./data/credibility_v1_5k `
  --split both `
  --feature-check-scenes 64 `
  --output-json ./output/data_validation.json
```

---

## 模型训练：结构与策略

训练模块把合成点云作为监督数据，输出逐点 logits。它既可以做二分类，也可以做 38 类细分。

### 数据加载

`SyntheticPointCloudDataset` 从 `.npz` 文件读取：

- `coords`: 归一化 XYZ 坐标。
- `feats`: 归一化强度、PCA 曲率、有符号局部高度残差；全部可由真实点云在推理时计算。PCA 曲率为 `lambda_min / trace(C)`，局部残差是相对 kNN 切平面的有符号正交距离并按邻域半径归一化。
- `labels`: 38 类标签；二分类时动态折叠为 `labels > 0`。
- `valid_mask`: padding 后忽略无效点，避免 loss 和 mIoU 被填充点污染。

训练集会分层保留病害点，降低 batch 中全背景样本过多的问题；验证和测试集使用由场景索引确定的均匀随机抽样，保留原始病害比例并使 mIoU、ECE 与 bootstrap 可复现。旧 `.npz` 若缺少当前特征 schema，会从 `XYZ + intensity` 重算特征，历史标签派生通道不会进入新训练。

### Backbone 与 mHC

可选骨干：

| Backbone | 特点 | 适用判断 |
| --- | --- | --- |
| `swin3d` | 窗口注意力点云 Transformer | 表达能力强，但显存压力更高 |
| `pointmamba` | Morton 顺序点序列扫描混合 | 更轻量，适合当前二分类快速实验 |

mHC 默认开启，用 Sinkhorn 风格通道混合增强深层特征流动；消融时可加 `--no_mhc`。

### 损失与优化

当前训练目标：

```text
Loss = FocalLoss + DiceLoss + Soft EdgeLoss
```

- `FocalLoss`: 缓解背景点远多于病害点的问题。
- `DiceLoss`: 强化小目标、稀有病害区域。
- `Soft EdgeLoss`: 在全部有效点的 BEV 网格上比较病害边界；标签只作为监督目标，不作为模型输入或筛选掩码。
- `val_mIoU`: macro mIoU，排除背景类，与评估报告口径对齐。

优化器默认是 `Muon + AdamW` 混合：

- 二维矩阵参数使用 Muon。
- bias、norm、标量等一维参数使用 AdamW。
- 如果当前 PyTorch 环境没有 `torch.optim.Muon`，使用 `--optimizer adamw`。
- 默认学习率按优化器区分：Muon 为 `1e-2`，AdamW 为 `1e-3`；可用 `--lr` 覆盖。Muon 的二维矩阵更新与 AdamW 的 bias/norm 更新保持分组。

### 二分类训练

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/credibility_v1_5k `
  --label_stage binary `
  --backbone swin3d `
  --optimizer muon `
  --batch_size 4 `
  --max_points 2048 `
  --embed_dim 48 `
  --depths 1 1 2 1 `
  --num_heads 3 3 6 6 `
  --window_size 32 `
  --max_epochs 5 `
  --num_workers 4 `
  --precision 16-mixed `
  --auto_class_weights `
  --metric_min_support 500
```

快速诊断：

```powershell
python roadmc/scripts/quick_diagnose.py `
  --binary `
  --backbone pointmamba `
  --steps 200 `
  --batch_size 2 `
  --max_points 1024 `
  --binary_class_weights 1.0,3.0
```

### 切回 38 类

二分类稳定后，用二分类 checkpoint 初始化 backbone，分类头重新训练为 38 类：

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/synthetic_output `
  --pretrained_binary ./lightning_logs/version_X/checkpoints/best.ckpt `
  --backbone pointmamba `
  --optimizer muon `
  --batch_size 2 `
  --max_points 2048 `
  --max_epochs 50 `
  --num_workers 4 `
  --precision 16-mixed
```

直接训练 38 类时，去掉 `--binary` 即可：

```powershell
python roadmc/train.py baseline --data_dir ./data/synthetic_output --backbone pointmamba
```

---

## 评估

38 类 checkpoint 可使用：

```powershell
python roadmc/evaluate.py `
  --checkpoint ./lightning_logs/version_X/checkpoints/best.ckpt `
  --data-dir ./data/credibility_v1_5k `
  --label-stage binary `
  --max-points 2048 `
  --scan-binary-thresholds `
  --threshold-calibration-scenes 170 `
  --bootstrap-samples 500 `
  --output-json ./eval_report.json
```

二分类阶段主要看训练日志中的 `val_mIoU`，或用 `quick_diagnose.py` 做短跑检查。

---

## 数据格式

每个场景为一个 `.npz` 文件：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `points` | `(N, 3)` | XYZ 点坐标 |
| `labels` | `(N,)` | 38 类标签，二分类训练时动态折叠 |
| `feats` | `(N, 3)` | 归一化强度、PCA 曲率、有符号局部高度残差（`roadmc.observable_features.v1`） |
| `normals` | `(N, 3)` | 法向量 |
| `pavement_type` | 标量 | `asphalt`、`concrete` 或 mixed 生成结果 |
| `feature_schema` | 标量 | 模型输入特征契约；正式训练必须为 `roadmc.observable_features.v1` |

---

## 38 类标签

`0` 为背景。`1-20` 为沥青路面病害，`21-37` 为水泥混凝土路面病害。

| ID | 类别 |
| --- | --- |
| 0 | Background |
| 1-8 | 裂缝类：龟裂、块裂、纵向裂缝、横向裂缝，含轻重程度 |
| 9-10 | 坑槽 |
| 11-12 | 松散 |
| 13-14 | 沉陷 |
| 15-16 | 车辙 |
| 17-18 | 波浪拥包 |
| 19 | 泛油 |
| 20 | 沥青修补 |
| 21-22 | 水泥板破碎 |
| 23-24 | 水泥裂缝 |
| 25-26 | 板角断裂 |
| 27-28 | 错台 |
| 29 | 唧泥 |
| 30-31 | 边角剥落 |
| 32-33 | 接缝损坏 |
| 34 | 坑洞 |
| 35 | 拱起 |
| 36 | 露骨 |
| 37 | 水泥修补 |

---

## 项目结构

```text
roadmc/
  data/
    dataloader.py
    real/dataset.py
    synthetic/
      config.py
      generator.py
      primitives.py
  models/
    attention/window_attention.py
    backbone/
      swin3d.py
      pointmamba.py
    gan/
      generator.py
      discriminator.py
    mhc/
      mhc.py
      spectral_analysis.py
    model_pl.py
  scripts/
    generate_synthetic.py
    expand_synthetic_dataset.py
    quick_diagnose.py
    grid_search_binary.py
  train.py
  evaluate.py
readmeimage/
  model_architecture.png
```

---

## 当前实验路线

1. 冻结当前二分类 checkpoint 与独立阈值，先完成独立合成测试集。
2. 对 Swin3D、PointMamba、mHC 和 Muon/AdamW 做受控消融，统一使用全局混淆矩阵和 scene-block bootstrap 统计。
3. 按 `binary -> four -> eight -> full38` 课程迁移训练，并分别记录每一阶段的类别支持度和 per-class IoU。
4. 用带元数据的真实点云建立可复现的传感器校准与域差基线。
5. 在真实标签可用后，再验证域随机化、GAN 或无监督域适配是否带来稳定收益。

---

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

> [English](README.en.md)

</div>
