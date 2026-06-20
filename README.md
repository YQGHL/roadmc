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

---

## 模型训练：结构与策略

训练模块把合成点云作为监督数据，输出逐点 logits。它既可以做二分类，也可以做 38 类细分。

### 数据加载

`SyntheticPointCloudDataset` 从 `.npz` 文件读取：

- `coords`: 归一化 XYZ 坐标。
- `feats`: 强度、曲率、裂缝边界距离。
- `labels`: 38 类标签；二分类时动态折叠为 `labels > 0`。
- `valid_mask`: padding 后忽略无效点，避免 loss 和 mIoU 被填充点污染。

采样逻辑会优先保留病害点，降低 batch 中全背景样本过多的问题。

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
- `Soft EdgeLoss`: 用可微边缘约束辅助裂缝/边界区域。
- `val_mIoU`: macro mIoU，排除背景类，与评估报告口径对齐。

优化器默认是 `Muon + AdamW` 混合：

- 二维矩阵参数使用 Muon。
- bias、norm、标量等一维参数使用 AdamW。
- 如果当前 PyTorch 环境没有 `torch.optim.Muon`，使用 `--optimizer adamw`。

### 二分类训练

```powershell
python roadmc/train.py baseline `
  --data_dir ./data/synthetic_output `
  --binary `
  --backbone pointmamba `
  --optimizer muon `
  --batch_size 2 `
  --max_points 2048 `
  --max_epochs 20 `
  --num_workers 4 `
  --precision 16-mixed `
  --binary_class_weights 1.0,3.0
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
  --data_dir ./data/synthetic_output `
  --output_json ./eval_report.json
```

二分类阶段主要看训练日志中的 `val_mIoU`，或用 `quick_diagnose.py` 做短跑检查。

---

## 数据格式

每个场景为一个 `.npz` 文件：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `points` | `(N, 3)` | XYZ 点坐标 |
| `labels` | `(N,)` | 38 类标签，二分类训练时动态折叠 |
| `feats` | `(N, 3)` | 强度、曲率、裂缝边界距离 |
| `normals` | `(N, 3)` | 法向量 |
| `pavement_type` | 标量 | `asphalt`、`concrete` 或 mixed 生成结果 |

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

1. 扩大合成数据集到约 5000 个场景。
2. 先把二分类 `val_mIoU` 提升到 `0.7-0.9` 区间。
3. 对比 `pointmamba` 与 `swin3d`，优先选择显存稳定、收敛更快的骨干。
4. 二分类稳定后，用 `--pretrained_binary` 迁移到 38 类细分。
5. 再考虑 GAN 域适配和真实点云加载。

---

## License

MIT. See [LICENSE](LICENSE).

<div align="center">

> [English](README.en.md)

</div>
