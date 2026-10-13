# RoadMC

[English](README.en.md) | 简体中文

RoadMC 是一个用于道路表面点云研究的实验性代码库，重点是程序化生成带逐点标签的道路表面、检查合成场景的几何与材料变化，并为点云分割模型提供训练入口。项目也包含固定真实域评估协议的工具。合成场景可用于受控实验和实现诊断；它们不能代替真实道路数据，也不能证明模型在真实道路上的效果。

## 能力与实验边界

- 基于 NumPy/SciPy 的道路表面与病害场景生成，支持沥青、水泥、网格采样及可选观测模型。
- 可选的设计面、粗糙度、纹理三层表面表示，以及显式参数驱动的程序化纹理。
- 基于冻结 `SurfaceTile` 和显式 JSON 计划的确定性重放，提供裂缝、坑槽、松散和修补的受控原型。
- PyTorch 训练入口，以及读取真实点云、按道路划分、从真实标定样本拟合并冻结评估协议的工具。

标签定义见 [`labels.py`](roadmc/data/synthetic/labels.py)，共 38 类（含背景），不等于所有病害类别都已有完整、经验证的物理生成模型。冻结重放当前仅支持：沥青线裂缝 5–8、坑槽 9–10、松散 11–12、修补 20；水泥板裂缝 23–24、点状坑洞 34、修补 37。它不模拟沥青龟裂/块裂，也不覆盖 taxonomy 中所有病害。新纹理和重放参数仍是工程假设，尚未由达到毫米精度的真实表面数据标定。当前没有可用于真实域标定或独立验收的合格真实样本，因此项目没有可报告的真实域差距或“极小域差距”结论。

## 安装

项目要求 Python 3.11 或更高版本。完整项目依赖（包括训练用 PyTorch）见 [`pyproject.toml`](pyproject.toml)。只运行 NumPy/SciPy 合成和表面工具时，可在 Windows PowerShell 中从仓库根创建轻量环境；仓库根目录可直接导入 `roadmc`，不需要安装整个项目：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:ROADMC_GENERATOR_NO_TORCH = '1'
.venv\Scripts\python.exe -m roadmc.scripts.generate_synthetic --help
```

`ROADMC_GENERATOR_NO_TORCH=1` 避免生成器导入 PyTorch；它不提供训练能力。完整训练安装可用 `uv sync`（`pyproject.toml` 中的 PyTorch 索引配置由 uv 使用）；若使用 pip，请按目标平台配置 PyTorch wheel 来源，再安装项目依赖。LAS/LAZ 与 PLY 的可选依赖见 `real` extra，Open3D 相关功能见 `pcd` extra。

## 快速开始：生成合成数据

以下 PowerShell 示例在仓库根目录生成少量沥青场景，关闭观测模拟以保持生成简单。网格间距参数以米指定，但通用生成 CLI 默认 `normalize=True`，因此场景 NPZ 中的点坐标默认归一化到单位球，不应直接当作米制坐标。文件包含 `coordinates_normalized` 等元数据以说明变换；用于度量评估前需按记录的变换还原坐标。专项 `capture_surface_geometry` 流程固定 `normalize=False`，其 `SurfaceTile` 与捕获坐标保持米制。场景文件为压缩 NPZ，数据集根目录另有 `metadata.json`。

```powershell
$env:ROADMC_GENERATOR_NO_TORCH = '1'
python -m roadmc.scripts.generate_synthetic `
  --train-count 4 --val-count 2 `
  --output-dir output/readme-demo/synthetic `
  --seed 73 --surface-grid-spacing 0.02 `
  --sensor-output-points 2048 --width 3 --length 4 `
  --pavement asphalt --observation-model off
```

场景 NPZ 包含 `points`、`labels`、`feats`、`normals` 及坐标变换、采样分辨率等字段；具体字段以生成器版本和配置为准。`labels.py` 中的 0 表示背景，1–20 为沥青 taxonomy，21–37 为水泥 taxonomy。taxonomy 编号本身不保证某个场景包含该类，也不代表其合成形态已通过真实数据验证。

## 可选表面流程

以下命令捕获一块纯合成的三层表面，生成 `SurfaceTile`，再以显式设置替换纹理层并运行双材料冻结重放 demo。数值仅用于演示接口，不是实测范围或推荐标定值。所有产物写入 `output/readme-demo/`，不会读取或复制真实纹理 patch。

```powershell
$env:ROADMC_GENERATOR_NO_TORCH = '1'
python -m roadmc.scripts.capture_surface_geometry `
  --output-dir output/readme-demo/base --label 6 --seed 73 `
  --width 2 --length 1 --spacing 0.01 --surface-layers
```

```powershell
python -m roadmc.scripts.generate_texture_surface `
  --base output/readme-demo/base/surface_tile.npz `
  --settings examples/realism/asphalt-texture-settings.json `
  --seed 73 --output output/readme-demo/asphalt-texture/surface_tile.npz

python -m roadmc.scripts.generate_texture_surface `
  --base output/readme-demo/base/surface_tile.npz `
  --settings examples/realism/concrete-texture-settings.json `
  --seed 73 --output output/readme-demo/concrete-texture/surface_tile.npz

python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/asphalt-texture/surface_tile.npz `
  --plan examples/realism/asphalt-plan.json `
  --output output/readme-demo/replay/asphalt/replay.npz

python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/concrete-texture/surface_tile.npz `
  --plan examples/realism/concrete-plan.json `
  --output output/readme-demo/replay/concrete/replay.npz
```

两个计划和两份纹理设置样例位于 [`examples/realism/`](examples/realism/)，计划使用各自合法的材料标签。沥青与水泥的纹理设置目前只改变 provenance 中的 `material_id`；其数值相同，不代表已建模或测得两种材料的外观差异。两个流程复用同一设计面和粗糙度层，是为了隔离受控变量的共基底假设。计划需显式声明材料、道路 ID、事件、seed、参数、标签、来源与假设；每个事件都必须包含 `label`，未知标签写成 `null`，输出值为 `-1`，不会当作背景。基底未知标签也保持 `-1`。计划参数和暴露纹理均为未标定假设。

每个 replay NPZ 保存 `points`、`labels`、`normals`、`coordinates_normalized=False`、`metadata`、`plan_hash`、计划 JSON、米制最终高度、材料权重/纹理、事件 footprint 和基底分层；当前冻结重放不输出强度通道。快速生成器的数据集布局为 `train/scene_*.npz`、`val/scene_*.npz` 和根目录 `metadata.json`；表面例子布局为 `base/`、`asphalt-texture/`、`concrete-texture/`、`replay/<material>/`。

这个快速 demo 使用 10 mm 网格，而沥青裂缝事件宽度为 8 mm；该采样不能解析毫米级裂缝形态，也不构成毫米真值或精度验证。

line-profiler 是可选观测层：仅支持规则且横纵间距相等的单块网格。小型配置样例在 [`examples/realism/line-profiler-config.json`](examples/realism/line-profiler-config.json)，调用时显式指定 seed 和配置：

```powershell
python -m roadmc.scripts.replay_surface `
  --base output/readme-demo/asphalt-texture/surface_tile.npz `
  --plan examples/realism/asphalt-plan.json `
  --output output/readme-demo/replay/asphalt/line-profiler.npz `
  --observation line-profiler --observation-seed 11 `
  --observation-config examples/realism/line-profiler-config.json
```

当前没有分段采样及跨段 profile 去重实现。`off` 模式保留米制网格坐标，不限制横纵间距相等。

## 固定真实域评估

评估入口为 `roadmc.scripts.diagnose_domain_protocol`。协议先按 `road_segment_id` 划分道路，再仅用真实 calibration 样本拟合并保存标准化和核带宽，最后用冻结协议评估候选合成数据与独立真实道路。评估样本清单是 JSON 对象，顶层包含 `samples` 数组；每项提供文件 `path`、`domain`、`role`、`road_segment_id` 和 `material_id`。真实输入还必须有可核验的坐标单位、质量、采样间距、测量精度和来源等元数据。不得将同一路段拆到不同组。

对上面的合成输出，可先做候选输入覆盖诊断。这里的诊断只检查合成输入与可用描述量，不拟合真实基准，也不做验收：

```powershell
python -m roadmc.scripts.diagnose_domain_protocol inspect `
  --manifest examples/realism/synthetic-candidate-manifest.json `
  --output output/readme-demo/candidate-diagnostics.json
```

该清单中的路径相对于清单文件解析，指向上面生成的沥青重放 NPZ。它不包含真实质量核验信息，也不能用于真实标定。

真实数据准备好并通过准入检查后，按以下顺序创建道路级划分、拟合冻结协议、评估清单。`real-manifest.json`、生成的 `split.json` 和 `evaluation-manifest.json` 均须列出真实文件及正确角色；不能用合成样本冒充真实 calibration、validation 或 holdout。道路划分需要至少 25 条独立真实道路，才能为 calibration、real validation、real holdout 各提供至少 5 条道路。

```powershell
python -m roadmc.scripts.diagnose_domain_protocol split `
  --manifest real-manifest.json --output split.json
python -m roadmc.scripts.diagnose_domain_protocol calibrate `
  --manifest split.json --output-protocol fixed-protocol.json
python -m roadmc.scripts.diagnose_domain_protocol evaluate `
  --manifest evaluation-manifest.json --protocol fixed-protocol.json `
  --output evaluation-report.json
```

当前没有满足准入要求的真实扫描，因此上述标定与真实评估命令在现有数据上不会产生有效真实域结论；工具应报告 `not_calibrated` 或 `not_evaluated`，缺少证据的量不会按零差距处理。固定协议按道路分组，避免同一路段同时进入标定和独立验收；数据需先核实单位、采样间距、测量精度、质量与来源。README 给出的阈值仅说明工具当前需要的最低道路数，不是数据质量达标的替代证明。

## 模型训练

训练与合成数据生成是独立流程。训练需要已安装的 PyTorch 及训练依赖；`label_stage` 提供 `binary`、`four`、`eight`、`full38` 训练映射，但这些设置不能证明所有标签都有完整物理生成覆盖。下面是一个短的合成数据 smoke run，不是模型质量评估：

```powershell
python roadmc/train.py baseline `
  --data_dir output/readme-demo/synthetic `
  --label_stage binary --max_epochs 1 --batch_size 2 `
  --max_points 2048 --num_workers 0
```

训练参数、骨干网络、优化器、checkpoint 与运行目录选项见 `python roadmc/train.py --help`。小样本 smoke run 不能用来报告泛化性能；正式实验应使用足够数据、固定数据划分、记录配置与随机种子，并在独立数据上评估。

## 目录与验证

- `roadmc/`：生成器、数据读取、模型、训练与实验脚本。
- `examples/realism/`：可复用的显式 JSON 计划、纹理设置、观测配置和候选清单。
- `output/`：本地命令生成的实验产物；通常不纳入版本控制。

检查脚本语法和代码风格可在仓库根目录运行：

```powershell
python -m py_compile roadmc/scripts/generate_synthetic.py roadmc/scripts/capture_surface_geometry.py roadmc/scripts/generate_texture_surface.py roadmc/scripts/replay_surface.py roadmc/scripts/diagnose_domain_protocol.py
python -m ruff check roadmc
```

重放与评估工具仅支持受控程序化实验与一致性检查，不提供实路自然性、病害全类覆盖、传感器校准或真实域性能证据。重放是规则网格样板，不支持完整裂缝拓扑、不规则坑缘、多层坑壁或分段设备观测。

## 许可证

本项目按 [MIT License](LICENSE) 发布。
