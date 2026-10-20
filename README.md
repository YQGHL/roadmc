# RoadMC

[English](README.en.md) | 简体中文

RoadMC 是一个用于道路表面点云研究的实验性代码库，重点是程序化生成带逐点标签的道路表面、检查合成场景的几何与材料变化，并为点云分割模型提供训练入口。项目也包含固定真实域评估协议的工具。合成场景可用于受控实验和实现诊断；它们不能代替真实道路数据，也不能证明模型在真实道路上的效果。

## 能力与实验边界

- 基于 NumPy/SciPy 的道路表面与病害场景生成，支持沥青、水泥、网格采样及可选观测模型。
- 可选的设计面、粗糙度、纹理三层表面表示，以及显式参数驱动的程序化纹理。
- 基于冻结 `SurfaceTile` 和显式 JSON 计划的确定性重放，提供裂缝、坑槽、松散和修补的受控原型。
- PyTorch 训练入口，以及读取真实点云、按道路划分、从真实标定样本拟合并冻结评估协议的工具。

标签定义见 [`labels.py`](roadmc/data/synthetic/labels.py)，共 38 类（含背景），不等于所有病害类别都已有完整、经验证的物理生成模型。冻结重放 v1 支持沥青线裂缝、坑槽、松散、修补，以及水泥板裂缝、点状坑洞和修补；v2 是沥青专用原型，支持显式线裂缝、龟裂/块裂图结构及材料覆盖顺序，并未覆盖全部 taxonomy。新纹理和重放参数仍需实测标定。目前已有公开纹理矩阵和立体重建点云可用于受限形态对照，但其单位、设备信息或道路独立性尚不满足真实域标定和独立验收要求，因此项目没有可报告的“极小域差距”结论。

## 安装

项目要求 Python 3.11 或更高版本。完整项目依赖（包括训练用 PyTorch）见 [`pyproject.toml`](pyproject.toml)。只运行 NumPy/SciPy 合成和表面工具时，可在 Windows PowerShell 中从仓库根创建轻量环境；仓库根目录可直接导入 `roadmc`，不需要安装整个项目：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib pillow
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:PYTHONIOENCODING = 'utf-8'
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

当前 line-profiler 是高度场插值与工程噪声模型，没有实现光斑积分、光学三角测量或几何自遮挡。采样间距不等于设备精度，入射角参数只影响噪声代理。输出审计明确记录这些限制；生产路径保留采样器的 XY 支持标签和插值潜在法线，强度与曲率按 XY 近邻传递，避免 Z 噪声改变点的表面归属。

## 可选的采样与纹理形态实验

以下入口用于受控诊断，不修改生产默认配置。网格实验保存同一 0.5 mm 母场，比较 5、2、1、0.5 mm 源网格及两种观测步距；粗网格不做隐式几何平滑，保留混叠反例：

```powershell
python -m roadmc.scripts.diagnose_sampling_resolution --output-dir output/resolution-check
python -m roadmc.scripts.calibrate_texture_shape `
  --archive path/to/rsta20220166_si_001.zip --output-dir output/texture-shape-check
```

网格实验目录须为空或不存在，纹理实验目录须不存在。1 m 网格实验约有 400 万个源格点，额外诊断数组还会占用内存。采样点高度误差为零不等于连续纹理没有混叠；应同时检查横纵频率、源网格频带和输出步距。

纹理实验针对 Royal Society 发布的 DAC、MS、OGFC、SMA 四类沥青表面矩阵，冻结训练/文件留出划分，排除已探索文件及内容重复项。只用训练集选择频谱和非高斯目标矩，再报告留出文件的 PSD、自相关、标准化 W1 和实际输出矩。XY 使用论文报告的 0.5 mm 间距；CSV 的 Z 单位和处理阶段仍未确认，因此这些结果属于归一化形态比较，不能用于绝对高度或独立道路真实性验收。

冻结重放 v2 的[病害材料样例](examples/surface_replay_v2_materials.json)增加可选椭圆坑槽，以及坑槽、松散、修补的 `retain`、`iid`、`procedural` 纹理模式。椭圆 `contour` 与旧 `radius_m` 二选一，`texture` 与旧 `exposed_rms_m` 二选一。样例基底需覆盖 x=[-0.6, 0.6] m、y=[-0.45, 0.45] m；`retain` 是保留当前纹理的对照选项。v2 暂限沥青，参数仍须显式声明为未标定假设。v1 有效旧计划的几何与随机序列保持兼容；线激光插值修正会改变观测结果。

纹理还可显式使用 `spectrum_profile` 的 `frequency_cycles_m` / `relative_psd` 数组代替 `hurst`，以统计谱包络生成全新随机表面；不会读取实测相位或拼接实测纹理。非高斯设置可选 `spectral_matching={"iterations":50,"tolerance":1e-5}`，迭代约束频谱与高度分布，保存实际迭代次数及分布残差。次数和容差是数值设置。最终优先保持程序场频谱，边际分布不保证精确匹配。现有文件对比中，谱包络降低了频谱和自相关误差，但 W1 未优于此前非高斯候选，因此没有自动替换默认方法。

v2 坑槽可选 `profile={"kind":"tilted_power_bowl","edge_power":1.0,"tilt_major":0.5,"tilt_minor":0.0}`，在相同足迹内调整坑壁及最深点位置；示例数值属于假设。深度按连续解析峰值归一，不按采样格点最大值重新缩放。v2 元数据还记录当前材料损失场的面积、体积和最终事件归属积分；修补清除的旧损失不计入最终体积，纹理起伏和隐藏层不计入该量。这些是合成器内部量，不是实测材料损失。

以下命令要求显式输入路径，输出目录须不存在。纹理对照复用既有冻结清单，已分析过的测试文件必须称为“已见文件基准”；坑槽对照只检验同视图的空间块插值，不能解释为独立坑槽泛化。

```powershell
python -m roadmc.scripts.benchmark_texture_envelopes `
  --archive path/to/rsta20220166_si_001.zip `
  --split-manifest output/texture-shape-check/split_manifest.json `
  --prior-config output/texture-shape-check/fitted_configurations.json `
  --output-dir output/texture-envelope-check

python -m roadmc.scripts.diagnose_pothole_profiles `
  --dataset-root path/to/dataset3 --output-dir output/pothole-profile-check
```

## 可复现的真实性研究

统一入口按清单核对数据用途、单位、文件哈希和已暴露状态，再执行训练内部纹理选择、冻结基准、病害材料对照与原生单位坑槽诊断。它保存实际执行源码、随机种子、分阶段日志和产物哈希。准备方法见[研究工作流](examples/realism/study-workflow.md)，数值协议见[示例配置](examples/realism/study-protocol.json)。原始数据和实验大文件放在仓库外。

```powershell
python -m roadmc.scripts.run_realism_study `
  --manifest path/to/verified-manifest.json `
  --protocol examples/realism/study-protocol.json `
  --output-dir path/to/new-study
```

续跑时在同一命令后加 `--resume`；输入、代码、依赖阶段或产物改变时，不复用过期结果。用 `--stages admission` 可只核查清单，其他阶段名见 `--help`。纹理选择只使用原 TRAIN 文件，在三个主要指标均不劣于旧候选时才推荐新候选，否则回退；已见 TEST 只做冻结后的对照。推荐配置不会自动替换生产默认。

v2 椭圆 `contour` 可加 `radial_harmonics={"cos":[0.08,-0.04,0.03],"sin":[-0.03,0.02,0.01]}`，以3、4、5阶径向项表达不规则坑缘，并在连续域保持椭圆面积。几何损失、足迹、材料更新使用同一轮廓。系数是工程假设，不能称为实测分布；省略或全零时保留旧轮廓。

新增的显式选项包括：不重新拟合的 `pothole_outline_fairness` 阶段、v2 坑壁/坑底 `material_partition`，以及保留未知单位和辅助列的研究数据导入。用法与组合检查见[工程扩展说明](examples/realism/engineering-extensions.md)。导入束只用于原生/归一诊断，不自动成为米制训练输入或独立真实验收数据。

KIT 正常混凝土、SUD 原生 LAS 字段和 SHREC 视频已有专用读取与受限对照工具，见[新增数据研究说明](examples/realism/new-data-study.md)。ELA 圆环单位明确；PLY 尺度推断、SUD 未知语义码和 SHREC 压缩视差分别保留门槛，不自动成为米制病害真值或独立道路验收输入。混凝土统计谱候选可显式复用，生产默认保持原设置。

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

重放与评估工具支持受控程序化实验、显式裂缝图结构和一致性检查；公开数据的受限形态比较不构成独立真实域验收。v2 已有可选不规则坑缘原型，但规则高度场尚不能表示多层坑壁、悬垂或裂缝内部体积结构，也未实现分段设备观测、材料状态和传感器标定。

## 研发数据来源与引用声明

感谢公开数据集的作者和维护者。RoadMC 研发阶段曾使用下列数据的有限子集进行纹理描述、原生字段检查、轮廓或观测敏感性诊断；这不表示已经用这些数据完成训练、现场标定或独立真实域验收。

| 数据来源 | 研发中的实际用途与限制 |
|---|---|
| [Pan 等：沥青纹理补充矩阵](https://doi.org/10.1098/rsta.2022.0166) | 归一化纹理形态与频谱比较；Z 单位/处理链未闭合，已暴露 TEST 不称盲测 |
| [Fan 等：立体坑槽数据](https://github.com/ruirangerfan/stereo_pothole_datasets) | dataset3 有限视图的边界、缺测和坐标诊断；不作为独立激光真值 |
| [KIT 混凝土路面测量](https://doi.org/10.35097/ku8qs1jt347zxyeq) | 正常路面 ELA/PLY 和同片重测比较；不是病害标签真值 |
| [SUT-Crack](https://doi.org/10.17632/gsbmknrhkv.1) | 本地 130 张 mask 与 3 张原图的二维探索；不是 130 对完整分割样本 |
| [SUD-ROAD](https://github.com/msqiu/SUD-Road) | LAS 原码与局部几何检查；本地字节码映射和物理单位仍未核实 |
| [Pothole Mix / SHREC 2022 V2](https://doi.org/10.17632/kfth5g2xk3.2) | RGB/视差/自动视频 mask 的受限轮廓分析；不是米制三维人工真值 |
| [实验室纹理/摩擦数据](https://doi.org/10.17632/kkcztnxzph.1)、[现场沥青扫描](https://doi.org/10.5281/zenodo.20606458)、[成对 MLS 扫描](https://doi.org/10.5281/zenodo.19267189) | 选定样本的描述、格式和观察支持诊断；不宣称全库使用或损伤材料机理标定 |

其他选定表面、M2S-RoAD 副样本及未完全追溯来源见 [数据来源明细](docs/DATASETS.md)，可导入的引用在 [datasets.bib](docs/datasets.bib)。使用数据或发表派生结果时，请引用对应原始数据记录及作者要求的论文，保留版本、许可、处理与划分记录。项目 MIT 许可证仅覆盖本项目代码，不替代第三方数据许可；本仓库不随此声明重新分发原始数据。

## 当前代码与研究状态

生产生成/重放代码与研究用原型分开维护。核心病害与原生数据诊断代码位于本分支；主干是否已包含这些改动应以 PR [#15](https://github.com/YQGHL/roadmc/pull/15) 的实际合并状态为准。H1 材料历史、占据域空腔和 A/B/C 系统对照仍是仓库外隔离研究组件，不是当前默认生成器或 replay schema 的原生功能。详见 [研究状态与集成边界](docs/RESEARCH_STATUS.md)。

## 许可证

本项目按 [MIT License](LICENSE) 发布。
