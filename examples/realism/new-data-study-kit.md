# KIT 正常混凝土研究数据与复用说明

本页记录本机实际运行的 KIT 研究协议、保存产物和可复用命令。它不改变生产默认参数，也不把受限内部对照升级为真实道路标定。科学状态保留 `not_calibrated / not_evaluated`。

复现范围说明：本页涉及的 `replay_extended_candidate.py` / `replay_phase2_postlint.py` 是历史仓库外工作脚本，没有随本次发布。相关代码块仅记录历史运行，替换路径不能使其自动可运行。须取得匹配的冻结源码、配置和收据后才能重放；公开CLI及其`--help`与这些wrapper分开理解。文中`tests/`测试命令也属于历史本地检查，本次未发布其全部测试文件，不作为干净检出的验证命令。

## 数据身份、单位和处理链

数据来源为 `10.35097/ku8qs1jt347zxyeq`，论文为 `10.3390/data11100255`。本机原 TAR 位于 `D:\path\to\data\10.35097-ku8qs1jt347zxyeq.tar`，原件未改。

- 19 份 PLY 对应 16 个物理 patch、3 份重复扫描、4 个地点、2 条公路。重复扫描不是新 patch 或独立道路。
- `L1/L2` 是地点；`M1` 为应急车道，`M2/M4` 为第一/第二车道右轮迹，`M3/M5` 为车道中心。`L/R` 为同一测区左右半区。
- A5 是正常露骨料工艺面；A6 的 M2/M3 为拖布面，M1/M4/M5 为扫帚面。“露骨料”工艺名称不能直接转成露骨料病害标签。
- ELA CSV 明确使用位置 mm、高度 µm。32 份 CSV 每份有两条 800 点圆环剖面，间距 0.5 mm、圆周 400 mm；成对剖面不作为独立道路。
- PLY 和论文没有直接声明导出坐标单位。`1 源单位 = 1 mm` 由论文名义 20×30 cm 区域及坐标范围交叉支持，仍是推断。原生读取器保留 source units；所有转换成米的网格结果必须带条件尺度标记。
- 作者做过清扫、遮阴、喷消光剂、校准、裁剪和孤立片去除。ELA 被作者说明为 ISO 13473-1:2019 滤波，A5 附带报告有 2004 标头差异。本轮没有恢复滤波前信号或认证 ISO 滤波实现。
- 原 ETD 换算混杂、砂铺法直径列存在单位/版本差异，均未用作本轮拟合目标。

## 已保存的三个纹理阶段

根目录：`D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit`。

|阶段目录|内容|冻结文件与实际产物|
|---|---|---|
|`ela_attempt01`|有明确单位的一维环剖面对照；生产真实路径及细/默认网格敏感性；L1 有限选型|`protocol.json`、`results.json`、`profiles_si.npz`、三份 `candidate_*.json`、中文说明/图、`all_metric_changes.csv`|
|`mesh2d_attempt01`|19 网格条件尺度二维谱、方向和 ACF；144 个有限 Gaussian 网格候选|`mesh2d_protocol.json`、`grid_inputs.json`、`results.json`、候选 JSON、方向图与退步说明|
|`mesh2d_extension_attempt01`|L1 更低 H、聚合二维统计谱、聚合分位数及 IAAFT 边际候选|`mesh2d_extension_protocol.json`、`input_freeze.json`、`training_statistics.npz`、候选 JSON、`generated_seed2601.npz`、`iaaft_convergence.json`、中文说明/图|

所有协议均在对应数值分析/扩展选择前冻结，并绑定输入和代码 SHA256。ELA/二维参数和种子是显式研究选择；候选不是默认生成器设置。

ELA 主带为共同的 20–500 cycles/m 诊断带，另存不加额外频带及 20–80 cycles/m 敏感性。合成场按相同 800 点、400 mm 圆环采样，半径为 `400/(2π) mm`。真实环的相位、精确中心和配准未知，合成左右中心只是固定选择。圆环频谱是沿弧长的 **1D path PSD**，不能当成二维径向 PSD，也不能据此识别道路方向。

生产默认 5 mm 网格的轴向 Nyquist 为 100 cycles/m。高于其分辨能力的重采样成分不当作真实可表达纹理；主比较另用显式细网格研究基线。生产默认的 MPD/RMS 比例 0.81 原有 5 mm 网格口径，在细网格下未重新认证为物理 MPD 标定。

二维阶段使用中央 180×260 源单位、间距 0.5 源单位的网格。95% 支持率规则预先固定：19 份中 17 份合格；`A5_L2_M2_rep` 与 `A5_L2_M3` 不合格。去掉重复扫描，实际分组评分使用 15 个 patch，即 L1=8、L2=7；三类 L1 数量为 3/2/3，L2 为 2/2/3。

二维名义尺度分析统一去平面、20–500 cycles/m 带、Hann 二维功率谱、24 个径向壳层、频率方向角矩和沿 x/y 的循环 ACF。0.9/1/1.1 倍尺度假设会同时改变坐标、高度、采样物理间距，重算固定物理频带和 50 mm ACF；不是只改频轴名称，也不是测量置信区间。

扩展只读取合格非重复 L1 patch 来估计 24×12 箱的二维 PSD 与 1001 个聚合分位数，丢弃实测 phase 和空间布局，用新随机 phase 生成场。旧 `spectral_synthesis.match_marginal_spectrum` 未修改；它最终优先保谱，边际近似并记录每次迭代。所选候选的 12 次评价均达到冻结 0.02 容限，不等于获得真实道路验证。

## L1/L2、退步和科学边界

L1 仅开发，L2 是已曝光的内部地点 benchmark，不称未见或独立道路 test。评价 seed2601–2604 与训练 seed1601–1604 分开，但已经用于之前阶段，也不称未曝光随机留出。

各阶段负结果都保留：ELA 的 90 项比较有 10 项退步，二维首轮有 4 项跨尺度退步；扩展的 270 项基线/尺度/指标比较有 62 项退步。计数是描述，不是显著性检验。扩展后的名义尺度 L2 摘要：

|表面|首轮→扩展径向 PSD log10 MAE|首轮→扩展标准化分位 MAE|
|---|---:|---:|
|露骨料|0.254804→0.214283|0.033191→0.052411，退步|
|拖布|0.165354→0.056094|0.053201→0.018102|
|扫帚|0.140109→0.049793|0.096562→0.065969|

扫帚的方向 ACF 比首轮退步；拖布首轮极小的 RMS 误差也增大。Gaussian 和非高斯候选都保存，不能因部分指标改善只展示一个“胜者”。这些结果不支持按 L2 重新挑候选或将新设置推广为生产默认。

二维真值是 source XY 垂直残差经过投影、裁剪、成格和填孔后的观察统计，**不是 latent physical GT**。全15个评分 patch 没有统一刚性转到真实切平面，也未以外部测量验证 source x/y 与道路轴的配准。频率能量方向不等于沟槽方向。

缺孔仅为 FFT 作固定插值，mask/count 保留。真实网格有缺孔，合成完整；全局95%支持率并不能保证无频谱偏差。点平均足迹、喷剂和仪器/软件滤波也未完整前向匹配。真实有限 patch 的循环 ACF 是诊断算子，不表示道路自然周期。

## 重复扫描与 common-support 修复

原网格/重复运行是 `mesh_attempt02`；四个已曝光 L1 M2 网格的刚性转正敏感性是 `tangent_repeat_attempt01`。变换定义为查值：

`moving(R(angle) @ reference_xy + translation)`。

它不是直接把 moving 点变到 reference 的参数；逆点变换为 `R.T @ (moving_xy - translation)`。刚性转正的正交基和 origin 已核对；其后仍重新成格、中心化和去平面，阶段差额不能全归因于纯刚性转正。

修前 `fit_before/fit_after/evaluation_before/evaluation_after` 各自取有效支持，分母不同。这些字段原样保留。新增 `common_support_repeat_scores` 及 `result.common_support`：

- FIT 和 EVAL 各自统一取 `finite(reference) & finite(before) & finite(after)` 的共同支持。
- 前后高度 offset 都只在共同 FIT 支持估计，评价高度不用于 offset。
- 前后指标使用同一批评价点；不足100个共同 FIT 点时不编造 offset/误差。
- 不修改 XY 变换，不重新优化配准，不重读 TAR，不重跑19网格。

已有审核补算为权威；修后新增函数和实际 CLI 与四组审核指标完全一致，最大数值差为0，原 aligned 数组逐数组重放一致：

|运行|重复对|共同评价点|共同支持 RMSE before→after，源单位|
|---|---|---:|---:|
|原坐标|A5_L1_M2|74782|0.346721312→0.298564098|
|原坐标|A6_L1_M2|80402|0.111457306→0.036579322|
|刚性转正后|A5_L1_M2|74796|0.343026236→0.294100957|
|刚性转正后|A6_L1_M2|80390|0.111495004→0.036542699|

这些是同片诊断，不能称仪器精度。A5 重复扫描仍明显不稳；配准、裁剪、成格及喷剂的影响不能直接从模型误差中扣除。

**分区限定：**只在已经预处理的 reference 网格登记 FIT/EVAL 条带。bbox 和 robust plane 用过全 patch；moving 搜索包络也未实施双域遮蔽。原测试只保证 reference 评价高度不直接进入该网格层的优化。不能扩写成双域严格留出、raw end-to-end holdout 或独立验证。

## 修前源码快照与历史运行绑定

修前3文件保存在：

`C:\path\to\research-work\data_study_2026-10-16\kit_source_before_common_support_fix`。

`manifest.json` 记录源/快照哈希及旧协议匹配：修前 `kit_concrete.py`、`diagnose_kit_repeats.py` 均与 `mesh_attempt02/protocol.json` 的 code SHA256 相同；测试文件没有列入该旧协议，不能声称旧协议绑定了其版本。

common-support 修复后，当前这两份源码的哈希与旧运行协议不同。旧结果及协议继续绑定快照，不改写成新源码产物。**这里对旧科学运行仅提供从已保存 NPZ 和冻结参数补算/重放的方法，不以当前源码声称重做了原始19网格运行。**新的补算输出有自己的协议、输入/代码哈希和收据，并明确记录旧代码与当前代码不相同。

## 已实际执行的补算与重放命令

以下命令从 `C:\path\to\roadmc` 执行过。结果目录禁止覆盖；再次复用时将 `--output` 改为新的目录名。

```powershell
.venv\Scripts\python.exe -m roadmc.scripts.diagnose_kit_repeats `
  --rescore-common-support D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit\mesh_attempt02 `
  --output D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit\common_support_fix_native_attempt01

.venv\Scripts\python.exe -m roadmc.scripts.diagnose_kit_repeats `
  --rescore-common-support D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit\tangent_repeat_attempt01 `
  --output D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit\common_support_fix_tangent_attempt01
```

该入口先冻结补算输入/代码，验证源 result 与 protocol 哈希，核对保存的 aligned 能由冻结参数精确重放，再计算共同支持指标。输出 `results.json` 保留 `original_own_support`，另存 `common_support`；`completion_receipt.json` 核对源输入未变。`common_support_fix_verification.json` 记录四组与审核权威值的逐项比较。

ELA 可复用函数在 `roadmc.data.synthetic.concrete_reference`：`load_ela_csv` 明确转 SI；`sample_rings` 禁止外推；`preprocess_ring/describe_ring` 记录诊断处理。原 `benchmark_concrete_reference --freeze/--run` 和二维工作脚本对应各自有限开发实验，现有封存目录不覆盖，不为本次 common-support 修复重复搜索。

统计候选的工作脚本在 `C:\path\to\research-work\kit_execution_2026-10-16`。下列 CLI 实际重放了扫帚候选，与 `generated_seed2601.npz` 保存场完全相同，最大逐数组误差为0；凭据见 `mesh2d_extension_attempt01/replay_verification.json`。同样需要为再次执行选择新的输出文件名。

```powershell
.venv\Scripts\python.exe ..\work\kit_execution_2026-10-16\replay_extended_candidate.py `
  --config D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\kit\mesh2d_extension_attempt01\candidate_broom_finish_new_selected.json `
  --seed 2601 `
  --output ..\work\kit_execution_2026-10-16\replayed_broom_seed2601.npz
```

输出 `height_m` 为360×520、x-major研究网格，另含米制 x/y 轴和条件单位/seed/统计目标哈希/IAAFT审计的 sidecar。候选 JSON 和 `training_statistics.npz` 是独立复用输入，不需要实测空间像素或 phase。代码绑定以 `input_freeze.json` 为准，统计文件哈希由重放入口核对；这仍是显式 opt-in 研究候选，不是默认完整路面场景。

## 检查范围

修后 `tests/test_kit_concrete.py` 的5组契约覆盖解析/归档、原生单位/支持、查值变换、共同支持及 FIT-only offset、保存数组补算保留旧字段。4组实际保存数据另与审核权威结果比较，未重新拟合；Ruff通过。没有把这些受影响检查称为完整历史测试或真实道路验证。

路径说明：本页的 `C:\path\to` / `D:\path\to` 为占位根目录，请替换为自己的源码、数据和历史研究收据位置。文中历史数值是已有研究记录，不是运行此页即自动获得的结果。
