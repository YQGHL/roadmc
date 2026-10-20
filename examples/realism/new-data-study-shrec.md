# SHREC / Pothole-Mix V2 受限复用说明

本研究仅对 Pothole-Mix V2 的 RGB、8-bit disparity 和自动生成视频 mask 做有限视频观察与二维轮廓诊断。数据包位于本机原始归档目录；研究结果保存在 `D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\shrec`。执行源码快照保存在 `C:\path\to\research-work\shrec_execution_2026-10-16\phase1_source_archive`。

复现范围说明：本页涉及的 `replay_extended_candidate.py` / `replay_phase2_postlint.py` 是历史仓库外工作脚本，没有随本次发布。相关代码块仅记录历史运行，替换路径不能使其自动可运行。须取得匹配的冻结源码、配置和收据后才能重放；公开CLI及其`--help`与这些wrapper分开理解。文中`tests/`测试命令也属于历史本地检查，本次未发布其全部测试文件，不作为干净检出的验证命令。

## 已冻结范围

phase1 选20个 mask clip（train 12、valid 8），每段固定抽首帧、floor((N−1)/2)中帧和末帧，共60组观察。按 RGB 颜色距离到 `(255,0,0)` 的欧氏距离≤32作主 mask，固定≤8和≤64做敏感性分析；红色只代表视频 mask 前景码，不赋予 pothole/crack 语义。触边轮廓不补边、不参加拟合。ellipse、harmonic 和训练弧 convex hull 共用相同 held-out 边界像素、同一源图像坐标和最短欧氏距离。

phase2 不再读视频。它从 phase1 NPZ 和已保存拟合参数复核曲线距离，并将 frame 误差按 capture group 等权汇总。65,536点周期曲线 KDTree 加局部有界细化用于ellipse/harmonic；凸包再用解析线段距离对照≤1px采样近似。冻结文件、输入哈希及 phase2 报告在输出目录的 `phase2` 子目录。

此数据的 train/valid 捕获前缀有重叠；capture group 不是独立道路身份。V2视频 mask 自动生成且不保证完美，所以这些数值只描述自动 mask 的二维边界拟合。没有可核实的相机标定或 disparity→metric-Z映射，不转换为米制深度、不用于真实道路几何标定，也不把结果写入 RoadMC 默认参数。

## 安全复现命令

项目 `.venv` 已验证可用：PyAV 19.0.1、pytest 9.1.1。命令从仓库根目录以模块方式运行，并设置 `PYTHONPATH`，确保 CLI 和研究模块来自当前 checkout。

PowerShell 中先进入仓库并准备运行时路径：

```powershell
$repo = 'C:\path\to\roadmc'
$py = Join-Path $repo '.venv\Scripts\python.exe'
$audit = 'D:\path\to\data\RoadMC_public_data\new_data_audit_2026-10-16'
$source = 'C:\path\to\downloads\kfth5g2xk3-2\RGB-D'
$env:PYTHONPATH = $repo
Set-Location -LiteralPath $repo
```

冻结与视频解码必须使用一个全新、空的输出目录。freeze-only 写入 `frozen_protocol.json`；之后正常 run 允许目录中只有这份冻结协议：

```powershell
$out = 'D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\shrec_reproduction_01'
& $py -m roadmc.scripts.diagnose_shrec_video --freeze-only --audit-root $audit --source-root $source --output-dir $out
& $py -m roadmc.scripts.diagnose_shrec_video --audit-root $audit --source-root $source --output-dir $out --protocol "$out\frozen_protocol.json"
```

第二条命令会实际解码冻结样本视频。当前入口在输出目录出现任何其他文件或子目录时拒绝复用；这保护分析 JSON、NPZ、manifest、run/error 日志及图表免于覆盖。`--finalize-only` 与 `--refit-models-only` 也受同一保护，不能对封存的非空结果目录原地更新；需使用独立的新工作目录和明确的副本流程，不要把它们当作修改封存结果的入口。

## 已完成的 post-lint 保存数组回放

已在项目 `.venv` 中执行一次 saved-NPZ phase2 replay，没有解码视频，也没有重拟合椭圆/谐波参数；训练弧凸包仍按冻结方法从保存轮廓确定性重建。旧 phase2 方法协议保持原文件和原 hash；本次外部 replay protocol 在其方法定义和 phase1 输入 hash 之上增加了当前源码绑定，记录 `shrec_video.py`、`test_shrec_video.py`、phase2 runner 与 wrapper 的 SHA-256。回放结果的 `protocol_sha256` 指向这个新的外部协议，不是旧 phase2 hash。旧方法协议原先记录的 implementation hashes 与本次新代码并非同一声明；现有 runner 校验旧协议和 phase1 输入 hash，但不会验证当前源码是否等于协议中较早记录的代码。因此收据同时保留两套来源，不声称旧协议原封不动地绑定新代码。

历史执行记录（非本次公开可直接复现入口；目标目录及外部协议已存在时，wrapper 会拒绝覆盖；需要新复现时复制 wrapper 并更换唯一目录名）：

```powershell
$work = 'C:\path\to\research-work\shrec_execution_2026-10-16'
& $py "$work\replay_phase2_postlint.py"
```

本次外部协议：`D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\shrec\phase2_replay_postlint01_protocol\phase2_protocol.json`，canonical SHA-256 为 `8eb659f4dd4d002b1f8cc11017c0a7ca5c0b83ae93710cb6df055cd48c57717f`。结果与逐帧/逐组差值收据在 `shrec\phase2_replay_postlint01\phase2_replay_comparison.json`。对22帧、8组、3种模型的逐帧误差、组均值和汇总统计逐项对照旧 phase2，最大绝对差为 **0.0 px**；旧 phase2 协议与结果文件哈希复核未变。新的输出manifest包含这份对照收据。

测试命令：

```powershell
& $py -m pytest -q tests\test_shrec_video.py
```

## 主要产物

- `report_zh.md`、`analysis.json`：phase1观察、来源限制和按22个有效完整轮廓帧计算的精确中位RMSE；每个模型均有22帧、8个 capture group，且写明分母。
- `phase1_metric_summary_correction.json`：叙述中曾出现的误差值与 JSON 重算值之更正记录。
- `shrec_sampled_frames_and_contours.npz`：抽样帧、mask位图、原始解码字段和边缘点，可供后续分析复用。
- `phase2/phase2_protocol.json`、`phase2/phase2_distance_results.json`：固定距离精化方案与逐帧/逐组误差、非星形射线比例及全部相对phase1误差增量记录。

常用入口代码为 `roadmc/data/real/shrec_video.py`，其闭合折线解析点距和周期参数曲线细化函数均保持独立于默认合成参数。

路径说明：本页的 `C:\path\to` / `D:\path\to` 为占位根目录，请替换为自己的源码、数据和历史研究收据位置。文中历史数值是已有研究记录，不是运行此页即自动获得的结果。
