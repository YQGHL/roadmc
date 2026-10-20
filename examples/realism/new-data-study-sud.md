# SUD-ROAD 真实 LAS 接入说明

本说明适用于当前受控读取器 `roadmc/scripts/diagnose_sud_road.py` 与 SUD-ROAD `SUD_road.las`。它保留 LAS 原始字段并给候选空间块做描述统计，不负责将标签码翻译成路面病害语义，也不把坐标换算成物理单位。

## 字段与解释门槛

- LAS 1.2 format 0 的 classification 原始字节必须原样保留；通用 LAS 结构可拆看 low-five class bits 与 high-three flags。SUD 文件实测字节为 0、33–38，但这只支持 `raw_code` 分组。作者 README 列出的七个名称没有整数码表，禁止按 README 顺序映射。
- `point_source_id` 是原字段，可作为来源分组进行联合统计；没有额外证据时，不得叫它道路、扫描趟次或病害类别。
- 文件 scale 为 0.001，但比例因子不等于单位。VLR payload `01000100000001000004000001000000` 解码出 GeoKey 1024=0，即模型 CRS 未知/未定义；没有 Projected CRS key 3072 或 projected linear units key 3076。物理 CRS 和线性单位仍未知，几何量只能以 LAS 源坐标单位报告，不得称米或毫米。依据：[OGC GeoTIFF 1.1](https://docs.ogc.org/is/19-008r4/19-008r4.html)。
- 几何、强度和局部形状统计只描述观测点云，不是裂缝/修补语义标签、深宽真值或已校准的检测阈值。

## 受控命令

CLI 接受位置参数 `source output` 和冻结协议参数。已有输出目录会在读取 source/protocol 前拒绝，且不会写入失败记录；仅本次 run() 独占创建输出目录后，失败路径才会记录 run_record。新运行必须使用新建的空输出目录，并让协议绑定输入 SHA256、当前 reader/CLI 执行源码 SHA256、输出路径和分析参数；不得复用旧协议或覆盖既有结果。当前诊断器还固定校验本次 SUD LAS 的 source SHA256，因此其它 LAS 不会被当成同一数据集接收。

```powershell
python -m roadmc.scripts.diagnose_sud_road `
  C:\path\to\downloads\SUD_road.las `
  D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attemptXX `
  --protocol D:\path\to\new_frozen_protocol.json
```

`attempt02` 的原冻结协议绑定的是运行时旧源码哈希；guard 修复后，当前源码哈希已变化，不能拿旧协议重跑。历史 attempt02 的运行记录曾被一次“目录已存在”负测覆盖，因此其现有 `run_record.json` 只记录拒绝结果；attempt02 的审计、NPZ、sidecar 文件哈希仍以交付清单中的历史记录为准。不要对 attempt02 重试。

当前 `attempt02/run_record.json` 保存的实际命令是：

```powershell
C:\path\to\python\python.exe C:\path\to\roadmc\roadmc\scripts\diagnose_sud_road.py `
  C:\path\to\downloads\SUD_road.las `
  D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attempt02 `
  --protocol D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attempt02_protocol.json
```

这条当前记录的 exit code 是 1，原因为输出目录已存在；它不是创建 attempt02 数据产物的成功运行记录。原成功调用的 `run_record.json` 内容未能恢复，因此不声称上面的失败记录就是首次成功运行的凭据。请以 attempt02 冻结协议、历史输出清单和 attempt03 派生更正共同追溯数据与解释。

## 本次权威数据与更正

- attempt02 原始数值审计、26 个 NPZ 和 26 个 JSON sidecar 位于 `D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attempt02`；历史完整清单在 `C:\path\to\research-work\sud_execution_2026-10-16\attempt02_delivery_manifest.json`。
- 权威更正审计为 `D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attempt03_derived_correction\sud_road_audit_corrected.json`，配套中文说明和 `source_evidence.json` 在同一目录。它只更正 GeoKey 文字，引用 attempt02 的 NPZ/sidecar 哈希，不复制或重写其点值。
- attempt03 v4 协议绑定 attempt02 当前55文件树及历史55文件清单，并披露 run_record 的单文件变化：`D:\path\to\data\RoadMC_public_data\data_study_2026-10-16\sud\attempt03_derived_correction_protocol_v4.json`。此前 attempt02 执行源码只有哈希，未找到源码快照；不得声称已重建其运行源码。



- guard 修复前的 reader/CLI/test 源码副本与 SHA256 清单保存在 `C:\path\to\research-work\sud_execution_2026-10-16\source_before_guard_fix`；修复前后哈希、冻结协议关系和验证结果见同目录 `guard_fix_validation.json`。


路径说明：本页的 `C:\path\to` / `D:\path\to` 为占位根目录，请替换为自己的源码、数据和历史研究收据位置。文中历史数值是已有研究记录，不是运行此页即自动获得的结果。

本次运行时来源修复只记录 reader 与 CLI 两份执行源码，不再要求本地存在测试文件。旧冻结协议绑定了旧源码及额外测试哈希，与当前执行版本不兼容；请保留旧协议，使用独立新输出目录为当前版本重新冻结，不能编辑旧哈希绕过校验。该诊断器针对已固定哈希的特定原始LAS和协议，并非任意LAS的通用入口。
