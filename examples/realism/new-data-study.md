# KIT、SUD与SHREC：原生读取和受限研究

三类数据保留不同的观测含义，不自动进入米制训练数据或独立道路评估。已完成的研究结果保存在仓库外 `D:\path\to\data\RoadMC_public_data\data_study_2026-10-16`；原始包不需重新下载。

| 来源 | 可复用入口 | 输出含义 | 准入限制 |
|---|---|---|---|
| KIT | `concrete_reference`、`diagnose_kit_repeats` | ELA圆环SI剖面、原生网格、共同支持重测评分 | ELA单位明确；PLY毫米尺度是推断；两公路不充当独立最终验收 |
| SUD | `sud_road`、`diagnose_sud_road` | 完整LAS字段审计与局部原码/强度/几何 | 保留raw code；单位和作者码表缺失，不映射病害语义 |
| SHREC | `shrec_video`、`diagnose_shrec_video` | 帧序号配对、原生像素、自动mask轮廓与采集组汇总 | 无已核验深度标定；像素轮廓不当三维病害GT |

来源、实际执行命令、输出字段、冻结源码关系和全部退步见：[KIT说明](new-data-study-kit.md)、[SUD说明](new-data-study-sud.md)、[SHREC说明](new-data-study-shrec.md)。

视频解码通过可选依赖 PyAV。已有项目环境只需安装这个额外依赖；完整安装项目可使用 `python -m pip install -e ".[video]"`。本机研究的初始视频解码使用全局Python；收尾时也在项目`.venv`安装了PyAV，并用该环境执行必要检查，具体版本和退出码见成果中的验证收据。

每个新运行选择尚不存在的输出路径。旧冻结运行绑定执行时的源码哈希，修复后的当前源码不能冒充旧版本：KIT保存了评分修复前源码并提供保存数组补算；SUD旧attempt02执行源码未完整保存，后续派生更正如实保留这个缺口；SHREC统计摘要更正与曲线距离重算分别记录。不要为了通过旧协议而改写其代码哈希。

统计谱研究候选仅使用聚合PSD/分位数及新随机相位，未复制真实patch。它们是显式选用的研究配置，不替换生产默认。现有 `study-import.v1`、道路分组和真实域评估门槛保持不变；总体科学状态仍为 `not_calibrated / not_evaluated`。

路径说明：本页的 `C:\path\to` / `D:\path\to` 为占位根目录，请替换为自己的源码、数据和历史研究收据位置。文中历史数值是已有研究记录，不是运行此页即自动获得的结果。
