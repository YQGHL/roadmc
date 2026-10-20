# 无 KIT 数据时可运行的四项工程检查

这些入口支持程序检查和已有数据的受限诊断。坑槽三帧的 XYZ 单位、视图独立性，以及真实坑壁和坑底材料统计仍未核实。示例中的几何尺寸、材质参数和随机种子是人工设置，不是实测自然分布。生产默认不随这些检查改变。

## 1. 复算冻结坑缘的三种误差

新增阶段 `pothole_outline_fairness` 只读取旧拟合结果，不调用优化器。它需要原有六个 `pothole_frame_01/02/03`、`pothole_label_01/02/03` 资产，以及新增资产 `pothole_frozen_metrics`。七个资产都须提供 SHA-256，并绑定同一个允许原生坑缘诊断的来源。

```powershell
python -m roadmc.scripts.run_realism_study `
  --manifest path/to/verified-outline-manifest.json `
  --protocol examples/realism/study-protocol.json `
  --stages admission pothole_outline_fairness `
  --output-dir path/to/new-outline-study
```

该阶段是显式选项，不包含在旧默认阶段中。也可以单独运行：

```powershell
python -m roadmc.scripts.realism_experiments pothole-outline-fairness `
  --dataset-root path/to/dataset3 `
  --frozen-metrics path/to/prior-pothole-native/metrics.json `
  --output-dir path/to/new-outline-diagnostic
```

`metrics.json` 并列保存：历史模型归一误差 `r/g-1`、公共椭圆坐标误差 `r-g`、乘以方向尺度后的原生投影径向误差。旧椭圆基准是 `r-1`。原生误差仍没有毫米单位，也不是到曲线的最短距离。百分比变化为“新/旧−1”，负值表示误差下降；只有旧误差非零时才计算百分比。

结果还保存16角区的贡献、两项 MSE 恒等分解、最大标注连通区的完整边界与有限 XYZ 边界的数量、空角区和缺测比例。所有比较使用同一有限边界子集，不补造缺测点。冻结输入哈希、TLS参考面及旧分数会核对；缺少历史核验记录时明确标为未知。历史分割仍是同视图插值诊断。

## 2. 显式启用坑壁与坑底材质

[完整示例](pothole-wall-floor.json) 只用于 `schema_version=2` 的 pothole 事件。将以下对象放入事件 `parameters.material_partition`，并移除同级的 `texture` 和 `exposed_rms_m`：

```json
{
  "mode": "wall_floor",
  "floor_rho_start": 0.45,
  "floor_rho_end": 0.8,
  "floor_texture": {"mode": "iid", "rms_m": 0.0004},
  "wall_texture": {"mode": "retain"}
}
```

两个阈值必须满足 `0 <= start < end <= 1`。过渡使用圆形或已旋转、已谐波变形轮廓的同一个归一半径，按 smoothstep 混合坑底与坑壁纹理。每个区域支持原有 retain、iid、procedural；区域随机流独立。retain 指该事件执行前的当前纹理，因此能继承先前病害或修补结果。

分区只改变原有获胜像素上的纹理。最终损失、足迹、事件归属和标签不由材质模式决定。输出增加最终分区 owner、坑底/坑壁权重和配置、随机流及最终统计；修补或后续损伤覆盖的位置会清除旧分区记录。未启用时不增加这些字段。权重代表程序中的混合比例，不能当作实测材料分区。

owner 和双区权重始终对应 `x_m/y_m` 的源规则网格；扫描后仍保持这个形状，不是观测 `points` 的逐点属性，不能按扫描点下标直接索引。

原有 `roadmc.scripts.replay_surface` 入口可直接读取这个示例计划，参数用法见其 `--help`。同一材质场的混合会严格回到原场；独立随机场即使设置相同，逐点结果也不相同。

## 3. 导入已有 CSV、SUR 和现场 TXT

按[导入清单模板](data-import-manifest.template.json)填写实际路径、档案与明确成员哈希、单位证据、处理链、分组、曝光和允许用途，然后运行：

```powershell
python -m roadmc.scripts.import_study_data `
  --manifest path/to/verified-import-manifest.json `
  --output path/to/new-native-import
```

输出目录必须不存在。全部源文件和选定成员先检查、解析，再写入 NPZ、逐源 provenance 和总 receipt。不会遍历未列出的 ZIP 成员。

- Royal CSV 保留发布矩阵的原值；Z尺度和发布处理阶段仍未知。固定 TRAIN 小样用于导入检查，不重新分析已见 TEST 或调整纹理参数。
- SUR 保留原始整数码、格式声明的有效性、按头字段换算的数值和各个候选敏感性掩膜。零值和上限码不会自动删除。敏感性掩膜需要与 validity_mask 明确相交后才能用于分析；码步长不是仪器精度。
- 四列现场 TXT 保存 XYZ 原值与 `unknown_aux`。第四列不会被解释成标签或强度。单列 profile 必须提供明确采样起点、步长和单位。使用 `max_rows` 时记录截断。

这些文件使用 `roadmc.study-import.v1`，不交给默认米制 `load_raw_scene`。目前仅允许原生值与归一化诊断，`metric_export_eligible` 和 `independent_validation_eligible` 固定为 false。

将导入束登记进研究准入时，在同一来源的 `datasets` 行增加：

```json
"study_import": {
  "npz_asset": "field_values",
  "provenance_asset": "field_provenance"
}
```

在 `assets` 中声明这两个文件的路径、哈希及同一 source_id。来源行的单位、单位证据、处理链、材料、角色、分组、曝光须与 provenance 一致；`group_identity` 对应导入的 `group_status`。允许指标只能缩小。修改 eligibility 标志不能升级权限。执行 `--stages admission` 可核对这些绑定；这不表示旧纹理或几何实验已经消费了导入 NPZ。

## 4. 检查组合与旧结果兼容性

以下测试使用小型人工表面，覆盖圆/旋转椭圆/谐波轮廓、松散和裂缝、修补后再次损伤、弱损伤覆盖、规则与不同轴间距网格，以及直接表面、无噪扫描、有噪扫描。不同轴间距网格支持直接表面，当前线激光入口会明确拒绝这种网格。

```powershell
$env:ROADMC_GENERATOR_NO_TORCH = '1'
python -m unittest discover -s tests -p test_realism_combination_matrix.py -v
python -m unittest discover -s tests -p test_realism_engineering_admission.py -v
python tests/test_material_partition_replay.py
python tests/test_study_import.py
```

另有 `test_outline_diagnostics.py` 的合成评分检查；配置 `ROADMC_OUTLINE_DIAG_FIXTURE` 与 `ROADMC_OUTLINE_FROZEN_METRICS` 后，执行同一 unittest 入口还可核对三帧历史诊断。测试成功表示上述工程条件满足；真实域状态仍为 `not_calibrated / not_evaluated`。

测试范围说明：上述指向历史`tests/`文件的命令是开发记录，相关测试未随本次公开文档完整发布，不能当作干净检出可直接执行的验证。当前新增的SUD来源回归可运行 `python -m unittest discover -s tests -p test_sud_provenance.py`。
