# Level 4 CPU 离线工具轨迹分析

该分析只读取已经保存的固定候选对证据，不启动 NSCC、GPU、模拟器、Torch、Taichi 或模型，也不向规划器提供执行真值。它比较 candidate_000 的同一批 `actual_tools` 作为名义物理参照、candidate_001 的实际工具轨迹，以及当前/previous/latest 三种保存位姿。

## 当前恢复数据的命令

在 Windows PowerShell 中执行：

```powershell
$Repo = 'C:\Users\Admin（无密码）\Documents\ChatGPT\Doughnet+LLM\Doughnet-LLM'
$Python = 'C:\Users\Admin（无密码）\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$Evidence = 'D:\SpecCraft\level 4\gpu_binding_pair_restore_20261016_22_retry\evidence'
$NominalTools = "$Evidence\execution\candidate_000\actual_tools"
$Output = 'D:\SpecCraft\level 4\cpu_offline_tool_analysis_20261016_01\tool_transform_report.json'

Set-Location $Repo
& $Python scripts/analyze_level4_tool_transform_offline.py `
  --evidence $Evidence `
  --nominal-tools $NominalTools `
  --output $Output
```

看到 `CPU_OFFLINE_ANALYSIS_PASSED` 后，报告在 `$Output`。脚本遇到缺帧、步号不一致或已存在输出文件会停止，不覆盖旧报告。

## 报告如何读

`frozen_center0_mm_unit02_yaw_plus_translate_current` 是 Level 4 当前约定：中心 `[0,0,0]`、`unit_to_meter=0.2`、正 yaw、先旋转再平移。`latest` 和 `previous` 是同一假设下换用对应保存索引的比较；它们只用于定位采样时序差异。

其余三个名称是诊断假设：单位换成 1.0、yaw 反号、或先平移后旋转。它们不能作为修复或规划输入，也不能改变 0.001 mm 门槛。

本次已生成的报告显示：candidate_000 在 `current` 下冻结变换误差为 0；其 `latest/previous` 与 current 最大相差约 0.004330 mm。candidate_001 在冻结变换、`current` 下最大误差约 0.004150 mm（frame 26）；改用错误单位或 yaw 反号会更差。因此当前证据支持“需要继续核对接口时序/控制轨迹”，不支持修改阈值、改选动作或宣称候选池可行。

该报告仍不评价任务 A/B、模型对齐、候选选择或物理累计时间；这些门槛保持原状态。
