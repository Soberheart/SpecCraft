# 工具字段来源与 v3 时钟复核（L4-01 A5，限时说明）

- 日期：2026-10-16；执行：Level 4 执行 agent；仅 CPU，读取作业 25837718 已保存证据（`D:/SpecCraft/level 4/gpu_binding_pair_restore_20261016_22_retry/evidence`，314 项内部 hash 本轮复核通过）。
- 性质：A5 是限时说明，**不是放行前提**。工具接口放行依据为 D-1/`tool_interface_contract_v2`（A4）。
- 新增文件：`independent_clock_replay.py`（不 import level4 的独立事件重放）、`independent_clock_replay.json`（输出）。

## 1. 各工具字段：来源、采样时刻、单位

| 字段（actual_tools 帧包 / 日志行） | 源码位置 | 物理含义与采样时刻 | 单位/类型 | 状态 |
|---|---|---|---|---|
| `tool_00i_pos/quat/vel/ang`（current） | `probe_level4_candidate_interface.py::log` → `ToolEntity.get_state()`，读 ring slot `sim.cur_substep_local` | 本帧最后一个 substep 处理完后、**下一 substep 的起始位姿**（`advect` 在 substep f 的 pre-coupling 写 `pos[f+1]`） | 仿真单位（×1000×0.2 → mm），quat wxyz；f32 | 已确认（源码+数据） |
| `previous_tool_00i_*` | 同上，读 slot `(cur_substep_global−1) mod max_substeps_local` | 上一 substep（即本帧最后一个被积分的 substep）的位姿 | 同上 | 已确认 |
| `latest_tool_00i_pos/quat` | `ToolEntity.substep_post_coupling → update_latest_pos/quat`：`latest = f32(pos[f])` | 最后被积分 substep 的位姿 = previous slot；`latest_previous_exact_all_frames=true`（63 帧、两手指逐位相等） | f32 | 已确认 |
| `ee` 汇总（日志 `pos/pos_left/pos_right/open/quat`） | `sim/generate/ee.py::EndEffector.get_state` | 由**两指 latest** 计算：中心=左右均值，open=左右距离（带符号），quat=左指 latest；`pos_left/right` 再由 `get_left_right_pos(pos,quat,open)` **重构**，不是直接读手指 | 仿真单位；open 为内表面间距 | 已确认；重构误差解释 summary/latest 0.00215 mm（frame25）属推断 |
| command / pending target（`_tgt`） | `ee.py::step/_move/_grip` 设速度；`ToolEntity.process_input` 在下一 step 开始写入该 step 全部 10 个 substep 的 `vel/ang`；`pos/quat` 仅在 `set_state` 时写首 slot | 控制器按**观测到的 latest 状态**闭环计算速度目标，下一 step 生效 | 速度：仿真单位/s | 已确认 |
| `step` / `sample_global_substep` / `current_local_substep` | `log` 记录 | `step = frame×16`（16 个 step/帧，每 step 10 substep，substep_dt=0.0002 s → 0.032 s/帧）；`sample_global_substep` 是**可被 set_state 清零的** solver 计数（frame60 时为 0），不能当累计时钟 | 整数 | 已确认 |

数据核验：两候选全部 63 帧、两手指满足 `current = previous + vel_prev·dt`（最大残差 4.1e-7 mm / 1.2e-6 mm）。因此 **current/latest 分离 = 恰好一个 substep（0.2 ms）的运动**，速度上限 25 mm/s 时 ≤0.005 mm（实测 0.0043/0.0040 mm）。这是采样相位差，不是误差；比较“预测输入”与“执行”时必须固定使用同一相位（建议：current，与 v1/v2 主比较一致）。

## 2. 物理变换残差（0.0041 mm，frame26）的来源：源码支持的假说，未复现

`ee.py::NdimPID` 对世界坐标 x/y/z 与欧拉角各轴**独立** PID，并按轴做输出限幅（`output_limits=(−vmax, vmax)`），且以每轴 `tol=1e-2` 判定到位。按轴限幅/到位判定对绕 z 轴的旋转不等变：candidate_001 的目标是名义目标的平移+yaw 旋转，当两轴速度饱和程度不同，执行轨迹就不再是名义轨迹的精确刚体变换；控制器读取的是 f32 的 latest 状态。残差峰值出现在运动段（frame25–32），符合这一机制。**未做**：离线重放 PID 以定量复现 0.0041 mm（需按控制器配置逐 step 重算，超出 1 个工作日限时）。已排除：平移/旋转顺序假说（A3 修复后 translate_then_rotate 最大 0.121 mm，比冻结约定差约 30 倍）；单位错误与 yaw 反号（0.85 mm / 9.66 mm）。

对 Phase B 的含义：预测输入采用“对 candidate_000 actual_tools 施加冻结变换”，与真实执行的差别 ≤0.0085 mm（v2 下通过）；它是控制器非等变造成的系统性小偏差，**不随帧单调累积**（峰在 frame26/32，不在 frame62）。

## 3. v3 时钟独立复核（frame60 累计积分时间）

独立重放规则：`step` 加 `n_substeps`；`get_state` 记快照；`set_state` 回滚到快照值；`reset_origin` 清零；每帧帧包的 `(clock_event_count, active_branch_substeps)` 必须等于重放前缀。

| 量 | candidate_000 | candidate_001 |
|---|---:|---:|
| 事件数 | 1788 | 2279 |
| 观测到的全部 substep（含临时分支） | 16420 | 21220 |
| 被回滚丢弃的 substep | 6400 | 11200 |
| restore 次数 | 66 | 75 |
| 每帧保留积分增量（集合） | {160} | {160} |
| frame0 保留 substep | 10 | 10 |
| frame60 保留 substep | 9610 | 9610 |
| frame60 距 frame0 | **1.920 s** | **1.920 s** |
| frame60 距共享 reset | 1.922 s | 1.922 s |
| frame62 距 frame0 | 1.984 s | 1.984 s |
| 与已存 clock 报告逐帧一致 | 是 | 是 |

结论分三层：

1. **标签重放一致**：已确认。两候选 63 帧保留分支积分与标签逐帧相等（误差 0）。
2. **累计保留积分可认证**：在下列假设下成立——所有改变材料/工具状态的路径都经过被插桩的 `Scene.step/get_state/set_state/reset`；没有未记录的直接写场（field）操作。事件重放无法证明“没有未插桩旁路”，这一条仍为假设。旧作业 25804224 的“frame25 起多保留 30 步/60 ms”在 v3 下**未再出现**（增量恒为 160）。
3. **旧模型训练/输入时序已对齐**：**未确认**。frame0 是 reset 后第一个 10-substep step 之后才记录的（距 reset 0.002 s），frame60 距 reset 1.922 s。旧训练数据是否采用同一 frame0 约定、是否含旧拓扑步进副作用，需要由 D4（Phase B 的 Dyn 误差对比）裁决。`model_alignment` 保持 unverified。

## 4. 未确认项（交给 D4 或后续）

- 0.0041 mm 物理变换残差的定量复现（PID 重放）。
- 未插桩状态写入旁路是否存在。
- 旧训练数据的 frame0 相位与拓扑副作用是否与 v3 一致（D4）。
- `resources_used.ngpus=0` 调度统计差异（与本说明无关，沿用原记录）。
