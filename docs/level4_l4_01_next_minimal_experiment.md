# next_minimal_experiment — L4-01 Phase B 请求（草案，待 Phase A 审核后冻结）

- 日期：2026-10-16；执行 agent；状态：**未提交**。本文件是 Phase B 的请求定义；冻结时追加 sha256 并复制到提交目录，之后不得按结果修改。
- 依据：任务单 L4-01 第 3、4 节；决策 D-1；`tool_interface_contract_v2`=passed_v2（A4）。
- 资源：NSCC，单作业 **1×A100**，只用 PBS 分配 UUID 的逻辑 device0（`level4.gpu_binding` 已验证方式）；不在登录节点跑 CUDA；不重训；不跑 Renderer/AE 前处理（frame0 观测/AE 复用）；不运行旧 22 号脚本。

## 0. 待证伪的问题与为什么旧数据答不了

| 问题 | 旧数据为何不够 |
|---|---|
| D4：v3 执行语义下 Dyn 误差是否与旧语义相当 | Level 4 尚未运行任何 Dyn；旧 L2/L3 误差来自旧时序 |
| D1：按预测选中的候选是否系统性偏乐观 | 只有 2 个执行候选，没有预测 |
| D2：断/不断边界 | 仅名义+1 个小偏移；CPU 量测显示二者均为“捏细不断”（见 §5） |
| S0–S4：0.02 mm 偏差是否被材料放大 | 没有重复与偏移执行 |

## 1. 冻结输入

| 项 | 值 |
|---|---|
| 初态 | 共享 reset 状态 `shared_source_state.npz` sha256 `dcc8fd5c…86b8`（同作业 25837718） |
| 源池 | `pool.json` sha256 `5a62a143…2c990a`（16 个冻结候选不变） |
| 扩展池 | `A6_pool32_r2/pool32_manifest.json`（36 登记：冻结 16 + seed1 16 + d2_edge 4；candidate_024 预筛初始穿插→不执行不计分） |
| 敏感性场景 | `A7_sensitivity_scenes/sensitivity_manifest.json`（S0/S1x/S1y/S2；S4 规则预登记） |
| 模型 | Dyn checkpoint `best.pth` sha256 `a05452f6…cacae`，冻结，不重训 |
| 名义工具 v3 | job 25837718 `candidate_000/actual_tools`（63 帧，frames sha256-of-hashes `4cfbb50f…0a18`） |
| 名义工具旧语义 | 旧语义名义工具轨迹（来源候选：核心档案 `files/level3_dyn_input` 或 job_25769308 runtime；**提交前核实具体文件与 hash**），仅用于 D4 对照组 |
| 主端点 | frame60 / +1.920 s；辅助 frame62 / +1.984 s，单独报告 |
| 阈值 | ρ/η/ε = null；C2/C5 用阈值扫描并报告全部阈值 |

## 2. 作业划分（全部单卡、多个作业并行，不用多卡作业）

| 作业 | 内容 | 候选 / run ID | 新仿真 | Dyn 调用 | walltime 申请 | 数据量 |
|---|---|---|---:|---:|---|---|
| B-pilot | 先跑小批，验证采集/复位/绑定/偏移 | candidate_002, candidate_016, candidate_032, S1_x_0p02 | 4 | 0 | 1 h | ≈0.1 GB |
| B-exec-1…4 | pilot 通过后并行提交 | 其余 30 个 execute_and_predict 候选 + S0、S1y、S2，分 4 组（9/8/8/8） | 33 | 0 | 每作业 1 h | ≈0.75 GB |
| B-dyn | 仅预测（无真值）：35 个有效候选 × 2 种工具输入（v3 名义变换 / 旧语义名义变换），每候选 rollout 至 frame60 并保存全部预测时刻；另 AE 重建与单步 | 35 有效候选（含 000/001） | 0 | ≈70 rollout + 单步/AE | 2 h | ≈1–2 GB |
| B-S4 | B-exec 完成、按预登记规则选 1–2 个候选后提交 | 待定（规则见 A7 manifest） | 2 | 0 | 1 h | ≈0.05 GB |
| **合计** | | | **39** | ≈70+ | | ≈2–3 GB |

新仿真 39 ≤ 45（任务单阈值）：池候选 33（冻结 14 + seed1 15 + d2_edge 4）+ S0/S1x/S1y/S2 4 + S4 2。单次仿真实测 58.6–101.1 s（含首次编译），按 ≤120 s/次 + 初始化估算，8 个/作业约 20 min；1 h walltime 有 2–3 倍余量。B-dyn 不依赖执行真值，可与 B-exec 同时运行；其输出在任何执行真值被读取前做 hash 冻结，形成 prediction-only selection manifest。

预测的工具输入 = 对 candidate_000 actual_tools 施加冻结 `transform_tools`（center=grasp to_pos，unit 0.2），不使用候选自身执行轨迹（任务单 B1）。S-run 的偏移不进入预测输入。

## 3. D4 通过线（看到任何 B 结果前冻结）

参照：L3 scene0000 名义动作旧语义 rollout VIoU 81.95%、单步 91.91%（`check_level3_dyn.py` 同一计算方法）。

| 判据 | 通过 | 不通过 | 不确定 |
|---|---|---|---|
| P1 名义 candidate_000，v3 工具输入，frame60 rollout VIoU | ≥ 78.95%（参照−3 pp） | < 78.95% | — |
| P2 有效执行候选的 frame60 rollout VIoU 中位数 | ≥ 78.95% | < 78.95% | 有效执行 < 20 个 |
| P3 单步 VIoU 中位数 | ≥ 88.91%（91.91−3） | < 88.91% | — |
| P4 u/v 跨度与颈宽误差中位数 | 与旧语义参照（Phase C 先用 L3 已保存预测/执行按同代码算出，**在读 B 结果前写入**）同一数量级（比值 ∈ [0.1, 10]） | 比值 > 10 | 旧参照无法从已存数据计算 |

D4 = 通过：P1–P3 全通过且 P4 通过或不确定；不通过：任一 P1–P3 不通过；否则不确定。v3 与旧语义两种工具输入下预测之差单独报告（不作为通过线）。

## 4. 停止规则（与时长无关）

- 任一作业中 >20% 仿真失败、复位门槛（0.001 mm）失败或 GPU 绑定失败：立即停止，不再提交后续作业，报告总览。
- B-pilot 未全部通过以下检查前，不提交 B-exec：63 帧材料/工具/原始帧齐全；reset 逐数组精确等于共享状态；GPU UUID 绑定通过；S1_x_0p02 的执行工具轨迹相对 candidate_000 的平移 = (0.02, 0) mm（tool_interface_contract_v2 口径，偏差 ≤0.02 mm）；clock 事件重放与标签一致。
- 新仿真需要超过 45 次：先书面说明再提交。
- 任一候选出现初始穿插（frame0 clearance ≤0）：该候选标记 invalid，不重抽。

## 5. 已知风险与预判（写在结果之前）

- **D2 很可能无“断”**：手指为 10×10 mm 方柱，初始开口 100 mm、闭合至 1 mm；名义与 candidate_001 在 frame60 均为单一显著组件（≥0.5% 质量），只有 1–5 个游离粒子；捏合面上的薄片厚度 6.7/7.5 mm（4 mm 切片）。要切断需在材料弦长 ≤10 mm 处捏合，而 frame0 无穿插约束把捏合面限制在沿闭合轴 |s| ≲ 20 mm（弦长 ≥ ~48 mm）。T4“均分两块”在当前动作合同下几何上不可达。d2_edge 4 个候选用于实测证伪这一预判；若成立，按任务单失败分叉报告并提出调整方案（更宽工具/更小初态/多步），不自行修改动作空间合同。
- 旧仿真器的 `topology.components=2`（frame60）来自单个游离粒子，不能当作“断开”证据。
- candidate_024（seed1 抽样）初始即穿插（−4.37 mm），按规则不执行；D1 有效样本为 31 个随机候选 + 4 个 d2_edge（d2_edge 默认不并入 D1 统计）。

## 6. 提交前清单（全部满足才提交）

1. Phase A 审核通过（总览确认）。
2. 新执行入口：批量读取 pool32/sensitivity 场景、复用 v3 transaction + gpu_binding + 采集；CPU mock 测试通过（**尚未实现**，见报告“下一步”）。
3. Level 4 Dyn 预测入口：prediction-only，名义工具变换，两种工具输入；CPU 形状/hash 测试（**尚未实现**）。
4. toolkit 打包：tool_manifest、源码 hash、request.json（含本文件 sha256、通过线、停止规则）、bash -n、PowerShell 5.1 解析检查。
5. Windows 真实解释器下 Level 4 全套测试两种编码条件通过（见报告 A2）。
6. 远端只读检查：项目根/venv/空间（≥5 GB）、qstat 无同名作业、NSCC 原件仍在。
7. 本文件冻结 hash 写入 request。

## 7. 验收（Phase C 前置）

每个 run 保存：动作、tool offset、预测轨迹（frame0–60 全部预测时刻）、执行 63 帧原始材料与工具、退出状态、clock 重放、绑定收据、耗时、hash；未执行明确 `not_executed`；失败与 unknown 保留在分母中。
