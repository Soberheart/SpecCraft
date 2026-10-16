# L4-01 Phase A 报告（CPU）— 2026-10-16

> 本文件是 `records/level4_discriminative_batch_20261016/report.md` 的公开副本（records/ 不入库）。文中 D 盘与 NSCC 路径指向本地/服务器归档，原始数据不在 GitHub。

- 执行：Level 4 执行 agent（Claude，Cowork）；依据：任务单 L4-01 > 决策 D-1 > 交接包 2026-10-16。
- 范围：只用 CPU；**0 次新仿真、0 次 Dyn/AE/Renderer、未登录 NSCC、未提交 PBS**。
- 运行环境：用户电脑上的 Linux 工作 VM（Python 3.10.12、NumPy 2.2.6、PyYAML），直接读写仓库与 D 盘数据；Windows 真解释器（Codex runtime）未能从本会话调用，Windows 复验脚本已备好（见 A2）。
- Phase A 产物目录：`D:\SpecCraft\level 4\L4-01_phaseA_20261016\`（下称 `PA`）；本目录 `records/level4_discriminative_batch_20261016/` 保存本报告与小型产物副本。

## 一、结论先行

| 项 | 状态 | 关键结果 |
|---|---|---|
| A1 核验 | 通过 | 交接包 368 文件 / 71,950,677 bytes 逐 hash 通过（本机副本与下发 zip 内副本 manifest 同为 `cdfce55c…`）；固定对 ZIP 314 项内部 hash 通过并独立 CPU 重审一致；HEAD `0dfc2e2d…` 不变；git status 与 `git_state.json` 一致（4 M + 15 ??；交接正文写“16 个 untracked”，实际索引为 15，属笔误）；代码快照 212 文件与工作树逐字节相同 |
| A2 JSON `Invalid \escape` | 根因确认并修复；VM 三条件全绿；**Windows 真解释器复验待执行** | 不是“路径未经 json.dumps”。写入方都是 `json.dump(ensure_ascii=False)`+UTF-8，正确；读取方（测试）用 `read_text()` 默认编码。Windows 中文系统默认 cp936/GBK，用户目录 `Admin（无密码）` 中 `）`=EF BC 89，0x89 被 GBK 当作双字节首字节，吞掉 JSON 转义 `\\` 的第一个反斜杠，留下 `\D`/`\.` → Invalid \escape。最小字节级复现与回归测试已加 |
| A3 备用变换单位 | 修复 + 独立期望测试；主分支不变 | frozen 主分支 candidate_001 最大值仍为 **0.004149973392486572 mm（frame26）**，与旧报告逐字段相同；修复后 translate_then_rotate 为 0.121 mm（旧错误值 40.15 mm），比冻结约定差约 30 倍 → 平移顺序假说排除 |
| A4 合同 v2 | 实现并出审核 | `pair_result_review_v2.json`：`tool_transform_interface=passed_v2`（最差 0.008532 mm，summary opening，frame32）；v1 复现 `failed_v1`，v1 文件 sha256 前后不变；`model_alignment=unverified`；D-1 已追加到 `docs/level4_decisions.md` |
| A5 字段与时钟说明（限时） | 按时提交 | current/previous/latest/summary/command 来源与相位已从源码+数据确认；v3 下两候选 frame60 保留积分 = 9600 substeps = **1.920 s**（距 frame0），距共享 reset 1.922 s；旧“多保留 60 ms”未再出现；旧模型时序对齐仍未确认（交 D4） |
| A6 池扩到 32 | 完成（含 1 个预筛排除） | `generate_actions(0,32)` 前 16 个**不同**（第 5、14 行因拒绝重采样不同）→ 冻结 16 不变 + seed=1 新 16（candidate_016–031）+ d2_edge 4（032–035）。candidate_024 初始手指穿入材料 4.37 mm → 登记但不执行不计分 |
| A7 工具整体偏移 | 完成；S3b 不支持 | 偏移在场景锚点 `grasp.to_pos` 实现，偏移 0 时场景完全不变；S0/S1x/S1y/S2 场景已生成。**S3b 结论：现有入口只支持单次 grasp（每个 grasp 以瞬移 `set` 开始、无释放/抬起），多步执行不支持 → 转入 L5** |
| A8 D2 量测 | 完成 | 连通组件（以初态渗流半径 r0 为基准，r0=2.637 mm）、颈宽、质量份额；合成形状测试全对。实测：名义与 candidate_001 在 frame60 都只有 1 个显著组件，仅 1–5 个游离粒子；捏合面薄片厚 6.7/7.5 mm（frame60，4 mm 切片）→ “捏细不断” |
| A9 请求 | 草案完成 | 39 次新仿真（≤45）、≈70 次 Dyn rollout、5 个执行作业 + 1 个 Dyn 作业 + 1 个 S4 作业；D4 通过线已预先写定 |

**Phase A 判定**：A2–A4、A6–A9 的 CPU 工作全部完成，A5 按时提交。剩余一项需要你的电脑执行：A2 的 Windows 真解释器两种编码复验（一条 PowerShell 命令，见 §三）。在它通过前，我把 A2 记为“VM 通过、Windows 待复验”。

## 二、修改的文件与理由

仓库 `Doughnet-LLM`（未 commit、未 reset/clean；hash 为 sha256 前 12 位，旧值来自交接代码快照）：

| 文件 | 类型 | 理由 | 旧 → 新 |
|---|---|---|---|
| tests/test_level4_{archive,candidate_interface,clock_review,cpu_resume,cuda_probe,initial_frame,pipeline,remote_session,shared_input,shared_reset,transaction_runner}.py | 改 | 45 处 `read_text/write_text/open` 补 `encoding='utf-8'`；断言未改 | — |
| scripts/probe_level4_cuda.py | 改 | 读取本脚本写出的 UTF-8 JSON 处补编码（150/151/166 行） | f80c09fb6ca2 → 4bd1fa498d04 |
| scripts/level4_remote_session.py | 改 | Windows 侧读取 `job_exit_code.txt` 补编码 | dd867596aedd → 747dca7b1c6e |
| scripts/analyze_level4_tool_transform_offline.py | 改 | A3 单位修复；schema v2；input_receipt（输入 hash、名义参照是否同 run） | 2cee6c41d575 → 61049bedf96e |
| tests/test_level4_offline_tool_analysis.py | 改 | A3 独立期望测试 4 项（旧代码下 4 项全失败，已验证） | 392c0f584205 → 3e9579345244 |
| docs/level4_decisions.md | 改 | 追加 D-1 原文与实现记录（CRLF，与文件既有段一致） | 5e8f359865b9 → d61d632d19b0 |
| level4/tool_interface_contract.py | 新 | v1/v2 合同；v2 对全部位置/开口比较门控（比 v1 覆盖更全），四元数 1e-6 不变 | c94b720548a5 |
| scripts/audit_level4_pair_contract_v2.py | 新 | 冻结 v1 审计（hash 校验）复现 + 原始轨迹重算诊断逐项相等 + v1/v2 判定 | 39b900c6b31f |
| level4/material_split.py | 新 | A8 连通组件 / 渗流半径 / 颈宽 / 质量份额 / 敏感性 | 2359c529bfa2 |
| level4/finger_clearance.py | 新 | A6 frame0 手指-材料穿插预筛 | 6dbbe78198d7 |
| level4/tool_offset.py | 新 | A7 工具整体偏移 | 927da38b02f4 |
| scripts/build_level4_pool32.py | 新 | A6 池扩展（先用冻结 001–015 验证场景构造方法） | 29af1a9e844c |
| scripts/build_level4_sensitivity_scenes.py | 新 | A7 S-run 场景 | 8804417adbc6 |
| tests/test_level4_{text_encoding,tool_interface_contract,material_split,pool32,tool_offset}.py | 新 | 回归与合成例 | — |

有意**未改**：冻结并已投递到 NSCC 的 `probe_level4_{clock_diagnostic,pending_candidate,transaction,initialization}.py`、`review_level4_clock_evidence.py` 中 6 处默认编码读取（服务器 UTF-8 locale 运行，读 ASCII 清单；改动会破坏其投递 hash）。它们被逐条列入 `tests/test_level4_text_encoding.py` 的白名单，新增任何默认编码 I/O 都会使测试失败。

## 三、命令、退出码与日志

| 命令（VM，仓库副本于非 ASCII 路径 / GBK locale） | 退出 | 结果 |
|---|---|---|
| `python3 -B -X utf8 -m unittest discover -s tests -p 'test_level4_*.py' -v`（修改前） | 0 | 186 OK（Linux 下原 5 错不出现，因为路径无反斜杠） |
| 同上，`LC_ALL=zh_CN.GBK`、无 `-X utf8`（修改前） | 1 | 10 errors（Linux 的非 ASCII 文件名在 GBK locale 下另有 surrogate 问题，不是 Windows 复现） |
| 修改后：UTF-8 / GBK locale / `-X warn_default_encoding` 三条件 | 0/0/0 | **219 tests OK**；EncodingWarning 只来自白名单两处 |
| `audit_level4_pair_contract_v2.py …` | 0 | `FIXED_PAIR_AUDIT_V2=passed_v2; V1_UNCHANGED=failed` |
| `analyze_level4_tool_transform_offline.py --nominal-tools <同作业 candidate_000>` | 0 | 新报告；非备用分支 0 字段差异 |
| `build_level4_pool32.py` / `build_level4_sensitivity_scenes.py` | 0 | 36 登记 / 4 场景 |
| `verify_handoff.py`（含 `--restore-parent`，在云端工作区） | 0 | `passed_real_fixed_pair_CPU_audit_recovery` |

日志：`PA/A2_tests_vm_final/`。字节级根因复现：`tests/test_level4_text_encoding.py::test_windows_user_path_json_is_utf8_and_gbk_decoding_reproduces_failure`。

**Windows 真解释器复验（请在 Windows PowerShell 执行一次）**：

```powershell
powershell -ExecutionPolicy Bypass -File 'D:\SpecCraft\level 4\L4-01_phaseA_20261016\A2_windows_check.ps1'
```

脚本只运行测试（`-B` 不写缓存），在 `PA` 下新建 `A2_tests_windows_<guid>` 保存两种条件（`-X utf8` 与 `-X utf8=0` 默认 cp936）的日志与退出码。期望两者都为 `exit=0`、`Ran 219 tests … OK`。

## 四、根因证据与剩余不确定（A2、A3、A5）

- A2：已确认。不确定：Windows 解释器版本/NumPy 版本下其他测试是否有平台差异——由上面的复验回答。
- A3：已确认的分析工具缺陷；修复不影响主分支。
- A5（详见 `PA/A5_clock_note/level4_tool_fields_and_v3_clock_note.md`）：
  - 已确认：current = 下一 substep 起始位姿；previous = latest = 最后积分 substep 的位姿（63 帧逐位相等）；`current = previous + vel·dt`（残差 ≤1.2e-6 mm），所以 current/latest 的 0.004 mm 是**一个 substep 的相位差**，不是误差；summary 由两指 latest 重构。
  - 源码支持但未定量复现：0.0041 mm 物理变换残差来自 `NdimPID` 的按世界轴限幅/到位判定，对 yaw 旋转不等变。
  - v3 时钟：三层结论——标签重放一致（确认）；累计保留积分可认证（在“无未插桩状态写入”假设下）；旧训练时序对齐（未确认，交 D4）。

## 五、D2 的 CPU 预判（写在任何 Phase B 结果之前）

手指是 10×10 mm 方柱、初始开口 100 mm、闭合到 1 mm。名义捏合在 frame60 形成厚约 6.7 mm 的薄片但不断开；要切断需在材料弦长 ≤10 mm 处捏合，而初始不穿插约束把捏合面限制在沿闭合轴 |s| ≲ 20 mm（弦长 ≥48 mm）。因此**当前动作合同下很可能不存在断/不断边界，T4“均分两块”几何上不可达**。d2_edge 4 个候选专门用于实测证伪。旧仿真器 `topology.components=2` 来自 1 个游离粒子，不可当作断开。若 Phase B 证实，按任务单失败分叉报告，提出工具宽度/初态尺寸/多步方案，由总览决定。

## 六、下一项最小实验（详见 `PA/A9_request/next_minimal_experiment.md`）

- B-pilot：4 次仿真（candidate_002、016、032、S1_x_0p02），1×A100、1 h；通过采集/复位/绑定/偏移检查后再提交其余。
- B-exec-1…4：33 次仿真；B-dyn：≈70 rollout（v3 名义 / 旧语义名义两种工具输入），不读真值、先冻结 hash；B-S4：2 次。合计 39 次新仿真 ≤45。
- D4 通过线已写定：名义 rollout VIoU ≥78.95%、候选中位数 ≥78.95%、单步中位数 ≥88.91%、属性误差与旧参照同数量级。
- **提交前还缺两段工程**（不属 A1–A9，但 B 依赖）：① 批量执行入口（读 pool32/S 场景，复用 v3 transaction + gpu_binding + 采集，CPU mock 测试）；② Level 4 Dyn 预测入口（prediction-only、名义工具变换、两种工具输入）。预计 10/27 前完成，届时冻结请求并报总览。

## 七、产物与归档状态

| 产物 | 位置 | 状态 |
|---|---|---|
| Phase A 全部输出 | `D:\SpecCraft\level 4\L4-01_phaseA_20261016\`（≈1 MB） | 已写入；源确认 ✓；独立备份 ✗（C/D 同盘，第三份未配置）；未登记 `level4.archive` catalog（Phase C 统一登记） |
| A6 第一次输出 `A6_pool32/` | 同上 | 被 `A6_pool32_r2/` 取代（YAML 逐字节相同，manifest 增加 phase_b_plan 字段）；保留不删 |
| 仓库修改 | 工作树（未 commit） | 本报告第二节列出；旧文件 hash 可由交接代码快照复核 |
| 历史结果 | 全部未改写 | v1 审核、旧离线报告、旧测试日志保持原样 |

另：用 VM Python 3.10 运行时在 `level4/__pycache__` 新增了 `*.cpython-310.pyc`（gitignored，可随时删除，不影响结果）。

## 八、三项独立判断

1. **工程是否完成**：未完成。接口门（v2）已放行，CPU 回归在 VM 三条件下全绿（Windows 复验待做）；Phase B 的批量执行入口与 Dyn 预测入口尚未实现，32 候选执行与预测均未运行。
2. **研究假设证据**：D1、D4、H1–H4 **尚无证据**（没有预测、没有新执行）；未开展不等于被证伪。D2 只有 CPU 几何预判与 2 个已执行候选的量测，倾向“当前合同下无断开边界”，需 Phase B 证实。S0–S4 未运行，0.02 mm 尚无实验支持。
3. **Level 5 准入**：不具备。Oracle 底座的模型对齐（D4）、实际可行性和选中达标都未验证。
