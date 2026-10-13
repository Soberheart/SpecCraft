# Level 4 首次交付报告

2026-10-12后续：服务器登录节点55项CPU测试、计算节点42项CPU测试通过。首个名义CUDA模拟退出0，耗时89.774s，但因初态最大坐标误差0.002675mm超过暂定0.001mm，提取/重置门槛失败；第二次重置未运行。完整诊断包待下载复核（用户报告hash为5fa19f42d1e3cacce1ab1f35a0125ae5650b179562bcce6eeec0df6db4e018b0）。工程仍未完成，研究假设未获评估，Level5不就绪。以下保留2026-10-10首次交付的历史状态，不作为最新GPU状态。

2026-10-10。**M0本地审计及M1 CPU合同完成；Level4总体工程和研究实验未完成，尚不满足Level5准入。**

## 实际实现与修复

保留已有Level4架构、Level3代码、frame60_alignment.patch及历史产物。没有自动commit/push或新增运行依赖。

| 模块 | 实际交付 |
|---|---|
| spec / geometry / evaluate | A/B可执行规格校验、体素完整支撑跨度、单位/固定轴语义、三值逐项报告、初态检查、dense实际过程/共同采样诊断、frame62独立辅助 |
| selection / statistics | 可行后最小cost、固定破同分、拒绝/unknown、S→V与S→U拆分、按初态分组配对统计 |
| pipeline / artifacts | prediction-only文件边界、PointGoal目标隔离、执行前不可覆盖选择清单与执行hash关联、全池前瞻诊断接口、独立子进程、严格resume、两次名义reset审计 |
| 采集/证据 | untracked源码实际内容、ignored board和依赖源码快照；权重/输入hash、退出日志、失败目录、raw材料轨迹、解码调用/峰值显存/存储与耗时字段 |
| CLI / 配置 / 测试 | validate/evaluate/prepare/predict/execute/analyze/freeze/snapshot/audit-reset/validate-reset-gate；开发池与uncalibrated A/B模板；CPU复验入口；raw日志表示敏感性与dense摘要 |

新发现而保留的基线问题：builder两次取yield_low_high[0]；step0在一次solver step后才记录；query覆盖和2mm支持半径未认证。未通过改模拟参数、放宽reset容差或填任意任务数字掩盖。

## 本次实际验证

工作目录：`C:/Users/Admin（无密码）/Documents/ChatGPT/Doughnet+LLM/Doughnet-LLM`。

```powershell
& 'C:/Users/Admin（无密码）/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest discover -s tests -p 'test_level4_*.py' -v
& 'C:/Users/Admin（无密码）/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/verify_level4_cpu.py --output records/level4_cpu_20261010_delivery_02 --probe-ssh
```

Level4 **47 tests OK**；含Level3 helper回归的全套 **74 tests OK**，退出0。最终逐命令时间、argv、cwd、退出码及stdout/stderr保存在commands.json及对应文本文件；不以先前38/43/46项测试代替最新版验证。相关Python以3.9 grammar解析及配置JSON解析通过；实际运行Python3.12.14，不声称服务器3.9兼容已执行。三个CLI --help与开发配置验证退出0；uncalibrated A/B正式验证预期退出2。PBS脚本LF字节检查通过；Bash语法检查本机未运行，准确命令在runbook。

测试覆盖解析长方体/球/单元、轴/单位、平移/刚体旋转、阈值等号/裕量、零参照、空/NaN、污染/掩码、触边/出界、缺/重复/错序时间、已知违例与unknown、稠密漏峰、候选合法性/同分/拒绝、未来truth替换/损坏/移除、目标latent隔离、前瞻清单/执行关联、重复reset路径/实际输入及源码环境变化、raw表示解析支撑、配对/分母和源码可恢复性。

board实际重算hash与provenance一致；本地实验输入盘点为空。BatchMode SSH实际退出255，认证失败，不是审批拒绝。**没有Level4 CUDA预测、独立MPM执行、真实标定、固定池研究结果、constraint-CEM或正式对照。**

## 产物索引

| 路径 | 内容 / 性质 |
|---|---|
| docs/level4_audit.md | 事实等级、Level3来源边界、环境/资产/材料/时间审计 |
| docs/level4_protocol.md | 最小A/B任务、参考轴/单位/时间/约束、数据权限、unknown与Level5接口；非正式冻结协议 |
| docs/level4_calibration.md | 尚未标定字段及其分别需要的证据 |
| docs/level4_decisions.md | 技术决定与保留基线问题 |
| docs/level4_server_runbook.md | 同步保护、实际输入核验、两次reset、16候选、后处理和条件性M3命令 |
| configs/level4/ | schema、A/B未标定模板、开发池、未冻结split |
| records/level4_cpu_20261010_delivery_02/commands.json | 完整真实命令、时间和退出状态 |
| 同目录 result.json / environment.json / inventory.json | CPU验收、依赖与资产/真实输入盘点 |
| 同目录 synthetic_example/ | 解析合成spec/网格/预测packet/报告/选择清单；**不是研究数据，不可授权真实执行** |
| 同目录 tested_source_hashes.json | 实际测试时源码/资产版本 |
| 同目录 runtime_snapshot/ | 最终源码/配置/文档/相关依赖与资产的实际内容及manifest |
| 同目录 level4_delivery.zip / delivery_manifest.json / delivery_index.json | 可审查交付包及校验索引 |

此前 `records/level4_cpu_20261010_delivery_01/` 保留，为早一版本日志。最终证据以02源码hash为准。权重与服务器原始历史日志尚未从服务器归档；本地源码快照不能替代它们。

## 下一阶段最小输入与实验

需要可用的既有NSCC会话、现有venv/checkpoint和scene00 Dyn artifacts的实际文件；核对服务器当前源码差异、材料有效参数、配置/权重/input哈希及normal/personal授权资源。无需重新跑Level3，也无需重训。

先两次名义独立进程验证reset/time，再一开发初态16候选预测+全池独立执行。每批之前确认raw log、query/工具轨迹、预测latent/occupancy、失败日志和源码均可恢复。先回答实际可行解，再预测识别，再选中结果。标定后用新前瞻池完成M3；达到门槛后才接M4/M5。

raw记录已准备能供后续图/视频恢复；真实日志恢复和双视角渲染本次未验证，尚无F1多解、F3散点/矩阵、F5过程反例、F4/T1正式主表、V1真实视频。F2流程接口可据协议整理；其他结果图不得由CPU示例冒充。没有path-only实际反例，H2不能判断。

## 三项最终判断

| 判断             | 当前结论                           | 原因                                             |
| -------------- | ------------------------------ | ---------------------------------------------- |
| 工程是否完成         | **整体未完成；M1 CPU合同完成，M0服务器部分开放** | 真实CUDA、模型加载、独立reset/执行和恢复链未验                   |
| 研究假设是否获支持      | **H1–H4均尚不可判断**                | 新任务没有真实Level4实验；Level3排序反转仅提供诊断动机，CPU测试只证明合同行为 |
| 是否具备进入Level5条件 | **不具备**                        | 未证明真实可行候选存在、Oracle重复选中真实成功，数值/分组/预算/底座尚未冻结     |

当前建议继续Level4 M2/M3。未来若Oracle工程可靠但相对PointGoal收益未成立，应收紧论文主张，再决定是否进入Level5做独立语言贡献；不能以LLM覆盖底座问题。
