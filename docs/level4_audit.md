# Level 4 接手审计（M0）

日期：2026-10-10。本文件依据本地实际源码和本次运行日志；服务器数字另标为交接报告。

## 结论与证据等级

| 对象 | 当前真实状态 | 证据 / 缺项 |
|---|---|---|
| Level 3 三场景 | 交接报告为 18 次选中动作独立执行；本次未重读服务器原始结果 | `../Level3总结与Level4交接_2026-10-10.md`（位于仓库上一级）；scene00 原始 JSON 均值仍待重算 |
| 任务、CPU 几何、三值判定、选择、数据隔离 | 已实际运行 CPU 验证 | `records/level4_cpu_20261010_delivery_02/`；47 项 Level 4、74 项全套 helper 测试 |
| prepare / CUDA predict / 独立 execute | 实现及 CPU 文件接口存在；真实运行未验证 | 本机缺 torch、Taichi、OmegaConf、SciPy、权重及场景 packet |
| 名义重置、原始日志恢复 | 审计入口及合成合同测试通过；真实重置尚未运行 | 两次独立名义执行为下一门槛 |
| 固定池、标定、研究比较 | 未运行 | rho、eta、epsilon 仍 null；split 未冻结；PointGoal 代表目标尚未认证 |
| 可行优先搜索 / 正式测试 | 未接入 / 未运行 | 不把小池选择当作 constraint-CEM 已完成；M2/M3 门槛未达 |
| 科学图表、真实回放视频 | 没有 Level 4 研究结果产物 | synthetic_example 仅 CPU 示例，不能充作论文实验 |

HEAD 为 `57512d4d0bcbb1f61a9498c36c258cc11f4c46a0`。Level 4 目录、配置、脚本和测试为未提交工作；原有 `frame60_alignment.patch` 保留。本次没有 commit、push、改写历史结果或修改 Obsidian。未发现仓库及父目录适用的 AGENTS.md。

## Level 3 的可引用范围

交接记录的 frame60 actual IoU 均值为 scene00 CEM/Random 90.884/88.697%，scene01 80.7939/80.2627%，scene02 84.2954/89.3597%。这些是处理后模拟 occupancy 的辅助相似度，包含工具 mask，不是新任务严格成功率。三个 seed 是搜索重复，不是独立材料初态或训练重复。

scene02 六个已选动作存在预测分数与执行质量排序反转，支持固定池诊断；尚不能推出整个动作空间排序错误、CEM 普遍失败或应重训。scene00 原始 JSON、scene02 历史 PBS Exit_status、18 次完整几何/拓扑审计及可恢复视频，本次均未独立核验。

## 本地环境与资产

实际运行解释器为 `C:/Users/Admin（无密码）/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`：Python 3.12.14，NumPy 2.3.5。WindowsApps 的 python 是 alias，未使用。没有新增依赖。没有定位到可用 Bash；System32/bash 是 WSL 入口，不能冒充已完成 PBS 语法验证。Python 3.9 语法解析通过，不代表已经在服务器 Python 3.9 执行。

递归输入盘点未发现本地 .pth、真实 observations.npz、log.pkl 或 comparison_frame JSON。完整依赖/可执行文件清单、输入盘点、实际哈希在 evidence 目录的 environment.json / inventory.json。

本地 board 资产确实存在，合计 20,340,249 bytes。重算并与 provenance 对照通过：

| 文件 | 实测 SHA-256 |
|---|---|
| board-128.sdf | `11dafe88308ddd73ef1710269772de0f338926b6a7f5592981ddcd5632e0ef83` |
| board-board.obj | `3b450ba3926334799eae09e6a18f38b0577c36d2c9073ad6fedb0458c9d3ebfe` |
| board-board.provenance.json | `cbee236d3b886974a5b487fdbdf709ac0b5c5d4c014333fc0d1970a7b3998f0a` |

OBJ 是从 SDF 用 marching tetrahedra 恢复的 133,034 顶点 / 266,064 三角形资产，并非作者原始 OBJ。运行快照保留实际内容、来源与哈希。ignored 的 sim/sdftoolbox、net/nvdiffrast 源码也纳入快照；已安装外部包记录版本，包二进制/系统 CUDA 仍需独立归档。

## 必须保留并查明的问题

1. `sim/generate/config/materials/dough.yaml` 指定 mu=357.14、lambda=1428.57、rho=1.0、jittered、yield=[0.0005,0.001]，但 `sim/generate/builder.py` 的 yield_lower / yield_higher 都取 index 0。本地有效参数为 0.0005/0.0005。没有静默修复，以免改变已训练/执行基线；必须与服务器实际 builder 及 resolved config 比对。
2. 每循环先 scene.step 再记录 step0，时间标签仍为 step*dt。dt=.002，log 每16步，名义 .032s 间隔；warmup 默认80步后 reset，首记录已多走一个 solver step。1.920s 主时间标签保持，不能自行改主帧；重置初态与日志相位必须从原始数据验证。
3. `net/config/common.yaml` unit_to_meter=.2；部分仿真注释按 .1 描述长度。这些注释不充当换算。运行根据 metadata 与模型实际配置核对，mm = simulation_units * 1000 * unit_to_meter。
4. 初态全材料粒子是特权参照；初始 grid = bbox + [50,50,20]mm padding，只是有限诊断范围。2mm 球半径是开发表示参数，不是实际粒子半径或标定容差。触边/出界为 unknown。
5. recorded frame60 目标 latent 为 PointGoal 的额外输入，不自动证明对 A/B 合理或过程可行。代表目标需按开发任务独立认证。
6. 服务器多场景、PBS、模型/仿真源码可能与本地不同。已有训练 AE/Dyn 历史 hash 分别为 `43e388893e404081d4db953791d8976a94dc38ceec5a5111131a544ce1778a0e` / `a05452f681f89ee26763959ec77db234dc0bdcf2cbd2996f7361beebb563acae`，本次没有读取对应文件，不标为新核验。

## 服务器访问边界

本次只读 `ssh -o BatchMode=yes -o ConnectTimeout=10 NSCC-ASPIRE2A pwd` 实际退出255，返回 `Permission denied (gssapi-keyex,gssapi-with-mic,password)`，保存于 ssh_probe.stderr.txt。说明没有可用非交互登录；不是自动审批拒绝。没有提交 GPU 作业、申请新资源或读取服务器原始实验。本机也无法完成 CUDA 执行。

M0 的本地部分已完成；服务器权重、resolved 配置、代码差异、调度资源、真实初态/时间及历史结果核验仍开放。后续入口见 level4_server_runbook.md。
