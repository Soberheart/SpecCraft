# Level 4 技术决策记录

2026-10-10，基于当前交接和CPU验证；不替代后续开发证据。

| 决策 | 依据与影响 |
|---|---|
| 复用level4模块及Level3工具转换/采样 | 保持既有接口；没有重建平行系统 |
| 固定world_axis_extent、whole_material、mm | 可解析量测；刚体旋转可能满足，明确不作纯塑性/拓扑主张 |
| privileged初态参照 | 当前完整参照来自仿真粒子；未来可见状态参照需另验 |
| B实际以冻结dense raw schedule为主，common另报 | 防止稀疏漏峰冒充过程满足；无连续时间保证 |
| 逐项S/V/U、已知违例优先，初态不可信全部U | 缺数据不通过；已知错误不能被unknown掩盖 |
| 假接受与实际未知分开 | S→V证明观察到违例，S→U只是未认证，分母和零分母null明确 |
| 预测端不接execution | 可识别packet损坏/替换不能影响选择；PointGoal目标只影响其自身 |
| 执行前清单+执行端hash关联 | 阻止后补选择记录伪装prospective；全池诊断支持同一清单 |
| 保存untracked源文件内容、ignored资产和依赖源码 | git diff/文件名不够复现；包/CUDA/权重另需稳定归档 |
| failed attempt保留，resume严格hash | 不清空/覆盖失败或静默改变网格和解码参数 |
| 两次独立名义重置先行 | 不把重复同一路径算独立执行；对齐源码/配置/环境和暂定容差 |
| yield_higher现有index0差异先登记 | 直接改index1会改变模型所学/历史模拟基线；先比对服务器，必要时另行决策 |
| rho/eta/epsilon保持null | 无真实开发证据，不能捏造标定或成功率 |
| M4搜索和M5正式对照守门 | 未完成实际可行性/预测可靠性诊断，不提前用更多CEM迭代掩盖问题 |

改动均位于Level4评价、候选采集、证据留存和测试/文档范围。没有重训、调用LLM、增加力触/新平台、引入未经验证的拓扑硬约束或改变Level3历史结果。

2026-10-12 共享初态后续核查：真实作业 25770711.pbs101 的两次进程均准确恢复同一完整动态状态，实际材料静态参数/映射一致；首个日志帧最大差为 0.000014901161193847656 mm，低于未改动的 0.001 mm 门槛。依据为已下载且重新校验的状态文件本体，报告位于 `records/level4_shared_input_20261012_10/shared_result_review.json`。

该结果只通过共享初始化诊断。旧输入的初态观测/latent 不再用于新共享初态；新入口从本次 attempt1 的首个日志帧重新预处理、Renderer 和 AE 编码。原工具点云时序与 PointGoal frame60 代表目标独立声明来源，尚不构成新初态下的工具接口认证或任务可行证据。`execute` 对新共享初态池保留拒绝门槛，待候选工具位姿与材料恢复接口另行核验。首个日志帧作为时间零点保持不变，主 frame60 为 +1.920 s，辅助 frame62 为 +1.984 s；保存的恢复状态在此零点前 0.002 s。

本轮新准备作业只生成初态观测、AE 编码和 16 个开发候选配置；不运行 Dyn 或新模拟，阈值仍为 null。CPU 全套 147 项通过，冻结传输副本另有 1 项从用户主目录执行的回归测试通过；GPU 新输入阶段尚未提交。历史检查清单与未提交草稿均保留。

2026-10-13 后续实际状态：输入作业 25786017.pbs101 失败，NumPy2 派生 pickle 的 `numpy._core` 引用在冻结 NumPy1.24.4 上不可读；未完成 AE 或候选池准备。改用 NPZ/JSON 逐字段精确传输，由预处理主机创建自己的派生 pickle；不修改服务器依赖或历史源日志。qsub 前新增真实服务器 CPU 字段/参照及 pickle 往返检查，失败禁止 GPU 提交；compute 再验证并记录。153 项本地测试及 2 项冻结传输测试通过，服务器 NumPy1 实际检查和修复后 GPU 阶段待运行。见 `records/level4_shared_input_20261013_11/report.md`，失败科学状态与校验成功分别登记。

2026-10-13 输入作业 25786343.pbs101：实际下载的登录/计算节点首帧往返报告均通过（NumPy1.24.4）；原预处理、Renderer/AE 已保存新首帧观测与编码。之后 `checked_initial_input` 延迟导入缺失的 `level4.archive`，导致退出1，候选池未完成。交付11漏了此源码依赖。不能重标该作业成功，也无需重复GPU编码。

新增完整Level4包覆盖和函数内延迟导入的静态依赖闭包检查；检查放在GPU处理之前。新增CPU只读续跑入口，仅使用hash核验后的保存编码/材料/工具时序。它在新目录复制原快照并记录实际CPU环境，保留原GPU环境证据；该CPU封存不能用来证明GPU环境兼容，后续预测执行仍须独立封存/核验。初态、时间、网格、候选16/seed0、未标定阈值不改。

本地隔离进程实际生成16候选/1,555,260查询点（8.906s），六组保存输入的dtype/shape/数值一致；未加载torch/Renderer/模拟器。再从312文件证据包恢复到另一个D目录，重算初态占据、参考轴量测和动作采样，结果一致，生成真实4163粒子图。该结果是CPU候选准备和恢复通过，不是候选预测/独立执行或任务成功。157项全套测试和3项冻结传输测试通过；登录服务器CPU续跑尚未执行。详细报告 `records/level4_shared_input_20261013_12/report.md`。所有本地失败尝试和服务器原失败证据保留，第三份独立备份未配置。

2026-10-13 服务器 CPU 续跑更新：实际读取下载目录 `D:/SpecCraft/level 4/shared_input_cpu_20261013_e01aeb06f002455cb077d0360a713f66`，7项传输文件及312项包内文件均重新核验。CPU退出0，28.528835644s，生成16候选和1,555,260查询点。六组保存编码/初态/工具输入完全一致；原输入GPU作业25786343仍是失败，CPU成功单独登记。代理没有实时SSH访问，当前服务器目录存在性不作为已直接核验。

本地与服务器候选动作、初态参考和query数组完全一致。只有candidate_014 normalized L2动作成本差1 ULP（1.1102230246251565e-16）；独立标量math.fsum/sqrt核对通过。保留服务器池原值和hash，不修改历史数据、物理任务阈值或重置门槛。新接口诊断按预定候选ID选000/001，不使用cost或执行真值选动作；新环境准备须逐项精确核对动作和全部数值输入，成本仅允许独立标量结果2 ULP内的算术差异并单列记录，不是规划选择容差。

从服务器CPU证据ZIP解包到 `D:/SpecCraft/level 4/server_cpu_package_restore_20261013_13`，隔离进程使用包内源码重算真实初态占据、u/v支撑跨度（各63mm）和16动作，完全一致；生成4163真实粒子的CPU图。恢复耗时6.25s，不依赖NSCC路径、不加载GPU/模型。仅证明已保存真实初态和CPU衍生池可恢复，不证明未来候选任务可行或完整GPU环境已恢复。

下一步交付 `records/level4_candidate_interface_20261013_14` 是两候选独立接口诊断，未提交。1 GPU、30min上限，两独立子进程；完整共同pre-action材料状态/静态材料映射、首次pending工具targets、63帧实际solver工具位姿和原始材料轨迹留存。仪器观测不调用额外process_input/solver step。帧级原始数据即时保存，失败也留存；其余候选明确not_executed，任务阈值仍null，shared_saved_state普通execute门槛仍保留。全套167项测试、冻结交付4项验证通过。13号未提交工具草稿、14号首次路径断言验证失败均保留；后者用resolved恢复目录修复测试断言，与科学计算无关。

2026-10-13 实际接口作业25787714.pbs101更新：重新核验下载的16项传输文件和452项ZIP内部文件。名义candidate_000仿真退出0，63帧完整；阶段与作业因接口后处理拒绝而退出1，candidate_001未执行，所有Dyn预测未运行。完整reset/首次动作前材料与地面状态精确一致，实际静态材料映射一致；logged0材料误差0.00001043081283569336 mm，低于未改动的0.001 mm。

实际solver手指位置与legacy ee对称重建摘要最大差0.004348531365394592 mm（frame56），默认严格审核失败；63帧legacy ee摘要却与旧名义日志逐位一致。源码表明get_state读取当前solver索引，而EE读取latest字段后重建；latest在post_coupling(f)更新，advect写f+1。旧采集没有exact previous/latest数组，不能靠插值证明全部差异来自子步时间滞后。旧作业保持失败，0.001 mm限值不放宽，旧模型工具点云时间对齐仍not_certified。

新交付15只准备尚未执行的candidate_001，复用旧nominal actual solver轨迹做离线参照。采集只读追加exact current/previous/latest与索引，审核保留legacy摘要变换检查，另用same-index物理参照和latest==previous精确检查。该分支不是模型输入认证，candidate_pool_released始终false，不能用执行真值帮助在线规划选动作；固定ID来自此前预定001。沿用1GPU/30min上限，零Dyn/AE/Renderer、不重复名义仿真，尚未提交。

170项本地测试和4项冻结交付验证通过，实际PowerShell5.1解析、Python3.9语法和PBS LF通过；SSH/scheduler mock，不代替GPU结果。真实失败包已恢复到独立D目录，63帧全部嵌套字段dtype/shape/value与完整日志核对一致，并从原始粒子生成frame0/frame60图，复现严格接口拒绝；这是失败数据CPU恢复通过，不是任务成功或完整GPU环境恢复。报告records/level4_candidate_interface_20261013_15/report.md。旧catalog/历史manifest保留，新增条目和later_evidence_updates登记事实；第三份独立备份未配置，不清理NSCC原件。

2026-10-13 作业25789650.pbs101核查：candidate_001仿真退出0，143.67688453709707s，63帧完整；阶段因原审核器错误地假定global solver counter不重置而退出1，历史失败不改。传输16项/ZIP内部222项重新核验；完整reset、静态材料和首次动作前材料/地面一致，logged0误差0.000011920928955078125mm，原0.001mm门槛未动。exact latest==previous在63帧两指位置/旋转全部逐位成立。

实际父runtime代码与原request hash核对显示：接触触发的既有Topology.check会Scene.set_state，重置global counter，并可能额外保留一次Scene.step。Scene.t还计入临时检查分支，不能替代累计物理时间。旧事件历史未采集，不能后补或从63帧精确恢复；frame60和原标签1.920s不改，但累计物理时间认证unknown。没有关闭拓扑模块、改控制器或把未知时间解释成已通过。

完整离线诊断另查出同保存帧标签下工具轨迹数值超限：实际左指对名义候选变换最大0.019054487347602844mm（56帧），开度差0.047332048416137695mm（60帧）；不是只修时间审核就能认证工具输入。日志重建与exact latest还差0.002135336399078369mm。物理时间未认证时不能声称同帧就代表同累计物理时刻，也未证明每项漂移的具体因果。任务阈值继续null，所有Dyn预测未运行，其余14候选not_executed，candidate_pool_released=false。

CPU新增局部时钟检查/全量离线差异报告；新增只读step/get_state/set_state/reset事件采集和弱引用snapshot来源记录，临时分支回滚、额外保留积分计数、未知恢复源unknown。没有额外执行step，173项全套测试通过；事件采集未部署或GPU运行。实际失败包独立CPU恢复、63帧全部字段核对、原始粒子跨度/回放/曲线/PDF生成通过，原作业保持失败，完整GPU环境恢复待完成。报告records/level4_candidate_interface_20261013_16/report.md。后续若需确证累计时间，须另标新诊断补采集，不能重跑旧15号入口冒充补全旧事件。

2026-10-13 为补齐不可恢复的旧事件缺项，准备17号单次显式candidate_001重复诊断，尚未提交。增加独立事件重放：source snapshot、state restore、临时分支回滚、保留积分和每帧事件前缀逐项校验，未知来源拒绝。沿用normal/personal 1GPU/30min上限，只重复001，零Dyn/AE/Renderer，名义不重复；旧失败状态、材料、门槛、frame60/标签1.920s不改。新exit0仅表示采集/重放完成，不证明task或model-tool通过，candidate_pool_released始终false。本轮185项全仓测试通过，其中148项Level4、27项Level3回归、10项已有其他工具测试；冻结交付4项通过。首次冻结测试旧job_name断言错误日志保留，修正断言后通过，冻结运行工具包未改。封存首次把Level3计入Level4的统计断言在复制前拒绝，首版脚本保留，分类按实际模块校正；实际测试日志未改。工具包SHA1fa1c129f54bf88197b23dd914b67251795d8684b88030c650e0a7952d86501c，367依赖本体已核对；见records/level4_clock_diagnostic_20261013_17/report.md。

2026-10-14 实际补录作业25804224.pbs101核查：下载的F/退出0记录、16项传输文件和224项ZIP内部文件重新校验；candidate_001仿真退出0、96.266s，诊断阶段退出0仅表示采集完成。2074事件独立重放通过，35次state restore、1995次含临时分支的Scene.step。从frame25开始，多保留积分与原标签不一致；frame60从保存frame0起的保留积分时间1.980s（从共同pre-action reset起1.982s），标签1.920s不变；frame62为2.044s，标签1.984s不变。最后保存帧累计多30步/60ms，封存Topology.check末尾恢复state后scene.step与记录一致；不据此主张所有工具漂移均已解释。

工具相对同保存帧标签的名义变换仍超限：左指最大0.019054487347602844mm、开度0.047332048416137695mm。名义参考未记录完整时钟，模型工具时间对齐仍not_certified；旧25789650失败及时间unknown保留，0.001mm、frame60、阈值未改，候选池未放行，Dyn未运行。新增离线counter/event-prefix绑定、CPU恢复脚本和4项测试；152项Level4测试实际通过。使用272项冻结科学运行文件重建本地独立runtime，重现所有事件报告与工具逐帧记录；真实原始材料跨度/图/CSV恢复通过，367依赖本体重新核验。见records/level4_clock_review_20261014_18/report.md；完整GPU环境及第三份独立备份仍待完成。本轮没有提交新作业。


## 2026-10-14 prospective topology transaction v2

A new Level4 adapter removes the reviewed topology method final retained integration and restores physical state plus pending inputs/latest tools on normal and exceptional exits. Solver counter normalizes to0; logical clock rolls back through observed scene restore. Graph diagnostics remain. Forward-only; full MPM adjoint history is unsupported. This changes prospective execution semantics relative to legacy model data; model alignment remains unverified. Fixed000/001 delivery in records/level4_transaction_20261014_19 is sealed but not submitted. No thresholds/time labels changed and no historical experiments edited. CPU validation is not GPU certification; candidate pool remains unreleased.


## 2026-10-14 job25807139 failure and prospective v3 retry

Collected job25807139 exited1 before the first simulation subprocess because the runner attempted to exclusively create candidate_status.json twice. No simulator/Dyn ran. The actual16 transfer files and14 internal entries were reverified; original frozen runner reproduced the collision with zero child calls. Preserve the failed job and v2 delivery. The repaired runner saves distinct immutable status snapshots. Additional review found that normalization of the solver counter to0 must rebase the original previous MPM/tool state to the last ring slot; corrected under execution_version v3. All163 Level4 CPU tests and5 frozen transport/closure checks passed, including11 frozen runner/rollback tests. SSH/scheduler calls were mocked. New retry package records/level4_transaction_retry_20261014_20 is not submitted and allows the same two predeclared cases only,1GPU/30min, no Dyn/AE/Renderer. No task threshold/frame changed; CPU checks do not certify GPU or model alignment.

## 2026-10-14 作业25808237失败与零仿真CUDA诊断

核对实际下载的 F/退出1记录、16项传输文件、20项ZIP内部证据与解包本体。candidate_000 子进程在 Taichi cuInit 初始化处失败，没有创建场景、材料帧或完成仿真；candidate_001 和其余14候选未执行，Dyn未运行。distinct immutable status snapshots 修复已在实际证据中确认。历史失败保留，不将校验通过改为实验通过。

nvidia-smi 查询成功，但不能证明 CUDA 计算可用。PBS 分配 /dev/nvidia3，candidate 配置 cuda_id=0，实际封存 builder 会覆写 CUDA_VISIBLE_DEVICES/TI_VISIBLE_DEVICE；失败运行未记录继承 mask，因此绑定覆盖只是未验证假设，驱动/库或导入顺序故障也待排查。未改写历史 builder 或更换未分配的物理设备。

新增 allocation-preserving CUDA 诊断：driver、Torch 标量、Taichi 标量、导入实际冻结仿真模块后 Taichi，四个隔离子进程，保留调度器 mask，PBS/单可见设备守门、禁用 fallback、保留失败与超时。无 GenerationWorkspace、scene、物理 step、模型权重推理；任务0.001mm/frame60/阈值不改。新交付 records/level4_cuda_diagnostic_20261014_21 尚未提交，请求同一 normal/personal 路由1GPU/10min；零仿真、零Dyn。171项Level4 CPU测试、4项冻结交付检查通过，实际PowerShell5.1/Python3.9语法检查通过；SSH/PBS/CUDA全为mock，真实GPU诊断未运行。现有备份/catalog只追加新核查和交付条目，独立备份及完整GPU环境恢复仍待完成，候选池未放行。详见该目录 report.md。

## 2026-10-14 作业25808653实际CUDA通过与PBS绑定修复

实际下载的F/退出0记录、16项传输文件（284804 bytes）、23项ZIP内部文件与解包本体重新核验通过。driver、torch、taichi、generator_import_taichi四个隔离分支均退出0、无超时，耗时分别0.417740/15.749180/12.544197/33.729560s。新节点x1000c1s1b0n0继承GPU UUID mask未变，driver只见一张卡、Torch标量3、Taichi CUDA标量7。生成器分支只导入模块，未创建GenerationWorkspace/场景，零新仿真、零Dyn；不证明候选接口、模型或任务成功。

PBS设备节点/dev/nvidia2与nvidia-smi可见index0属于不同索引命名空间；两者用UUID核对，不能要求索引数字相同。初次本地审计误设index2而失败，原审计脚本与更正记录保存。旧失败25808237运行于另一节点x1000c0s4b0n1且未记录继承mask，不能声称这次通过证明旧失败唯一由mask覆盖造成。builder覆写两个可见设备变量的源码缺陷已确认，按当前证据修复该缺陷。

新增level4.gpu_binding仅在显式选择的新capture入口替换导入的get_scene callable。核对封存builder hash和实际加载函数，AST仅去掉两条mask覆盖，并在原us.init外增加守门；初始化参数及后续场景构造AST不变，原文件不写。保留PBS mask，只用其逻辑device0；创建场景前保存allowlist环境、driver logical0 UUID、后端与前后绑定记录，缺PBS、多设备、UUID不符、初始化失败或CPU fallback拒绝。不是os.environ映射monkeypatch，也不更换未分配设备。v3 topology transaction的行为不改，仍需实际GPU验证和旧模型工具/时间接口对齐。

184项Level4 CPU测试实际通过（40.537s、退出0）；新11项绑定测试及2项runner集成回归包含真实封存builder AST和mock CUDA。冻结22号交付另有24项隔离CPU回归及6项传输/闭包检查通过，PowerShell5.1解析/BOM、Python3.9语法、PBS LF通过。382项依赖文件本体重新核对；SSH/PBS/GPU测试调用为mock，本地bash检查未运行，服务器qsub前有bash -n守门。保存本次真实CUDA环境证据，独立目录CPU重算报告仅认证环境证据恢复；完整GPU软件环境恢复和第三份独立备份仍待完成。

下一步22号包只允许固定000/001各一次前瞻接口仿真，normal/personal、1GPU/30min、无Dyn/AE/Renderer。由于旧v3作业在创建场景前失败，这次需要新的v3名义参照，request明确no_repeated_nominal_simulation=false并记录原因。所有旧实验和失败保留。0.001mm、主frame60/+1.920s、辅助frame62/+1.984s、任务阈值null不变；候选池未放行。工程未完成、研究假设尚未支持、Level5尚不具备条件。新作业尚未提交；详见records/level4_gpu_binding_20261014_22/report.md。

## 2026-10-16 作业25837718固定候选对实际执行

实际下载的F/退出0记录、16项传输文件（27972433 bytes）、诊断ZIP及315项内部证据文件重新校验通过。预先固定的candidate_000/001均独立执行退出0，分别101.134s/58.574s；各有63帧工具轨迹、63帧原始材料帧和`scenes/0000/log.pkl`。两候选共享初态`reset_returned.npz`逐数组精确相等，首动作目标在solver step前记录，14个其余候选明确not_executed，预测未运行。实际新仿真2次、Dyn 0次，candidate_pool_released保持false。

新入口的PBS绑定守门在两个候选均通过：继承UUID mask `GPU-e3b3b9a0-5830-1789-3fe9-9fa9dc5de1e7`，driver logical0 UUID一致、cuInit=0、单可见设备、Taichi CUDA，初始化前后环境不变，原builder hash保持。两个时钟事件报告与保存标签匹配，但累计物理时间仍unknown，不能据标签宣称相同累计物理时刻。

工具接口离线诊断仍为violated，未改0.001mm限值：summary opening最大0.008532mm（frame32）；左右物理手指最大0.004108/0.004150mm（frame26）；summary手指/latest约0.002149mm。模型输入对齐未认证，任务约束未评价，因此不能放行候选池或进入预测、校准、可行优先搜索。PBS的`resources_used.ngpus=0`与实际CUDA/场景成功记录不一致，作为调度器统计差异留档，不重分类科学结果。详见`records/level4_gpu_binding_20261014_22/pair_report.md`和`pair_result_review.json`。

实际固定对已归档到`D:/SpecCraft/level 4/gpu_binding_pair_backup_20261016_22_retry`：518个文件、106391006 bytes，逐文件校验通过；从备份解包到`D:/SpecCraft/level 4/gpu_binding_pair_restore_20261016_22_retry`并在CPU隔离进程重跑固定对审计，结果与原始审核完全一致。首轮归档在复制后因恢复路径假设错误停止，目录`gpu_binding_pair_backup_20261016_22`及失败记录保留，未删除历史。C/D仍为同一物理磁盘，第三份独立备份未配置，NSCC原件保留。


## 2026-10-16 决策D-1：工具接口容差0.02 mm（tool_interface_contract_v2，预先登记）

来源：项目总览任务单L4-01与决策记录D-1（用户Celeste批准，建议人项目总览）。在任何D1/D2/D4、S0–S4结果产生之前登记，之后不得根据结果修改。以下第1–4节为D-1原文（标题层级下调一级），最后为执行端实现记录。

### 1. 决策内容

1. Level 4 固定对审核中的**工具接口比较**（手指物理变换、current/latest、summary/latest、summary 左右位置、summary 开口）的通过门槛，由 0.001 mm 改为 **0.02 mm**。以新版本合同 `tool_interface_contract_v2` 发布。
2. **共享初态复位门槛 0.001 mm 不变**。实测 0.0000149 mm，已经通过。
3. 旧合同 v1（0.001 mm）下的全部判定原样保留，包括作业 25837718 的 `tool_transform_interface=failed`。v2 判定写入新文件，不覆盖旧报告。
4. 放行范围仅限工具接口这一道门。以下各项状态不变：
   - 模型对齐 `model_alignment` 仍为 unverified，由 D4 裁决；
   - 累计物理时间仍为 unknown，由 D4 和时钟复核处理；
   - 任务阈值 ρ/η/ε 仍为 null；
   - 主端点 frame60/+1.920 s 不变。

### 2. 依据

**作业 25837718 实测最大值**：手指 0.00415 mm（frame26），summary 开口 0.00853 mm（frame32）。

**与项目中其他尺度的比较**：

| 参照 | 数值 |
|---|---|
| 粒子平均间距（4163 粒子 / 60×60×20 mm） | 约 2.6 mm |
| 量测网格（开发值 1 mm，敏感性审计最细 0.5 mm） | 0.5–1 mm |
| 粒子支撑半径 | 2 mm |
| Franka 标称重复定位精度（真机，需核实） | 约 ±0.1 mm |

**为什么取 0.02 而不是 0.05 mm**：门槛要能拦下已知的真实错误。
- 单位错误、yaw 反号会产生毫米级误差，0.02 mm 可以拦下。
- 修复前的旧语义下，开口漂移 0.047 mm、手指漂移 0.019 mm。0.02 mm 能拦下开口漂移，手指漂移处在边界上；0.05 mm 两者都拦不下。
- 当前 v3 语义的最大值 0.0085 mm 低于 0.02 mm。

**累积问题（用户提出）**：
- 接口误差在单个动作内不随帧数单调增长：最大值出现在 frame26/32，而不是 frame62。
- 多步时每一步的工具轨迹都按指令位姿重新生成，误差不会传递到下一步。
- 最坏情况按线性叠加估算：15 步 × 0.0085 mm ≈ 0.13 mm，仍小于量测网格。
- 真正会累积的是 DoughNet 开环预测误差，由闭环重观测和 L5 的校准处理，与本门槛无关。

### 3. 验证条件（S0–S4，作为本决策的配套实验）

| 编号 | 做法 | 目的 |
|---|---|---|
| S0 | 同一动作重复仿真 2 次 | 仿真自身的噪声底线 |
| S1 | 工具轨迹整体偏移 0.02 mm（x、y 各一次） | 门槛大小的偏差对属性的影响 |
| S2 | 偏移 0.1 mm | 真机量级的参考 |
| S3a | S1 的属性差随帧数（0–62）如何变化 | 单动作内的传播 |
| S3b | 多步序列（3–10 步），每步加 0.02 mm 偏移 | 跨步累积。若多步执行尚未实现，转入 L5 补做 |
| S4 | 在接近"断/不断"边界的动作上做 S1 | 临界点附近的敏感性 |

**通过标准**（预先登记）：

1. S1 和 S3 引起的属性变化 < ε/10。
2. ε 尚未标定前，暂用以下替代线：跨度和颈宽变化 < 0.1 mm（1 mm 量测网格的 1/10），且组件数不变。
3. ε 冻结后按正式值复判，两次判定都要保留。

**不通过时的处理**：

| 情况 | 处理 |
|---|---|
| S1/S3 在普通状态下就不通过 | 0.02 mm 失效。收紧门槛，或先查清误差来源再放行，并报告总览 |
| 只有 S4（临界点）不通过 | 不收紧门槛，作为"临界敏感性"证据记入 L5，由校准裕量处理 |

### 4. 论文中的写法

> Tool-trajectory deviation between planned and simulated execution was at most 0.0085 mm, below a pre-registered tolerance of 0.02 mm; perturbations of this magnitude changed measured attributes by < X mm (S1–S3).

X 由 S1–S3 的实测结果填入。

### 执行端实现（L4-01 A4）

- `level4/tool_interface_contract.py`：v1（0.001 mm，历史门控键）与 v2（0.02 mm）。v2 对全部14项比较中的位置/开口类一律用0.02 mm门控，包括v1只记录不门控的current/latest、summary/latest、latest/previous，因此覆盖范围不比v1窄；四元数限值1e-6不变（D-1未涉及）。缺项、多项、非有限值一律拒绝判定。
- `scripts/audit_level4_pair_contract_v2.py`：先以hash核对的冻结v1审计脚本复现原`pair_result_review.json`（除证据路径外一致，314项证据hash全过），再用当前`diagnose_tool_schedule`从原始轨迹/日志重算诊断并要求与保存值逐项完全相等，然后分别按v1、v2判定。v1仍为`failed_v1`；v2为`passed_v2`，最差门控量0.008532 mm（summary opening，frame32）。
- 输出写入新文件 `D:/SpecCraft/level 4/L4-01_phaseA_20261016/A4_contract_v2/pair_result_review_v2.json`；原v1审核文件sha256 `c2b8e66c…4511668` 运行前后不变。`model_alignment=unverified`、累计物理时间unknown、任务阈值null、frame60/+1.920 s均不变；候选池 `candidate_pool_released=false`，正式执行仍须L4-01 Phase A通过并冻结Phase B请求。
