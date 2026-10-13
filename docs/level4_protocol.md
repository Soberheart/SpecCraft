# Level 4 最小任务与接口协议

版本：TaskSpec 1.0.0；评价/选择报告 1.1.0。语义及 CPU 行为已验证；数值标定、split、正式预算与协议**尚未冻结**。

## 对象、坐标与属性

首轮任务名为**固定世界轴跨度控制**。对象 whole_material，semantics=world_axis_extent。原点、正交单位向量 u/v 在初态确定并固定，之后不做每帧 PCA/对齐。原点以 mm 表示；span 对平移与轴反号不变，对刚体旋转可能改变。旋转导致缩短仍满足该属性，不能声称已证明纯塑性变窄。

对完整材料支撑 P，s_u(P)=max(p·u)-min(p·u)，r_u(t)=s_u(P_t)/s_u(P_0)，v 类同。占据体素并集的跨度为中心投影跨度加 sum_i abs(u_i)*cell_size_i，包括整个单元支撑。primary operator 为 extrema，不去极值、不删除接触区。weighted quantile、mesh 和 raw particle span 只用于诊断。

当前 privileged_full_material_initial_state 用初态完整原始材料粒子构造参照和预测 t=0 的几何。DoughNet 当前可见初态 observation / latent 与完整参照分别留存。未来观察型参照估计需要独立验证，本轮不宣称 observation-only。

## A / B 最小规格

| 任务 | 硬条件 | 时间 |
|---|---|---|
| A：指定方向变窄 | r_u(T) <= 1-rho+epsilon | T=frame60=1.920s |
| B：变窄并保护另一方向 | A，且 abs(r_v(t)-1) <= eta+epsilon | 实际预先冻结的原始日志时刻 [0,1.920]s |

rho∈(0,1)、eta∈[0,1] 表示任务意义；epsilon>=0 表示量测容差；m>=0 为单独开发标定的预测裕量。预测允许上限再减 m，实际不减 m。等号通过，仅用32倍浮点 epsilon 消除算术舍入，不替代物理容差。m=0 是未加保守机制，不能称为已完成误差校准。

`configs/level4/task_{A,B}_uncalibrated.json` 为模板，null 参照、rho、eta、最小可测跨度及 epsilon 故意使 validate-spec 失败。不能随手填数字然后称为冻结实验。开发池的 geometry collection 不要求先设任务阈值。

```json
{
  "object": "whole_material",
  "semantics": "world_axis_extent",
  "reference_frame": {"kind":"initial_fixed_world","unit":"mm","origin_mm":[0,0,0],"u":[1,0,0],"v":[0,1,0]},
  "terminal_hard": {"id":"narrow_u","rho":null,"relation":"le","hard":true},
  "process_hard": [{"id":"protect_v","eta":null,"relation":"le","hard":true}],
  "soft_preferences": ["minimum_normalized_action_l2"]
}
```

完整字段结构见 task_spec.schema.json；运行时校验另检查有限数、轴长度/正交、时间顺序及语义。JSON Schema 文件可解析；本机没有 jsonschema 包，没有声称执行该包的 schema 校验。

## 表示与时间合同

预测：冻结模型 eval/no_grad，仅初态 observation 初始化，自回归 latent rollout；材料解码类别1..4，0背景。t=0 几何是已披露的特权参照，其后来自模型。执行：每候选新 simulator 子进程，所有原始 obj.particles.pos 的等半径球并集转 occupancy，不用 AE 重建、工具/桌面点或 contact mask。测量器不会凭坐标识别桌面/工具，上游必须保证 material-only，污染/遮挡标记导致 unknown。

simulation 坐标乘1000*unit_to_meter 得 mm；yaw_deg 用度、四元数 wxyz。开发 grid 1mm、粒子球支持半径2mm、padding [50,50,20]mm、max_cells=4,000,000，皆为待敏感性审计的表示选择。原始粒子出界或任一边界层占据，输出 unknown，不能删边界或根据测试未来几何扩范围。2/1/.5mm 与半径敏感性重处理入口不得扩大初始声明范围。

主端点1.920s固定，辅助frame62=1.984s单列且不参与选择、成功聚合或换主结果。common prediction schedule 模板为0,.160,…,1.920；执行 dense schedule 模板为0,.032,…,1.920（61点），实际 prepare 核对原始日志。`match_tolerance_s=1e-8` 是时间匹配舍入量。缺帧、重复、乱序、非有限时间拒绝/unknown；不插值。首次违例是**首次采样检测**；不保证连续时间安全。共同采样评价只作为诊断，不覆盖 B 的稠密实际主评价。

## 判定、选择与信息权限

每项输出 value_mm、reference_mm、ratio value、threshold、tolerance、prediction_margin、raw_residual、violation、status 和 reason。已知违例优先于其他 unknown；无违例但存在 unknown 为 unknown；全部通过为 satisfied。初态参照不可信时硬条件全部 unknown。缺失实际执行保留在全池分母，不给成功认证。

Oracle-Constraint：合法且全部预测硬条件 satisfied 才选择，随后最小 cost，再 candidate_id 破同分。Oracle-TerminalOnly：选择只用终态，最终仍按完整 B 评价。PointGoal：独立代表目标 token 平均 cosine 最大，再 cost/id；不默认增加本方法筛选。没有通过者返回 no_feasible_candidate。约束违例可为未来搜索提供归一化引导，不能用最低违例候选冒充可行。

动作 a=(dx_mm,dy_mm,yaw_deg)，固定工具和闭合时序，边界±[40,40,10]，cost=norm(a/[40,40,10])。当前合法性仅有限位姿边界，不是机器人关节可达/碰撞安全保证。开发池16个动作，含 candidate_000 名义零动作，seed0，截断高斯 std=[8,8,10]，候选数入口仅2..32。

在线 prediction_decisions 不打开 executions 或未来 source logs；只读冻结池的初态/工具/预测 packet 和规格。PointGoal 额外读 z_goal；修改目标 latent 不改变约束选择。freeze-selection 在任何 execution 开始前写不可覆盖 manifest，包含实际规格、预测文件/输入 hash、时间、全部候选和选择。execute --selection 检查清单并登记哈希；全池诊断另加 --all-candidates。analyze 校验关联后才标 prospective，后补清单不获认证。

离线 Execution-Oracle pool reference 只在实际可行候选中找最低cost，用于池内覆盖，不进入预测、选择或搜索，不称全空间最优。

## 实验冻结与失败规则

正式运行前还需冻结：真实模型/配置hash、rho/eta/epsilon/m及各自开发依据hash、minimum span、query范围/表示、重置定义、PointGoal构造与有效性、split、同预算搜索/强基线、可靠性界限/最小有意义效应、资源上限及基础设施重跑规则。既有三场景默认开发；独立初态/hash与task family分组，同初态A/B和seed不跨split。

每候选保留 trajectory.npz、latents.npz（预测）、完整 raw log.pkl（执行）、scene yaml、generate.log、exit.json、complete/failure.json；池保存 config、source_metadata、网格、白名单planner_input、runtime源码/资产内容快照和hash。失败 attempt不覆盖；成功resume同时检查源码、输入、packet、latent/raw log和参数，失败用新 attempt。包版本不替代外部二进制/权重归档。

拒绝与unknown不是成功，区分未执行、进程失败、truth提取失败、任务违例；非零进程退出暂不自动断言为纯基础设施问题，待日志归因。全池混淆矩阵的S/V/U原计数及分母保存；S→V假接受与S→U未认证分开。正式统计按独立初态配对，seed不增加独立样本数，一个初态不报跨初态区间。

## Level 5 预留调用合同

输入：TaskSpec + 仅执行前可用 observation/reference及其质量/权限 + action/model/protocol/budget版本。输出：selected / rejected / invalid_spec / unknown、可选action、逐项预测报告、原因、cost与manifest。评价独立传入execution+同一TaskSpec。保留initial_state_hash/task_family/split、语义标注来源与允许规格集合字段；当前不调用LLM或生成语言训练数据。适配返回字典的接口已经明确，正式冻结底座尚未形成。
