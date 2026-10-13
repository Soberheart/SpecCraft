# Level 4 NSCC 最小开发实验运行说明

2026-10-12 最新状态（用户回传，完整raw证据包待下载复核）：作业25769308.pbs101已通过计算节点封存核验及42项CPU测试；名义candidate_000的CUDA模拟退出0，耗时89.774s，但初态核验失败，PBS退出1。两边均4163粒子、63帧、标签t0至1.984s，最大坐标误差0.002675mm，超过不变的暂定0.001mm门槛。不能据此认定重置通过或直接放宽；reset_2与reset_audit尚未运行。当前应下载 `records/level4_initial_reset_diag_20261012_01.zip` 并核对hash，不能直接重新提交下方旧入口。详细记录见 `records/level4_server_followup_20261012_12/` 和 `_13/`。

2026-10-12 后续状态：服务器 Python 3.9.25 的原 47 项 CPU 测试已通过，真实输入 preflight 已下载审查。board 来源记录差异仅为绝对路径改成相对路径；实际 SDF/OBJ 未变。名义作业 25768191.pbs101 在计算节点因缺少 git 退出127，Git路径重跑 25768683.pbs101 进一步确认计算节点没有 /usr/bin/git，均未进入模拟。

无Git入口的作业25768898.pbs101也在模拟前失败：期望Python3.9.25，实际计算节点版本不同。环境诊断作业25769075.pbs101确认：同一虚拟环境路径在登录节点为Python3.9.25，在计算节点为Python3.9.18；枚举包版本全部一致。该诊断退出2、未运行模拟，不是重置成功。

当前重跑入口是 `records/level4_server_followup_20261012_09/retry_reviewed_compute.py`。它读取服务器上的实际compute/report.json，只接受已观察到的完整Python版本字符串及相同包版本，复核该报告所引用的原清单哈希、作业和节点身份，并把报告实际内容及哈希归档。新清单分别记录提交环境和执行环境；PBS上下文严格使用执行环境，登录上下文严格使用提交环境。没有按主版本放宽检查，也没有重写历史清单。源码/资产、包版本和probe证据变化仍拒绝运行；后续pool快照记录实际执行环境，不能跨环境复用reset门槛。

该入口已在NSCC执行：登录节点55项CPU测试和封存检查通过；计算节点42项CPU测试通过，首个CUDA模拟退出0，但初态核验失败。原服务器源码及旧失败结果保留。本地82项回归和空PATH/环境分角色子进程验证已通过，随后新增的独立reset诊断工具另有5项CPU测试通过；不能把不同版本测试数相加描述为一次完整回归。

下文第1/2节是初始交付说明，第3/4节的直接 qsub 命令仍要求计算节点有Git；当前应使用冻结源码入口，后续开发池也须复用已验证的冻结源码与清单。不能把下面旧命令再次直接提交当作本次修复。

以下命令尚未在NSCC运行。本次BatchMode认证失败；本机没有CUDA/Taichi/真实输入。需要既有授权交互服务器会话，不需要新模型、重训或新平台。先同步并核对本次Level4源码，再使用已有账户允许的队列与小规模资源。PBS请求normal/personal、1GPU/8CPU/64GB/4h不是已核实分配；若当前授权/队列不适用，不能盲目提交。

## 1. 安全同步本次交付

本地交付archive在 `records/level4_cpu_20261010_delivery_02/level4_delivery.zip`，只含Level4相关模块/脚本/配置/测试/文档及manifest；完整基线/资产快照另在同一证据目录的runtime_snapshot。没有自动push或上传。

可在已经可以认证的终端上传（PowerShell）：

```powershell
scp 'C:/Users/Admin（无密码）/Documents/ChatGPT/Doughnet+LLM/Doughnet-LLM/records/level4_cpu_20261010_delivery_02/level4_delivery.zip' NSCC-ASPIRE2A:/home/users/ntu/mingjie0/projects/doughnet/records/level4_delivery_20261010_cpu02.zip
```

在NSCC把文件解到新的审查目录；下面的安装脚本先校验全部hash，只安装不存在的文件或复用完全相同的文件。任何现有不同内容会报错并停止，避免覆盖服务器上尚未交接的改动。若发现差异，先比较并在Level4范围合并，不覆盖服务器模拟/Level3基线；随后重新生成实际运行快照与测试证据。

```bash
set -euo pipefail
cd /home/users/ntu/mingjie0/projects/doughnet
source /home/users/ntu/mingjie0/venvs/doughnet/bin/activate
STAGING=$(mktemp -d records/level4_source_review_XXXXXX)
python - "$STAGING" <<'PY'
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
root=Path.cwd();stage=Path(sys.argv[1]).resolve()
archive=root/'records/level4_delivery_20261010_cpu02.zip'
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(name)
    z.extractall(stage)
manifest=json.loads((stage/'delivery_manifest.json').read_text())
for name,digest in manifest['files_sha256'].items():
    data=(stage/name).read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest,name
    target=root/name
    if target.exists() and target.read_bytes()!=data:
        raise RuntimeError('Existing different file; review and merge first: '+name)
for name in manifest['files_sha256']:
    target=root/name
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(stage/name,target)
print('Level4 sources installed/reused without overwriting differing files',stage)
PY
```

注意archive中必须含本次的verify_level4_cpu.py、audit-reset、freeze-selection和execute --all-candidates接口；仅从当前GitHub main拉取不会得到这些未提交文件。

## 2. CPU / 输入 / 队列 preflight（登录节点只作检查）

```bash
cd /home/users/ntu/mingjie0/projects/doughnet
source /home/users/ntu/mingjie0/venvs/doughnet/bin/activate
PREFLIGHT=$(mktemp -d records/level4_preflight_XXXXXX)
git status --short > "$PREFLIGHT/git_status.txt"
git rev-parse HEAD > "$PREFLIGHT/commit.txt"
python --version > "$PREFLIGHT/python.txt" 2>&1
python -m unittest discover -s tests -p 'test_level4_*.py' -v > "$PREFLIGHT/cpu_tests.txt" 2>&1
bash -n scripts/run_level4_pool.pbs
python scripts/level4.py --help > "$PREFLIGHT/cli_help.txt"
python scripts/level4.py validate-pool-config --config configs/level4/development_pool.json
qstat -Qf normal > "$PREFLIGHT/queue.txt" 2>&1
qstat -Bf > "$PREFLIGHT/server.txt" 2>&1
export DYN_ARTIFACTS=/home/users/ntu/mingjie0/projects/doughnet/records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts
python - "$DYN_ARTIFACTS" <<'PY' > "$PREFLIGHT/inputs.json"
import hashlib,importlib.metadata,importlib.util,json,pickle,shutil,sys
from pathlib import Path
source=Path(sys.argv[1]);m=json.loads((source/'metadata.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
scene=Path(m['scene_dir'])
pairs=[(Path(m['checkpoint']),m['checkpoint_sha256']),
       (scene/'config.yaml',m['scene_config_sha256']),
       (scene/'log.pkl',m['source_log_sha256']),
       (Path(m['processed']),m['processed_sha256'])]
for p,expected in pairs:assert digest(p)==expected,str(p)
for relative,expected in m.get('source_file_sha256',{}).items():
    assert digest(relative)==expected,'Source packet/runtime mismatch: '+relative
for package in ('torch','taichi','scipy','omegaconf'):
    assert importlib.util.find_spec(package),package
with (scene/'log.pkl').open('rb') as f:logs=pickle.load(f)
assert logs[0]['step']==0
assert abs(float(logs[60]['t'])-1.920)<1e-8
assert abs(float(logs[62]['t'])-1.984)<1e-8
assert m['unit_to_meter']==.2
assert m['rollout_frames'][-1]==60
from omegaconf import OmegaConf
cfg=OmegaConf.load(scene/'config.yaml');OmegaConf.resolve(cfg)
yaml=OmegaConf.to_container(cfg,resolve=True)
print(json.dumps({'files_sha256':{str(p):digest(p) for p,_ in pairs},
  'observations_sha256':digest(source/'observations.npz'),'model_config_sha256':digest(source/'model_config.yaml'),
  'metadata_sha256':digest(source/'metadata.json'),'raw_times':[float(r['t']) for r in logs],
  'scene_config_resolved':yaml,'packages':{p:importlib.metadata.version(p) for p in ('torch','taichi','scipy','omegaconf')},
  'raw_log_bytes':(scene/'log.pkl').stat().st_size,
  'raw_log_18_execution_scale_bytes':18*(scene/'log.pkl').stat().st_size,
  'free_bytes':shutil.disk_usage('.').free},indent=2))
PY
sha256sum sim/generate/builder.py sim/generate.py sim/generate/config/materials/dough.yaml \
  sim/mpm/assets/meshes/processed/board-128.sdf \
  sim/mpm/assets/meshes/processed/board-board.obj > "$PREFLIGHT/simulation_assets.sha256"
python scripts/level4.py snapshot --output "$PREFLIGHT/runtime_snapshot" > "$PREFLIGHT/snapshot.json"
```

metadata与本地历史引用不同、源模型接口不一致、材料builder有效参数变化、时间/单位无法核对时先停止并保存差异。raw log×18只是存储规模估算，不含模型packet、source快照和拓扑规模变化；检查可用空间。尚未从服务器核验的AE/Dyn历史hash见audit；本次preflight须以真实文件登记并核对版本。

## 3. 两次名义独立重置（最先提交）

```bash
RESET_PARENT=$(mktemp -d records/level4_nominal_request_XXXXXX)
RESET_JOB=$(qsub -v "DYN_ARTIFACTS=$DYN_ARTIFACTS,POOL_CONFIG=configs/level4/development_pool.json,STAGE=reset,RUN_PARENT=$RESET_PARENT" scripts/run_level4_pool.pbs)
printf '%s\n' "$RESET_JOB" > "$RESET_PARENT/job_id.txt"
qstat -f "$RESET_JOB" > "$RESET_PARENT/qstat_submitted.txt"
```

作业在两个不同pool中各只执行candidate_000，虽然每pool准备16个配置，不预测/执行其他动作。按0.001mm暂定阈值审计，不放宽以通过。任一失败会保留日志并退出非零。需要查明step0的一步相位与候选初始接触；即使名义两次通过，全池仍逐候选检查初态。

完成后及时保存调度历史，核查exit_code、reset_audit及两个raw log，不把completion marker等同于PBS Exit_status：

```bash
QSTAT_STATUS=0
qstat -xf "$RESET_JOB" > "$RESET_PARENT/qstat_finished.txt" 2>&1 || QSTAT_STATUS=$?
printf '%s\n' "$QSTAT_STATUS" > "$RESET_PARENT/qstat_finished_exit_code.txt"
RESET_AUDIT=$(python - "$RESET_PARENT" <<'PY'
import json,sys
from pathlib import Path
paths=list(Path(sys.argv[1]).rglob('reset_audit.json'))
assert len(paths)==1,paths
assert json.loads(paths[0].read_text())['status']=='passed'
print(paths[0].resolve())
PY
)
export RESET_AUDIT
```

Unknown Job Id只表示历史不可查，不认证退出结果。不可恢复的调度状态缺失要记录；shell exit_code与PBS状态分别登记。

## 4. 16候选开发采集（重置通过后）

不修改源码/配置/环境后继续；PBS通过已CPU验证的validate-reset-gate核对reset证据与当前实际observations、模型配置、checkpoint、metadata、源码及包环境，防止换版本仍沿用旧门槛。该检查只认证名义重复重置，不替代完整时间相位和全池初态审计。

```bash
DEV_PARENT=$(mktemp -d records/level4_pool_request_XXXXXX)
DEV_JOB=$(qsub -v "DYN_ARTIFACTS=$DYN_ARTIFACTS,POOL_CONFIG=configs/level4/development_pool.json,STAGE=all,RESET_AUDIT=$RESET_AUDIT,RUN_PARENT=$DEV_PARENT" scripts/run_level4_pool.pbs)
printf '%s\n' "$DEV_JOB" > "$DEV_PARENT/job_id.txt"
qstat -f "$DEV_JOB" > "$DEV_PARENT/qstat_submitted.txt"
```

all只做prepare/predict/独立execute，阈值仍未标定。不等标定结果占GPU，作业结束释放资源。4h能否容纳16候选尚未测量；先检查名义耗时、磁盘和实际分配。失败不可清空重跑；新attempt或只复用hash相符的成功阶段。

作业结束后在CPU处理保存日志；下列POOL定位仅接受本次DEV_PARENT内唯一pool，不用“全records最新目录”：

```bash
QSTAT_STATUS=0
qstat -xf "$DEV_JOB" > "$DEV_PARENT/qstat_finished.txt" 2>&1 || QSTAT_STATUS=$?
printf '%s\n' "$QSTAT_STATUS" > "$DEV_PARENT/qstat_finished_exit_code.txt"
POOL=$(python - "$DEV_PARENT" <<'PY'
import sys
from pathlib import Path
p=list(Path(sys.argv[1]).rglob('pool.json'));assert len(p)==1,p
print(p[0].parent.resolve())
PY
)
SUMMARY=$(mktemp -d records/level4_summary_attempt_XXXXXX)
python scripts/summarize_level4_development.py --pool "$POOL" --output "$SUMMARY/summary"
python scripts/audit_level4_representation.py --pool "$POOL" --output "$SUMMARY/representation" \
  --spacing-mm 2 1 .5 --radius-mm 1 2 3 --max-cells 32000000
```

半径1/2/3mm仅开发敏感性，不是物理认证。细网格可能超过budget，报错并保存失败，不静默减精度。若需重跑，使用新output目录。额外细网格模型解码尚未实现/运行；只有分类敏感性需要时再加，不能将raw重处理当成预测分辨率已验证。

## 5. 标定后M3前瞻固定池诊断

先根据真实开发结果写 `configs/level4/task_B_frozen_development.json`（当前不存在）及真实evidence hash、split；也冻结PointGoal合理目标构造、失败规则和诊断门槛。不能复制synthetic阈值。既有全池执行后的analyze明确exploratory，不可后补freeze。

新的池按同一prepare/predict入口准备，在执行目录为空时：

```bash
python scripts/level4.py validate-spec --spec configs/level4/task_B_frozen_development.json --experiment
python scripts/level4.py freeze-selection --pool "$POOL" --spec configs/level4/task_B_frozen_development.json --output "$POOL/selection.json"
python scripts/level4.py execute --pool "$POOL" --selection "$POOL/selection.json" --all-candidates
python scripts/level4.py analyze --pool "$POOL" --spec configs/level4/task_B_frozen_development.json --selection "$POOL/selection.json" --output "$POOL/analysis.json"
```

这里POOL必须是**新prepare/predict且尚未执行的M3池**，不能沿用第4节已执行池；execute仍须在GPU allocation内运行，不在登录节点执行。以上命令只定义标定完成后的调用顺序，当前冻结输入不存在，未运行。仅执行方法所选动作时去掉--all-candidates，不能据少量选中动作估计全池混淆矩阵。

M3按实际可行数→预测识别→选中真实结果守门。M4 constraint-CEM/对应Random搜索入口当前不存在，故不提供虚构命令；M5新初态正式测试、F1/F3/F5及V1真实展示也未完成。
