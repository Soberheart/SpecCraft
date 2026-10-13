# Level 4 材料归档合同

本合同扩展 `level4.artifacts` 的 `files_sha256`、独占写入和时间戳接口。归档逻辑在 `level4/archive.py`，不参与动作评分、仿真或成功判定。当前盘点与实际结果见 `records/level4_backup_audit_20261012_01/report.md`；其中服务器状态未经当前直接访问核验。

## 清单与状态

`scripts/archive_level4.py inventory` 接收 JSON：

```json
{
  "run": {
    "run_id": "unique_attempt_id",
    "level": 4,
    "task": "development_geometry_collection",
    "method": "fixed_pool",
    "data_kind": "independent_simulation",
    "scientific_status": "failed",
    "stability_basis": "checked retained process exit and ended scheduler state",
    "requirements_checked": false,
    "dependencies": [{"role": "checkpoint", "body_source_id": "weights", "required": true}]
  },
  "sources": [
    {"id": "run", "path": "/verified/stable/run", "category": "raw_run"},
    {"id": "weights", "path": "/verified/resolved/checkpoint.pth", "category": "weights"}
  ]
}
```

路径必须先核验，示意路径不能直接提交。`--metadata-only` 先统计容量；此清单不能用于复制。去掉该参数才会对文件本体生成 SHA-256。运行中或状态无法核验的材料可盘点，不能进入完整复制状态。扫描不跟随软链接，必须登记并显式加入解析后的实体依赖。缓存、Git 凭据和常见密钥文件排除；只选择本项目实验的明确目录，不扫个人主目录。

每个 run 的科学状态与备份状态独立。`source_confirmed / archive_complete / copied / verified / restored / independent_backup` 分开保存。复制回执覆盖复制及校验状态，不改原清单。`run_catalog.json` 连接各 run、逐文件相对路径/大小/hash、服务器路径、本地副本、版本、seed、论文素材、缺项和恢复状态。未知字段为 null 并登记原因，不从文件名猜 seed 或初态身份。

`requirements_checked=true` 只能在全部适用材料合同已审查后设置。通过文件复制不代表模型、原始状态或环境齐全。新清单有生成时间和 retrospective 标记；旧清单保持原样。

## 每轮必须保留

- 小型证据：任务与标定依据、阈值、split、配置与命令、各 seed、候选评分/属性/逐项判定、选择和拒绝原因、时间表、预算、耗时、日志、原始退出记录、环境与清单。
- 核心数据：初态观测/参照、工具轨迹、query grid；开发固定池的重要预测与全池独立执行；正式搜索只对选中动作保留独立执行原始状态，未执行者写 `not_executed`。
- 依赖本体：实际 checkpoint、模型配置、执行源码快照、未提交和未跟踪源码、材料配置/资产/来源、必要外部源代码。版本号/hash/路径不替代本体；大权重可以独立存一份，多个 run 引用同一 hash。
- 论文素材：源表、绘图/渲染脚本及命令、相机、状态或无损帧、图片与成片、run_id 映射。展示平滑不进入量测几何。

成功、失败、unknown 同样保存。缺项必须标明：现有文件待整理、可由现有状态重算、未来需补记录、已丢失且需新实验。最后一种不自动重跑。

## 操作入口

```text
python scripts/archive_level4.py inventory --request CHECKED_REQUEST.json --output NEW_MANIFEST.json --metadata-only
python scripts/archive_level4.py inventory --request CHECKED_REQUEST.json --output NEW_HASHED_MANIFEST.json
python scripts/archive_level4.py copy --manifest NEW_HASHED_MANIFEST.json --output NEW_BACKUP_DIRECTORY
python scripts/archive_level4.py verify --backup NEW_BACKUP_DIRECTORY
python scripts/archive_level4.py pack --backup NEW_BACKUP_DIRECTORY --output NEW_ARCHIVE.zip
python scripts/archive_level4.py unpack --archive VERIFIED_ARCHIVE.zip --sha256 TRANSPORT_SHA256 --max-bytes APPROVED_UNCOMPRESSED_BUDGET --output NEW_RESTORE_DIRECTORY
```

所有目的目录/包均禁止覆盖。复制前后重新核查源集合、内容和目的文件大小/hash；部分失败保留并记录，不清理。压缩包检查内部清单、成员、大小/hash，拒绝路径越界、重复成员、软链接和超出解压预算的包。`pack/unpack` 不代表恢复成功。

传输使用已有 OpenSSH；不使用删除式同步。大文件失败可在本机交互式 SFTP 使用 `reget REMOTE_PATH LOCAL_PARTIAL_PATH` 续传同一次传输的部分文件，随后做完整大小/hash核验；不能对已有完整历史副本执行续传或覆盖。保留控制台传输记录和失败文件。

## 接入与恢复

PBS 脚本在预测/执行前调用 `check-pool --strict` 检查核心输入是否存在。退出回调先保留科学退出码与 qstat，再在运行目录旁创建小型证据副本。`backup_exit_code.txt` 独立记录归档结果，回调返回原科学退出码。小型副本明确不包括权重或原始大轨迹，重要阶段仍需全量清单、复制和恢复检查。当前集成只在本地完成测试，尚未用于新 NSCC 作业。

CPU 恢复入口：

```text
python scripts/restore_level4_evidence.py --kind synthetic --input RESTORED_SYNTHETIC_EXAMPLE --output NEW_RESULT_DIR
python scripts/restore_level4_evidence.py --kind reset --input VERIFIED_RESET_EVIDENCE_DIR --output NEW_RESULT_DIR
```

合成入口重算每候选约束报告和选择，与保存选择核对，输出跨度 CSV、状态 PNG、相机和 hash。真实 reset 入口使用包内的初态 NPZ 重算 reset 量测，与保存诊断核对并离线渲染原始粒子；不访问报告中的 NSCC 路径，不载入 pickle，不放宽 reset 门槛。它只恢复 reset 诊断，不宣称完整模型/仿真管线可复现。

第三份独立备份尚未配置时明确标注；任何本轮本地副本都不授权删除服务器原件。Level 3 全量材料整理继续作为交接事项，当前仅收集 Level 4 使用的依赖。
