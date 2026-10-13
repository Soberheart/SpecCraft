# SpecCraft 程序同步到 GitHub

GitHub 仓库：<https://github.com/Soberheart/SpecCraft>。原仓库 `Soberheart/Doughnet-LLM` 已于 2026-10-13 更名；GitHub 通常保留旧链接重定向，新的配置应使用 SpecCraft。

项目显示名和仓库名已更新。本地现有 `Doughnet-LLM` 目录、服务器 `~/projects/doughnet`、Python 环境和模型名保留，避免破坏历史记录和脚本中的路径。新克隆的目录默认为 `SpecCraft`。原始 DoughNet 的介绍、引用和许可保留。

## 1. 准备

- 安装 Git 和 Python 3.9+；Windows 可用 `python` 或 `py -3`，Linux 常用 `python3`。
- 配置 Git 提交身份；首次推送通过 Git Credential Manager、GitHub CLI 或 SSH 进行认证。不把密码或 token 写入代码或远程 URL。
- 仓库目前为公开仓库：同步前人工检查文档、路径、日志摘录和未公开研究材料。自动扫描只是常见凭据模式检查，不能保证找出所有敏感内容。
- 脚本只提交运行机器本地的文件；不自动下载或合并 NSCC 上尚未拷回的代码，也不上传实验数据。

如服务器克隆仍使用旧地址，在该克隆内执行：

```bash
git remote set-url origin https://github.com/Soberheart/SpecCraft.git
git remote -v
```

## 2. 日常运行

进入项目根目录，在 Windows PowerShell 中：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_github.ps1 -Message "Update Level 4 implementation"
```

Windows 启动器自动寻找 Python Launcher、Python 或本机 Codex bundled Python。`-ExecutionPolicy Bypass` 只作用于此脚本进程，不修改系统策略。Linux 或直接使用 Python 时：

```bash
python scripts/sync_github.py --dry-run
python scripts/sync_github.py --message "Update Level 4 implementation"
```

服务器上可使用已激活环境中的 `python`，或 `python3`。

第二条会展示修改清单，输入 `y` 才提交并推送。默认包含全部已跟踪修改、删除以及未忽略的新文件；**包括此前已暂存的修改**。它不是只提交本次编辑，也不是双向文件同步。

确认目录和文件清单无误后，可显式使用无交互模式：

```bash
python scripts/sync_github.py --yes --message "Sync SpecCraft code"
```

也可以从其他目录调用脚本：它根据自身位置找到项目，不依赖终端的当前目录。`--repo PATH` 仅用于明确指定另一份克隆或测试仓库。

## 3. 执行过程与安全边界

1. 检查 `origin` 的读取/推送地址，以及 Git URL 重写，必须指向 SpecCraft。
2. 检查当前分支、冲突和未结束的 merge/rebase；展示修改清单。
3. 扫描待上传文件；常见凭据、数据、权重、实验目录和单文件超过 25 MiB 会阻止上传。
4. 用户确认后获取远端当前分支，要求远端历史是本地 HEAD 的祖先。当前分支必须已存在于远端。
5. 检查已有待推送提交，以及暂存内容；提交全部非忽略修改。
6. 再检查待推送历史，然后普通推送当前分支，不推送其他分支或 tags，不使用 `--force`。
7. 对比远端和本地 HEAD，确认上传结果。

`--dry-run` 只做本地工作区检查，不暂存、不提交、不 fetch、不 push；不等于已验证远端权限或所有待推送历史。实际同步会额外检查已有待推送提交。

脚本使用 Git 自身的认证；不读取或存储 token。凭据检测不输出命中内容。自动检查无法替代人工隐私审查，也无法清理已经存在于 GitHub 的历史泄露。

## 4. 常见停止原因

- **远端领先/分叉**：先保存本地修改，查看 `git log --oneline --graph --all`。仅落后时，可在工作区干净后运行 `git pull --ff-only origin main`；分叉需人工决定如何整合，脚本不自动 merge/rebase。
- **上传被拦截**：将数据和敏感文件留在单独存储。`.gitignore` 只影响未跟踪文件，不会移除已提交内容；需要根据实际历史人工处理，不使用强推绕过检查。
- **认证或网络失败**：检查 GitHub 登录与仓库写权限。已有本地提交保留，修复后重跑；不删文件、不 reset。
- **锁文件存在**：先确认没有另一个同步进程，才人工删除 `.git/speccraft-sync.lock`。
- **服务器代码更新**：在服务器的克隆中单独运行脚本。不要两台机器同时编辑并推送同一分支；另一台先更新后再编辑。

数据集、checkpoint、预测缓存和完整仿真证据继续按原归档流程保存；GitHub 保存程序与允许公开的配置/文档，不能替代实验备份。
