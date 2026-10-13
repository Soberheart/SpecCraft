# Windows launcher: prefer installed Python, then the bundled Codex runtime.
[CmdletBinding()]
param(
    [string]$Message = '',
    [switch]$DryRun,
    [switch]$Yes
)
$ErrorActionPreference = 'Stop'
$taskPythonCandidates = @()
$taskPyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($taskPyLauncher) { $taskPythonCandidates += @{Exe=$taskPyLauncher.Source; Prefix=@('-3')} }
$taskPythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($taskPythonCommand) { $taskPythonCandidates += @{Exe=$taskPythonCommand.Source; Prefix=@()} }
$taskBundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $taskBundledPython) { $taskPythonCandidates += @{Exe=$taskBundledPython; Prefix=@()} }
$taskSelectedPython = $null
foreach ($taskCandidate in $taskPythonCandidates) {
    try {
        & $taskCandidate.Exe @($taskCandidate.Prefix) -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) { $taskSelectedPython = $taskCandidate; break }
    } catch {
        # Windows Store aliases and unavailable launchers are not usable Python runtimes.
        continue
    }
}
if (-not $taskSelectedPython) { throw 'Python 3.9+ not found. Install Python and retry.' }
$taskScriptArgs = @((Join-Path $PSScriptRoot 'sync_github.py'))
if ($Message) { $taskScriptArgs += @('--message', $Message) }
if ($DryRun) { $taskScriptArgs += '--dry-run' }
if ($Yes) { $taskScriptArgs += '--yes' }
$taskOldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $taskSelectedPython.Exe @($taskSelectedPython.Prefix) @taskScriptArgs
    $taskSyncExit = $LASTEXITCODE
} finally {
    $env:PYTHONIOENCODING = $taskOldEncoding
}
exit $taskSyncExit
