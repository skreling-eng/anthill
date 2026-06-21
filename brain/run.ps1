# Launch brain using its isolated venv (does not use repo-root .venv).
$ErrorActionPreference = "Stop"
$BrainDir = $PSScriptRoot
$Root = Split-Path -Parent $BrainDir
$py = Join-Path $BrainDir ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "brain/.venv not found. Run: powershell -File brain\setup_venv.ps1" -ForegroundColor Yellow
    exit 1
}

Set-Location $Root
& $py -m brain.app @args
