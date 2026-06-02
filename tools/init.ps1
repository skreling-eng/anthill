# Fresh Anthill checkout: install Python deps and external venvs.
# Models/test_data: download_all_models.bat (not part of init).
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\init.ps1
#   init.bat -SkipSage
#   init.bat -SkipFfmpeg
#   init.bat -UploadTestData              # maintainer: push test_data/ to skreling-eng/anthill
param(
    [switch]$SkipVenvs,
    [switch]$SkipFfmpeg,
    [switch]$SkipSage,
    [switch]$UploadTestData,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name. Install it and re-run init."
    }
}

Write-Host "=== Anthill init ===" -ForegroundColor Cyan
Write-Host "Repo: $Root"
Write-Host ""

Require-Command uv
Require-Command hf

Write-Host "=== 1/5 Base runtime (uv sync) ===" -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipVenvs) {
    Write-Host ""
    Write-Host "=== 2/5 External venvs (.venvs/*) ===" -ForegroundColor Cyan
    & "$PSScriptRoot\setup_external_venvs.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host ""
    Write-Host "=== 2/5 External venvs: skipped (-SkipVenvs) ===" -ForegroundColor Yellow
}

$isWindows = ($env:OS -match "Windows") -or ($IsWindows -eq $true)
if ($isWindows -and -not $SkipFfmpeg) {
    Write-Host ""
    Write-Host "=== 3/5 ffmpeg (tools/ffmpeg/win64) ===" -ForegroundColor Cyan
    uv run python tools/download_ffmpeg.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ffmpeg download failed (network?). Re-run: download_ffmpeg.bat" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} elseif ($SkipFfmpeg) {
    Write-Host ""
    Write-Host "=== 3/5 ffmpeg: skipped (-SkipFfmpeg) ===" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "=== 3/5 ffmpeg: skipped (not Windows) ===" -ForegroundColor Yellow
}

$envTemplate = Join-Path $Root ".env.template"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile) -and (Test-Path $envTemplate)) {
    Copy-Item $envTemplate $envFile
    Write-Host ""
    Write-Host "Created .env from .env.template" -ForegroundColor Green
}

if (-not $SkipSage) {
    Write-Host ""
    Write-Host "=== 4/5 Optional: SageAttention (.venvs/media) ===" -ForegroundColor Cyan
    $env:UV_PROJECT_ENVIRONMENT = ".venvs/media"
    try {
        & "$PSScriptRoot\setup_sage_windows.ps1"
    } catch {
        Write-Host "  SageAttention setup failed (optional): $_" -ForegroundColor Yellow
        Write-Host "  Re-run later: `$env:UV_PROJECT_ENVIRONMENT='.venvs/media'; powershell -File tools\setup_sage_windows.ps1"
    }
} else {
    Write-Host ""
    Write-Host "=== 4/5 SageAttention: skipped (-SkipSage) ===" -ForegroundColor Yellow
}

if ($UploadTestData) {
    Write-Host ""
    Write-Host "=== Upload test_data/ -> skreling-eng/anthill ===" -ForegroundColor Cyan
    $upArgs = @("tools/upload_to_hf.py", "--bundle", "test-data")
    if ($DryRun) { $upArgs += "--dry-run" }
    uv run python @upArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "=== Init complete ===" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Ensure .env has AH_EXTERNAL_VENV_* lines (printed by setup_external_venvs.ps1)
  2. Download models + test_data:  download_all_models.bat
     (or: download_all_models.bat -Profile minimal -UpstreamFallback)
  3. Maintainer publish test_data:  init.bat -UploadTestData  (Write HF token)
  4. Run:  uv run python run_ah.py examples\example_simple_image_generation.ah

ffmpeg status:  uv run python tools/download_ffmpeg.py --status
Bundle status:  uv run python tools/download_models.py --status
"@
