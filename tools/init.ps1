# Fresh Anthill checkout: install Python deps, external venvs, and model weights.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\init.ps1
#   powershell -ExecutionPolicy Bypass -File tools\init.ps1 -Profile minimal -SkipSage
param(
    [ValidateSet("minimal", "standard", "full")]
    [string]$Profile = "standard",
    [switch]$SkipVenvs,
    [switch]$SkipModels,
    [switch]$SkipSage,
    [switch]$UpstreamFallback,
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

Write-Host "=== 1/4 Base runtime (uv sync) ===" -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipVenvs) {
    Write-Host ""
    Write-Host "=== 2/4 External venvs (.venvs/*) ===" -ForegroundColor Cyan
    & "$PSScriptRoot\setup_external_venvs.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host ""
    Write-Host "=== 2/4 External venvs: skipped (-SkipVenvs) ===" -ForegroundColor Yellow
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
    Write-Host "=== 3/4 Optional: SageAttention (.venvs/media) ===" -ForegroundColor Cyan
    $env:UV_PROJECT_ENVIRONMENT = ".venvs/media"
    try {
        & "$PSScriptRoot\setup_sage_windows.ps1"
    } catch {
        Write-Host "  SageAttention setup failed (optional): $_" -ForegroundColor Yellow
        Write-Host "  Re-run later: `$env:UV_PROJECT_ENVIRONMENT='.venvs/media'; powershell -File tools\setup_sage_windows.ps1"
    }
} else {
    Write-Host ""
    Write-Host "=== 3/4 SageAttention: skipped (-SkipSage) ===" -ForegroundColor Yellow
}

if (-not $SkipModels) {
    Write-Host ""
    Write-Host "=== 4/4 Model weights (models/) ===" -ForegroundColor Cyan
    $dlArgs = @("tools/download_models.py", "--profile", $Profile)
    if ($DryRun) { $dlArgs += "--dry-run" }
    if ($UpstreamFallback) { $dlArgs += "--upstream-fallback" }
    uv run python @dlArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host ""
    Write-Host "=== 4/4 Models: skipped (-SkipModels) ===" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Init complete ===" -ForegroundColor Green
Write-Host @"

Next steps:
  1. Ensure .env has AH_EXTERNAL_VENV_* lines (printed by setup_external_venvs.ps1)
  2. If models are still missing: wait for anthill upload, or init.ps1 -UpstreamFallback
  3. Run:  uv run python run_ah.py examples\example_simple_image_generation.ah

Model status:  uv run python tools/download_models.py --status
"@
