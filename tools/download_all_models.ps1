# Download Anthill models/ + test_data/ from skreling-eng/anthill.
# Re-runs are incremental: only missing groups/files are fetched.
# Run from repo root:
#   download_all_models.bat
#   download_all_models.bat -Profile minimal
#   download_all_models.bat -UpstreamFallback
param(
    [ValidateSet("minimal", "standard", "full")]
    [string]$Profile = "standard",
    [switch]$SkipTestData,
    [switch]$UpstreamFallback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name. Install it and re-run download_all_models."
    }
}

Write-Host "=== Anthill model download ===" -ForegroundColor Cyan
Write-Host "Repo:    $Root"
Write-Host "Profile: $Profile"
Write-Host "Source:  https://huggingface.co/skreling-eng/anthill"
Write-Host ""

Require-Command uv
Require-Command hf

$dlArgs = @("tools/download_models.py", "--profile", $Profile)
if ($DryRun) { $dlArgs += "--dry-run" }
if ($UpstreamFallback) { $dlArgs += "--upstream-fallback" }
if ($SkipTestData) { $dlArgs += "--skip-test-data" }

uv run python @dlArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Download complete ===" -ForegroundColor Green
Write-Host "Status:  uv run python tools/download_models.py --status"
