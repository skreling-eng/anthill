# Upload local models/ to skreling-eng/anthill (incremental; skips .cache etc.).
# Run from repo root:
#   .\tools\upload_models_to_hf.ps1 -DryRun
#   .\tools\upload_models_to_hf.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$pyArgs = @("tools/upload_models_to_hf.py")
if ($DryRun) { $pyArgs += "--dry-run" }
if ($Force) { $pyArgs += "--force" }
if ($RepoId) { $pyArgs += "--repo-id", $RepoId }

uv run python @pyArgs
