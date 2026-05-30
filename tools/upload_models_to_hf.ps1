# Upload models/ to skreling-eng/anthill. Prefer: upload_to_hf.bat
#   uv run python tools\upload_to_hf.py --bundle models --token hf_...
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$pyArgs = @("tools/upload_to_hf.py", "--bundle", "models")
if ($DryRun) { $pyArgs += "--dry-run" }
if ($Force) { $pyArgs += "--force" }
if ($RepoId) { $pyArgs += "--repo-id", $RepoId }

uv run python @pyArgs
