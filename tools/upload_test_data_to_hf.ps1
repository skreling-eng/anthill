# Upload local test_data/ to skreling-eng/anthill (incremental).
# This is PowerShell — do NOT run:  python tools\upload_test_data_to_hf.ps1
# Run from repo root:
#   upload_test_data.bat
#   upload_test_data.bat -DryRun
#   uv run python tools\upload_test_data_to_hf.py --dry-run
#   init.bat -UploadTestData
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$pyArgs = @("tools/upload_to_hf.py", "--bundle", "test-data")
if ($DryRun) { $pyArgs += "--dry-run" }
if ($Force) { $pyArgs += "--force" }
if ($RepoId) { $pyArgs += "--repo-id", $RepoId }

uv run python @pyArgs
