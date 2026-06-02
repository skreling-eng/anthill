# Download ffmpeg + ffprobe into tools/ffmpeg/<platform>/
param(
    [switch]$Force,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$args = @("tools/download_ffmpeg.py")
if ($Force) { $args += "--force" }
if ($Status) { $args += "--status" }
uv run python @args
exit $LASTEXITCODE
