# Windows: install triton + SageAttention 2.1.1 wheels for torch 2.7.1+cu128 (RTX 40xx).
# Run from repo root. Close other Python processes using the target venv first.
param(
    [string]$VenvPath = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $VenvPath) {
    if ($env:UV_PROJECT_ENVIRONMENT) {
        $VenvPath = $env:UV_PROJECT_ENVIRONMENT
    } elseif (Test-Path (Join-Path $Root ".venvs\comfy-wan")) {
        $VenvPath = ".venvs/comfy-wan"
    } else {
        $VenvPath = ".venv"
    }
}

$Py = Join-Path $Root (Join-Path $VenvPath "Scripts\python.exe")
if (-not (Test-Path $Py)) {
    throw "No python at $VenvPath. Run: tools\setup_external_venvs.ps1 (or uv venv $VenvPath)"
}

Write-Host "Target venv: $VenvPath"

$tag = & $Py -c "import sys; print('cp' + str(sys.version_info.major) + str(sys.version_info.minor))"
if ($tag -ne "cp312") {
    Write-Warning "Wheels below are built for cp312; you have $tag. Pick matching wheels from woct0rdho releases."
}

$TritonUrl = "https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp312-cp312-win_amd64.whl"
$SageUrl = "https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1%2Bcu128torch2.7.1-cp312-cp312-win_amd64.whl"

Write-Host "Installing triton (Windows)..."
uv pip install $TritonUrl --python $Py

Write-Host "Installing sageattention 2.1.1+cu128torch2.7.1..."
uv pip install --force-reinstall $SageUrl --python $Py

Write-Host ""
Write-Host "Verify:"
$verify = Join-Path $PSScriptRoot "verify_sage.py"
& $Py $verify
