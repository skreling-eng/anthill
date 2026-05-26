# Windows: install triton + SageAttention 2.1.1 wheels for torch 2.7.1+cu128 (RTX 40xx).
# Run from repo root after uv sync --extra media. Close other Python processes first.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "No .venv. Run: uv sync --extra media"
}

$tag = & $Py -c "import sys; print('cp' + str(sys.version_info.major) + str(sys.version_info.minor))"
if ($tag -ne "cp312") {
    Write-Warning "Wheels below are built for cp312; you have $tag. Pick matching wheels from woct0rdho releases."
}

$TritonUrl = "https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp312-cp312-win_amd64.whl"
$SageUrl = "https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1%2Bcu128torch2.7.1-cp312-cp312-win_amd64.whl"

Write-Host "Installing triton (Windows)..."
uv pip install $TritonUrl

Write-Host "Installing sageattention 2.1.1+cu128torch2.7.1..."
uv pip install --force-reinstall $SageUrl

Write-Host ""
Write-Host "Verify:"
$verify = Join-Path $PSScriptRoot "verify_sage.py"
& $Py $verify
