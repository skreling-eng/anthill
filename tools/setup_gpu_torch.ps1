# Reinstall CUDA PyTorch (fixes torch_cuda.dll WinError 126 / os error 32 file lock).
# Close Cursor terminals and any "uv run python" using G:\_cur\_anthill\.venv first.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Site = Join-Path $Root ".venv\Lib\site-packages"

Write-Host "Moving aside old torch (avoids DLL lock on delete)..."
$torchDir = Join-Path $Site "torch"
if (Test-Path $torchDir) {
    $bak = Join-Path $Site ("_torch_removed_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Rename-Item $torchDir $bak -Force
    Write-Host "  -> $bak"
}
Get-ChildItem $Site -Filter "torch-*.dist-info" -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
        $bak = Join-Path $Site ("_" + $_.Name + "_" + (Get-Date -Format "HHmmss"))
        Rename-Item $_.FullName $bak -Force -ErrorAction SilentlyContinue
    }

$env:UV_LINK_MODE = "copy"
Write-Host "uv sync --extra image2video ..."
uv sync --extra image2video

Write-Host ""
Write-Host "Verify:"
& $Py -c @"
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"@
