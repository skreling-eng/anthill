# Install CUDA-enabled llama-cpp-python into brain/.venv (Windows).
# Requires NVIDIA GPU + driver. Run: powershell -File brain\setup_gpu.ps1
$ErrorActionPreference = "Stop"
$BrainDir = $PSScriptRoot
Set-Location $BrainDir

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Run brain\setup_venv.ps1 first." -ForegroundColor Yellow
    exit 1
}

Write-Host "=== brain/.venv — CUDA llama-cpp-python (cu124 wheel) ==="

$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv pip install llama-cpp-python `
    --force-reinstall `
    --no-cache-dir `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

$py = Join-Path $BrainDir ".venv\Scripts\python.exe"
Write-Host ""
Write-Host "Verifying GPU offload..."
& $py -c @"
import llama_cpp.llama_cpp as lc
ok = bool(lc.llama_supports_gpu_offload())
print('  gpu_offload:', ok)
if not ok:
    raise SystemExit('GPU offload still unavailable — see brain/README.md#gpu')
"@

Write-Host ""
Write-Host "Done. Launch with: powershell -File brain\run.ps1"
