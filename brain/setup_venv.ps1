# Create an isolated uv environment for brain (no anthill / ahlib install).
# Run from repo root or brain/:  powershell -File brain\setup_venv.ps1
$ErrorActionPreference = "Stop"
$BrainDir = $PSScriptRoot
Set-Location $BrainDir

Write-Host "=== brain/.venv (isolated from main anthill env) ==="

if (-not (Test-Path ".venv")) {
    uv venv .venv
}

$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv sync

$py = Join-Path $BrainDir ".venv\Scripts\python.exe"
Write-Host ""
Write-Host "Verifying imports..."
& $py -c "import webview; import llama_cpp; print('  ok: pywebview + llama-cpp-python')"
Write-Host ""
Write-Host "GPU inference (recommended — default wheel is CPU-only):"
Write-Host "  powershell -File brain\setup_gpu.ps1"
Write-Host ""
Write-Host "Launch:"
Write-Host "  cd brain && uv run brain"
Write-Host "  cd brain && uv run python run_brain.py"
Write-Host "  powershell -File brain\run.ps1"
