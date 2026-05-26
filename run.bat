@echo off
REM Orchestrator: base deps only. GPU externals use .venvs\media / .venvs\music (see tools\setup_external_venvs.ps1).
if not exist ".venvs\media\Scripts\python.exe" (
    echo [anthill] Run once: powershell -ExecutionPolicy Bypass -File tools\setup_external_venvs.ps1
    exit /b 1
)
uv run python run_ah.py example.ah
