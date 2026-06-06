@echo off
REM Run an .ah script from any folder; sessions go in the launch directory.
REM Add this file's directory to PATH (or run tools\install_ah.ps1).
setlocal EnableExtensions
if "%~1"=="" (
  echo Usage: ah path\to\script.ah
  exit /b 1
)
set "AH_LAUNCH_DIR=%CD%"
set "AH_HOME=%~dp0"
if "%AH_HOME:~-1%"=="\" set "AH_HOME=%AH_HOME:~0,-1%"
cd /d "%AH_HOME%"
uv run python run_ah.py %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
