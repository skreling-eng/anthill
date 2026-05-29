@echo off
REM Fresh checkout: install deps, external venvs, and models from skreling-eng/anthill.
REM   init.bat
REM   init.bat -Profile minimal
REM   init.bat -SkipSage -DryRun
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\init.ps1" %*
exit /b %ERRORLEVEL%
