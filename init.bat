@echo off
REM Fresh checkout: deps, venvs, models + test_data from skreling-eng/anthill.
REM   init.bat
REM   init.bat -Profile minimal -SkipSage
REM   init.bat -UploadTestData              maintainer: push test_data/ to HF
REM   init.bat -SkipModels -UploadTestData  upload only (no download)
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\init.ps1" %*
exit /b %ERRORLEVEL%
