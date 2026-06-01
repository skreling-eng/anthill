@echo off
REM Fresh checkout: deps + venvs (models: download_all_models.bat).
REM   init.bat
REM   init.bat -SkipSage
REM   init.bat -UploadTestData              maintainer: push test_data/ to HF
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\init.ps1" %*
exit /b %ERRORLEVEL%
