@echo off
REM Download models/ + test_data/ from skreling-eng/anthill (incremental on re-run).
REM   download_all_models.bat
REM   download_all_models.bat -Profile minimal
REM   download_all_models.bat -UpstreamFallback
REM   download_all_models.bat -SkipTestData
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\download_all_models.ps1" %*
exit /b %ERRORLEVEL%
