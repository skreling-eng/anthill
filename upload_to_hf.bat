@echo off
REM Upload models/ and test_data/ to skreling-eng/anthill (Python script).
REM   upload_to_hf.bat --token hf_...
REM   upload_to_hf.bat --bundle test-data --dry-run
setlocal
cd /d "%~dp0"
uv run python tools\upload_to_hf.py %*
exit /b %ERRORLEVEL%
