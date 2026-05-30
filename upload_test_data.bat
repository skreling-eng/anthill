@echo off
REM Upload test_data/ only (alias for upload_to_hf.bat --bundle test-data).
REM   upload_test_data.bat --token hf_...
setlocal
cd /d "%~dp0"
uv run python tools\upload_to_hf.py --bundle test-data %*
exit /b %ERRORLEVEL%
