@echo off
REM Download ffmpeg + ffprobe into tools/ffmpeg/<platform>/ (not committed).
REM   download_ffmpeg.bat
REM   download_ffmpeg.bat -Force
setlocal
cd /d "%~dp0"
uv run python tools/download_ffmpeg.py %*
exit /b %ERRORLEVEL%
