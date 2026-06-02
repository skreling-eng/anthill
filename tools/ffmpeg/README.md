# Vendored ffmpeg (not in git)

Anthill downloads **ffmpeg** and **ffprobe** here for `$detach_audio`, `$attach_audio`,
`$image_clip`, `$video_clip`, and related code.

## Install

From repo root:

```powershell
uv run python tools/download_ffmpeg.py
```

or:

```powershell
download_ffmpeg.bat
```

Layout after download:

```
tools/ffmpeg/
  win64/          # Windows: ffmpeg.exe, ffprobe.exe
  linux64/        # Linux x86_64
  linux-arm64/    # Linux ARM64
```

Binaries are **GPL builds** from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)
(~100 MB per platform). They are listed in `.gitignore` and are not committed.

## macOS

Auto-download is not provided. Use `brew install ffmpeg` or set:

```ini
AH_FFMPEG_DIR=/path/to/bin
```

## Overrides

| Variable | Purpose |
|----------|---------|
| `AH_FFMPEG_DIR` | Folder containing `ffmpeg` (+ `ffprobe`) |
| `AH_FFMPEG` | Full path to ffmpeg binary |
| `AH_FFPROBE` | Full path to ffprobe binary |
| `AH_FFMPEG_RELEASE` | BtbN release tag (default `latest`) |

Resolution order: explicit env → `tools/ffmpeg/<platform>/` → system `PATH`.
