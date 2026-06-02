"""Resolve ffmpeg/ffprobe: tools/ffmpeg/ (vendored) → PATH → error."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FFMPEG_ROOT = _REPO_ROOT / "tools" / "ffmpeg"


def repo_ffmpeg_root() -> Path:
    return _FFMPEG_ROOT


def platform_key() -> str:
    import platform

    if sys.platform == "win32":
        return "win64"
    if sys.platform == "darwin":
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos64"
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "linux-arm64"
    return "linux64"


def _exe_name(base: str) -> str:
    return f"{base}.exe" if sys.platform == "win32" else base


def vendored_bin_dir() -> Path | None:
    """Directory containing ffmpeg + ffprobe from tools/ffmpeg/<platform>/."""
    override = os.environ.get("AH_FFMPEG_DIR", "").strip()
    roots: list[Path] = []
    if override:
        roots.append(Path(override))
    roots.append(_FFMPEG_ROOT / platform_key())
    for root in roots:
        for candidate in (root, root / "bin"):
            ff = candidate / _exe_name("ffmpeg")
            fp = candidate / _exe_name("ffprobe")
            if ff.is_file() and fp.is_file():
                return candidate
    return None


def vendored_ready() -> bool:
    return vendored_bin_dir() is not None


def _prepend_path(directory: Path) -> None:
    extra = str(directory.resolve())
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if extra not in parts:
        os.environ["PATH"] = extra + (os.pathsep + path if path else "")


def get_ffmpeg_exe() -> str:
    explicit = os.environ.get("AH_FFMPEG", "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
    vendored = vendored_bin_dir()
    if vendored:
        return str((vendored / _exe_name("ffmpeg")).resolve())
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError(_missing_message())


def get_ffprobe_exe() -> str:
    explicit = os.environ.get("AH_FFPROBE", "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
    vendored = vendored_bin_dir()
    if vendored:
        return str((vendored / _exe_name("ffprobe")).resolve())
    found = shutil.which("ffprobe")
    if found:
        return found
    raise FileNotFoundError(_missing_message())


def _missing_message() -> str:
    key = platform_key()
    return (
        "ffmpeg/ffprobe not found.\n"
        f"  Install into repo: uv run python tools/download_ffmpeg.py\n"
        f"  Expected: tools/ffmpeg/{key}/ffmpeg"
        + (".exe on Windows)" if sys.platform == "win32" else ")")
        + "\n"
        "  Or install system ffmpeg and add to PATH, or set AH_FFMPEG_DIR=..."
    )


def require_ffmpeg() -> None:
    """Resolve binaries and prepend vendored bin dir to PATH (helps moviepy)."""
    vendored = vendored_bin_dir()
    if vendored:
        _prepend_path(vendored)
    get_ffmpeg_exe()
    get_ffprobe_exe()
