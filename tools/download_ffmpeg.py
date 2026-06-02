#!/usr/bin/env python3
"""Download ffmpeg + ffprobe into tools/ffmpeg/<platform>/ (not committed to git).

  uv run python tools/download_ffmpeg.py
  uv run python tools/download_ffmpeg.py --status

Uses BtbN/FFmpeg-Builds releases (win64, linux64, linux-arm64).
macOS: not auto-downloaded — use brew install ffmpeg or set AH_FFMPEG_DIR.

Requires: network
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from externals.video_audio.ffmpeg_paths import (  # noqa: E402
    platform_key,
    repo_ffmpeg_root,
    vendored_ready,
    vendored_bin_dir,
)

# Pin a release tag for reproducibility; override with AH_FFMPEG_RELEASE=latest
DEFAULT_RELEASE = "latest"

ASSETS: dict[str, tuple[str, str]] = {
    "win64": ("zip", "ffmpeg-master-latest-win64-gpl.zip"),
    "linux64": ("tar.xz", "ffmpeg-master-latest-linux64-gpl.tar.xz"),
    "linux-arm64": ("tar.xz", "ffmpeg-master-latest-linuxarm64-gpl.tar.xz"),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _release() -> str:
    import os

    return os.environ.get("AH_FFMPEG_RELEASE", DEFAULT_RELEASE).strip() or DEFAULT_RELEASE


def _download_url(asset_name: str) -> str:
    tag = _release()
    return (
        f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{tag}/{asset_name}"
    )


def _fetch(url: str, dest: Path) -> None:
    _log(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _install_binaries(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = ("ffmpeg", "ffprobe", "ffmpeg.exe", "ffprobe.exe")
    found: dict[str, Path] = {}
    for path in src_dir.rglob("*"):
        if path.is_file() and path.name in names:
            key = path.name.replace(".exe", "")
            found[key] = path
    if "ffmpeg" not in found or "ffprobe" not in found:
        raise RuntimeError(f"ffmpeg/ffprobe not found under {src_dir}")
    for key, src in found.items():
        out = dest_dir / src.name
        shutil.copy2(src, out)
        if sys.platform != "win32":
            out.chmod(out.stat().st_mode | 0o111)


def _extract_archive(archive: Path, work: Path, fmt: str) -> Path:
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(work)
    elif fmt == "tar.xz":
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(work)
    else:
        raise ValueError(fmt)
    subdirs = [p for p in work.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return work


def _dest_ready(dest: Path, key: str) -> bool:
    if key == "win64":
        return (dest / "ffmpeg.exe").is_file() and (dest / "ffprobe.exe").is_file()
    return (dest / "ffmpeg").is_file() and (dest / "ffprobe").is_file()


def download_platform(key: str, *, force: bool) -> None:
    if key not in ASSETS:
        raise SystemExit(
            f"No bundled download for platform {key!r}. "
            "Install ffmpeg manually and set AH_FFMPEG_DIR."
        )
    dest = repo_ffmpeg_root() / key
    if not force and _dest_ready(dest, key):
        _log(f"tools/ffmpeg/{key}/ already ready")
        return

    fmt, asset = ASSETS[key]
    url = _download_url(asset)

    with tempfile.TemporaryDirectory(prefix="anthill-ffmpeg-") as tmp:
        work = Path(tmp)
        archive = work / asset
        _fetch(url, archive)
        extracted = _extract_archive(archive, work / "extract", fmt)
        if dest.exists() and force:
            shutil.rmtree(dest)
        _install_binaries(extracted, dest)

    _log(f"Installed ffmpeg + ffprobe -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=sorted(ASSETS.keys()),
        help="Force platform (default: auto-detect)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if ready")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print readiness and exit",
    )
    args = parser.parse_args()

    key = args.platform or platform_key()
    _log(f"Platform: {key}")

    if args.status:
        ready = vendored_ready()
        _log(f"  vendored: {'yes' if ready else 'no'} ({repo_ffmpeg_root() / key})")
        if vendored_bin_dir():
            _log(f"  bin dir: {vendored_bin_dir()}")
        return 0 if ready else 1

    if key.startswith("macos"):
        _log(
            "macOS auto-download is not supported (use: brew install ffmpeg)\n"
            "  or set AH_FFMPEG_DIR to a folder containing ffmpeg and ffprobe."
        )
        return 1

    download_platform(key, force=args.force)
    return 0 if vendored_ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
