"""Locate or optionally download ace-synth (no HTTP server required)."""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from externals.music.model_paths import _PROJECT_ROOT, ace_step_dir

_TOOLS_DIR = _PROJECT_ROOT / "tools" / "acestep"
_RELEASE = "v0.0.5"
_BASE_URL = f"https://github.com/audiohacking/acestep.cpp/releases/download/{_RELEASE}"


def _platform_asset() -> tuple[str, str, str] | None:
    """Return (archive_name, kind, binary_basename) for this OS, or None."""
    if sys.platform == "win32":
        return "acestep-windows-x64.zip", "zip", "ace-synth.exe"
    if sys.platform == "darwin":
        return "acestep-macos-arm64-metal.tar.gz", "tar", "ace-synth"
    if sys.platform.startswith("linux"):
        return "acestep-linux-x64.tar.gz", "tar", "ace-synth"
    return None


def synth_bin_candidates() -> list[Path]:
    names = ("ace-synth.exe", "ace-synth")
    candidates: list[Path] = []

    raw = os.environ.get("ACESTEP_SYNTH_BIN", "").strip()
    if raw:
        candidates.append(Path(raw))

    for name in names:
        candidates.append(_TOOLS_DIR / name)
        candidates.append(ace_step_dir() / name)

    for rel in (
        "acestep.cpp/build/Release/ace-synth.exe",
        "acestep.cpp/build/ace-synth.exe",
        "acestep.cpp/build/ace-synth",
    ):
        candidates.append(_PROJECT_ROOT / rel)

    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_synth_bin() -> Path | None:
    for candidate in synth_bin_candidates():
        if candidate.is_file():
            return candidate
    return None


def _install_binary(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    if os.name != "nt" and not dest.suffix:
        dest.chmod(dest.stat().st_mode | 0o111)
    return dest


def _extract_archive(archive: Path, kind: str) -> None:
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(_TOOLS_DIR)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(_TOOLS_DIR)


def _locate_extracted_binary(basename: str) -> Path | None:
    direct = _TOOLS_DIR / basename
    if direct.is_file():
        return direct
    hits = sorted(_TOOLS_DIR.glob(f"**/{basename}"))
    return hits[0] if hits else None


def download_synth_bin() -> Path | None:
    """Download ace-synth for the current platform into tools/acestep/."""
    asset = _platform_asset()
    if asset is None:
        print(f"$music no ace-synth release for platform {sys.platform!r}.")
        return None

    archive_name, kind, binary_name = asset
    url = f"{_BASE_URL}/{archive_name}"
    archive_path = _TOOLS_DIR / archive_name
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"$music downloading ace-synth from {url}")
    urlretrieve(url, archive_path)
    _extract_archive(archive_path, kind)
    archive_path.unlink(missing_ok=True)

    found = _locate_extracted_binary(binary_name)
    if found is None:
        return None
    dest = _TOOLS_DIR / binary_name
    installed = _install_binary(found, dest)
    print(f"$music installed {installed}")
    return installed


def ensure_synth_bin() -> Path | None:
    """Return ace-synth path; optionally download when ACESTEP_DOWNLOAD_BIN=1."""
    found = find_synth_bin()
    if found:
        return found
    if os.environ.get("ACESTEP_DOWNLOAD_BIN", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    return download_synth_bin()


def synth_stack_ready() -> bool:
    from externals.music.model_paths import gguf_stack_ready

    return gguf_stack_ready() and find_synth_bin() is not None
