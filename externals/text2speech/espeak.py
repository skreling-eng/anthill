"""Configure espeak-ng for Kokoro / phonemizer on Windows."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DLL = _REPO_ROOT / "tools" / "libespeak-ng.dll"


def _candidates(explicit: str | None) -> list[Path]:
    paths: list[Path] = []
    if explicit and explicit.strip():
        paths.append(Path(explicit.strip()))
    env = os.environ.get("AH_ESPEAK_LIBRARY", "").strip()
    if env:
        paths.append(Path(env))
    paths.append(_DEFAULT_DLL)
    paths.extend(
        [
            Path(r"C:\Program Files\eSpeak NG\libespeak-ng.dll"),
            Path(r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"),
            Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so"),
            Path("/usr/lib/libespeak-ng.so"),
        ]
    )
    return paths


def configure_espeak(explicit: str | None = None) -> str | None:
    """Set PHONEMIZER_ESPEAK_LIBRARY when a library file exists."""
    for path in _candidates(explicit):
        if path.is_file():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(path.resolve())
            return str(path.resolve())
    return None
