"""Helpers for &action handlers — write files and return session-relative links."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pcm_to_float(data: Any) -> np.ndarray:
    """Normalize integer or float PCM to float64 in [-1, 1]."""
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64)
    info = np.iinfo(arr.dtype)
    peak = float(max(abs(info.min), info.max))
    return arr.astype(np.float64) / peak


def float_to_int16(data: Any) -> np.ndarray:
    """Convert normalized float PCM to int16."""
    arr = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16)


def apply_db_gain(data: Any, db: float) -> np.ndarray:
    """Apply gain in dB; returns normalized float64 PCM."""
    return pcm_to_float(data) * (10 ** (db / 20.0))


def _rel_link(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def save_bytes(
    base_dir: str | Path,
    op_dir: str | Path,
    array: str,
    filename: str,
    data: bytes,
) -> str:
    """Write bytes under op_dir/<array>/<filename>; return link for bundle[]."""
    root = Path(base_dir)
    dest = Path(op_dir) / array / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return _rel_link(root, dest)


def save_wav(
    base_dir: str | Path,
    op_dir: str | Path,
    filename: str,
    sample_rate: int,
    data: Any,
) -> str:
    """Write WAV under op_dir/sounds/; return link for sounds[]."""
    dest = Path(op_dir) / "sounds" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from scipy.io import wavfile

        wavfile.write(dest, sample_rate, data)
    except ImportError:
        import wave

        import numpy as np

        arr = np.asarray(data)
        if arr.dtype not in (np.int16, np.int32, np.uint8):
            arr = np.int16(arr)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1 if arr.ndim == 1 else arr.shape[1])
            wf.setsampwidth(arr.dtype.itemsize)
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())
    return _rel_link(Path(base_dir), dest)


def save_image(
    base_dir: str | Path,
    op_dir: str | Path,
    filename: str,
    image: Any,
) -> str:
    """Save PIL Image under op_dir/images/; return link for images[]."""
    dest = Path(op_dir) / "images" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return _rel_link(Path(base_dir), dest)


def save_text(
    base_dir: str | Path,
    op_dir: str | Path,
    array: str,
    filename: str,
    text: str,
) -> str:
    """Write UTF-8 text under op_dir/<array>/; return link for prompts[] / texts[]."""
    dest = Path(op_dir) / array / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return _rel_link(Path(base_dir), dest)
