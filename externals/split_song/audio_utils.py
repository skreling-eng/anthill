"""Audio load/save helpers for $split_song."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from externals.music_separation.audio_io import write_wav_bytes


def load_native(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 ``(channels, samples)`` at native sample rate."""
    from externals.music_separation.audio_io import _load_with_ffmpeg, _load_with_soundfile

    try:
        data, sr = _load_with_soundfile(path)
    except Exception:
        data, sr = _load_with_ffmpeg(path)
    if data.ndim != 2:
        raise RuntimeError(f"unexpected audio shape {data.shape} for {path}")
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.shape[0] == 1:
        return audio[0]
    return audio.mean(axis=0)


def slice_segment(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * sr))
    end = min(audio.shape[1], int(end_sec * sr))
    if end <= start:
        end = min(start + 1, audio.shape[1])
    return audio[:, start:end]


def write_segment(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_wav_bytes(audio, sr))
