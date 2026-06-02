"""Load and save audio for music separation."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(src_sr, dst_sr)
        up, down = dst_sr // g, src_sr // g
        out = resample_poly(audio, up, down, axis=1).astype(np.float32, copy=False)
        return np.ascontiguousarray(out)
    except ImportError:
        out_len = math.ceil(audio.shape[1] * dst_sr / src_sr)
        src_t = np.arange(audio.shape[1], dtype=np.float64) / src_sr
        dst_t = np.arange(out_len, dtype=np.float64) / dst_sr
        out = np.empty((audio.shape[0], out_len), dtype=np.float32)
        for ch in range(audio.shape[0]):
            out[ch] = np.interp(dst_t, src_t, audio[ch]).astype(np.float32)
        return out


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


def _load_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    from externals.video_audio.ffmpeg_paths import get_ffmpeg_exe, require_ffmpeg

    try:
        require_ffmpeg()
        ffmpeg = get_ffmpeg_exe()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "$music_separation needs ffmpeg to decode non-WAV audio.\n"
            "  uv run python tools/download_ffmpeg.py"
        ) from exc
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError(f"ffmpeg produced no audio for {path}")
    if raw.size % 2:
        raw = raw[:-1]
    audio = raw.reshape(-1, 2).T
    return audio, SAMPLE_RATE


def load_stereo(path: Path, *, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Return float32 stereo ``(2, samples)`` at ``target_sr`` and native sr."""
    try:
        audio, native_sr = _load_with_soundfile(path)
    except Exception:
        audio, native_sr = _load_with_ffmpeg(path)

    if audio.shape[0] == 1:
        audio = np.tile(audio, (2, 1))
    elif audio.shape[0] > 2:
        audio = audio[:2]

    if native_sr != target_sr:
        audio = _resample(audio, native_sr, target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32), native_sr


def resample_to_native(audio: np.ndarray, native_sr: int) -> np.ndarray:
    if native_sr == SAMPLE_RATE:
        return audio
    return _resample(audio, SAMPLE_RATE, native_sr)


def write_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {audio.shape}")
    pcm = np.clip(audio.T, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
