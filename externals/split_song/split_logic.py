"""Split song audio at low vocal-activity points (uses vocal stem envelope)."""

from __future__ import annotations

import numpy as np


def rms_envelope(mono: np.ndarray, sr: int, *, frame_sec: float = 0.05) -> tuple[np.ndarray, float]:
    """Per-frame RMS on mono audio; returns (envelope, frame_sec)."""
    hop = max(1, int(frame_sec * sr))
    n_frames = max(1, int(np.ceil(len(mono) / hop)))
    envelope = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(mono))
        chunk = mono[start:end]
        envelope[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
    return envelope, frame_sec


def _time_to_frame(t: float, frame_sec: float, n_frames: int) -> int:
    idx = int(t / frame_sec)
    return max(0, min(idx, n_frames - 1))


def find_split_segments(
    duration_sec: float,
    activity: np.ndarray,
    frame_sec: float,
    period_sec: float,
) -> list[tuple[float, float]]:
    """
    Partition [0, duration] into segments with length in [period/2, period]
    (last segment may be shorter only when the remainder fits in one chunk).

    Each internal cut is placed at the lowest vocal activity in the allowed window.
    """
    if duration_sec <= 0:
        return []
    period = float(period_sec)
    if period <= 0:
        raise ValueError(f"period must be positive, got {period_sec!r}")
    min_period = period * 0.5
    eps = 1e-6

    if duration_sec <= period:
        return [(0.0, duration_sec)]

    segments: list[tuple[float, float]] = []
    t = 0.0
    n_frames = len(activity)

    while t < duration_sec - eps:
        remaining = duration_sec - t
        if remaining <= period + eps:
            segments.append((t, duration_sec))
            break

        lo = t + min_period
        hi = t + period
        lo_i = _time_to_frame(lo, frame_sec, n_frames)
        hi_i = _time_to_frame(hi, frame_sec, n_frames)
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        window = activity[lo_i : hi_i + 1]
        if window.size == 0:
            split_t = hi
        else:
            split_t = (lo_i + int(np.argmin(window)) + 0.5) * frame_sec
            split_t = min(max(split_t, lo), hi)

        if duration_sec - split_t < min_period - eps:
            segments.append((t, duration_sec))
            break

        segments.append((t, split_t))
        t = split_t

    if not segments:
        return [(0.0, duration_sec)]
    return segments
