"""$join_stems — mix incoming sounds[] (stems) into one WAV."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from externals.api import ExternalContext, ExternalInput
from externals.music_separation.audio_io import (
    SAMPLE_RATE,
    load_stereo,
    write_wav_bytes,
)
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$join_stems needs sounds[] (stem WAV/MP3). Two or more are summed; one passes through.

Example:
  @mix: ( $select(sounds=[0]), stems% -> $select(sounds=[1]) ) -> $join_stems

AH_EMULATE_JOIN_STEMS=1 for stub output without audio deps.
"""


def _truthy(raw: str, *, default: bool = True) -> bool:
    text = raw.strip().lower()
    if not text:
        return default
    return text in ("1", "true", "yes", "on")


def _sound_path(ctx: ExternalContext, link: str) -> Path:
    path = Path(link)
    if not path.is_absolute():
        path = (ctx.base_dir / link).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"$join_stems: sound not found: {path}")
    return path


def _mix_tracks(tracks: list[np.ndarray], *, normalize: bool) -> np.ndarray:
    if not tracks:
        raise ValueError("$join_stems: no audio to mix")
    max_len = max(track.shape[1] for track in tracks)
    mix = np.zeros((2, max_len), dtype=np.float32)
    for track in tracks:
        if track.ndim != 2 or track.shape[0] != 2:
            raise ValueError(
                f"$join_stems: expected stereo (2, samples), got {track.shape}"
            )
        mix[:, : track.shape[1]] += track
    if normalize:
        peak = float(np.max(np.abs(mix)))
        if peak > 1.0:
            mix *= 0.99 / peak
    return mix


def _join_sounds(
    ctx: ExternalContext,
    sounds: list[str],
    *,
    sample_rate: int,
    normalize: bool,
) -> list[str]:
    tracks: list[np.ndarray] = []
    for link in sounds:
        audio, _ = load_stereo(_sound_path(ctx, link), target_sr=sample_rate)
        tracks.append(audio)

    mixed = tracks[0] if len(tracks) == 1 else _mix_tracks(tracks, normalize=normalize)
    wav = write_wav_bytes(mixed, sample_rate)
    return [ctx.new_link("sounds", ".wav", wav)]


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    sounds: list[str],
) -> None:
    content = "[emulated $join_stems]\n" + "\n".join(f"- {s}" for s in sounds) + "\n"
    out.sounds.append(ctx.new_link("sounds", ".wav", content))


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    sounds = list(inp.bundle.sounds)
    if not sounds:
        raise RuntimeError(_HELP.strip())

    normalize = _truthy(inp.args.get("normalize", ""), default=True)
    sample_rate = int(inp.args.get("sample_rate", str(SAMPLE_RATE)) or SAMPLE_RATE)

    out = inp.bundle.copy()
    out.sounds = []

    if os.environ.get("AH_EMULATE_JOIN_STEMS", "").lower() in ("1", "true", "yes"):
        _emulate(ctx, out, sounds)
        return out

    out.sounds = _join_sounds(
        ctx,
        sounds,
        sample_rate=sample_rate,
        normalize=normalize,
    )
    return out
