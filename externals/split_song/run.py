"""$split_song — split songs on low vocal activity (stem-guided intervals)."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.music_separation.model_paths import configure_models_environment
from externals.music_separation.models import (
    ROFORMER_MODELS_DIR,
    VOCAL_STEM_MODEL,
    resolve_variant,
)
from externals.music_separation.roformer import separate_file as roformer_separate
from externals.split_song.audio_utils import (
    load_native,
    slice_segment,
    to_mono,
    write_segment,
)
from externals.split_song.split_logic import find_split_segments, rms_envelope
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$split_song needs sounds[] in the input bundle.

Separates vocals, finds low-activity cut points on the vocal stem, then splits the
original audio into WAV segments (each at most period= seconds, at least period/2).

Example:
  @track: $file('song.mp3')
  @parts: @track -> $split_song(period=30)

Optional:
  period=30       — max segment length in seconds (min = period / 2)
  model=bs_roformer_viperx_1297 — vocal separation (same 2-stem as $music_separation)
  model_path=...  — optional RoFormer checkpoint override
  frame=0.05      — vocal activity frame size in seconds

AH_EMULATE_SPLIT_SONG=1 for stub output without models.
"""


def _output_name(source_stem: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", source_stem)
    return f"{ts}_{safe}_part{index:03d}.wav"


def _float_arg(args: dict[str, str], key: str, default: float, *, min_value: float) -> float:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"$split_song: invalid {key}={raw!r}") from exc
    if value < min_value:
        raise ValueError(f"$split_song: {key} must be >= {min_value}, got {value}")
    return value


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    sounds: list[str],
    *,
    period: float,
) -> None:
    for sound in sounds:
        for part in range(2):
            content = (
                f"[emulated $split_song period={period} part={part}]\n"
                f"from: {sound}\n"
            )
            out.sounds.append(ctx.new_link("sounds", ".wav", content))


def _extract_vocals(
    audio_path: Path,
    *,
    model_name: str,
    model_path: str,
    work_dir: Path,
) -> Path:
    configure_models_environment()
    ROFORMER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUDIO_SEPARATOR_MODEL_DIR", str(ROFORMER_MODELS_DIR))
    variant = resolve_variant(model_name, model_path=model_path)
    if variant.backend != "roformer" or "vocals" not in variant.stems:
        raise ValueError(
            f"$split_song: model={model_name!r} must provide a vocals stem "
            f"(try model={VOCAL_STEM_MODEL!r} or model=2stem)"
        )
    stems = roformer_separate(
        audio_path,
        model_filename=variant.model_filename,
        expected_stems=variant.stems,
        work_dir=work_dir,
        checkpoint=variant.checkpoint,
    )
    return stems["vocals"]


def _split_one(
    ctx: ExternalContext,
    sound_link: str,
    *,
    period: float,
    frame_sec: float,
    model_name: str,
    model_path: str,
    track_index: int,
) -> list[str]:
    audio_path = (ctx.base_dir / sound_link).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"$split_song: sound not found: {audio_path}")

    work_dir = ctx.op_dir / "work" / f"track_{track_index}"
    vocals_path = _extract_vocals(
        audio_path,
        model_name=model_name,
        model_path=model_path,
        work_dir=work_dir,
    )

    vocal_audio, vocal_sr = load_native(vocals_path)
    mono = to_mono(vocal_audio)
    activity, used_frame = rms_envelope(mono, vocal_sr, frame_sec=frame_sec)
    duration = mono.shape[0] / vocal_sr
    segments = find_split_segments(duration, activity, used_frame, period)

    original, native_sr = load_native(audio_path)
    sounds_dir = ctx.op_dir / "sounds"
    out_links: list[str] = []
    for part_index, (start, end) in enumerate(segments):
        chunk = slice_segment(original, native_sr, start, end)
        dest = sounds_dir / _output_name(audio_path.stem, part_index)
        write_segment(dest, chunk, native_sr)
        out_links.append(str(dest.relative_to(ctx.base_dir)).replace("\\", "/"))
    return out_links


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    sounds = list(inp.bundle.sounds)
    if not sounds:
        raise RuntimeError(_HELP.strip())

    period = _float_arg(inp.args, "period", 30.0, min_value=0.1)
    frame_sec = _float_arg(inp.args, "frame", 0.05, min_value=0.01)
    model_name = read_arg_list(inp, "model", VOCAL_STEM_MODEL)[0]
    model_path = inp.args.get("model_path", "").strip()

    if os.environ.get("AH_EMULATE_SPLIT_SONG", "").lower() in ("1", "true", "yes"):
        out.sounds = []
        _emulate(ctx, out, sounds, period=period)
        return out

    out.sounds = []
    for track_index, sound_link in enumerate(sounds):
        out.sounds.extend(
            _split_one(
                ctx,
                sound_link,
                period=period,
                frame_sec=frame_sec,
                model_name=model_name,
                model_path=model_path,
                track_index=track_index,
            )
        )
    return out
