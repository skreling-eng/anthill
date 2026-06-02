"""$attach_audio — mux sounds[] onto videos[] → videos[]."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$attach_audio needs videos[] and sounds[] in the input bundle.

Example:
  @muxed: (@clip, @music) -> $attach_audio -> $save('final.mp4')

Pairing: same count (zip), or one video + many sounds, or many videos + one sound.

Optional: shortest=1 (default) | shortest=0, audio_codec=aac

Requires ffmpeg: uv run python tools/download_ffmpeg.py (tools/ffmpeg/) or PATH.
Set AH_EMULATE_ATTACH_AUDIO=1 for a stub.
"""


def _resolve_paths(ctx: ExternalContext, links: list[str]) -> list[Path]:
    paths: list[Path] = []
    for link in links:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_attach_{index}.mp4"


def _bool_arg(inp: ExternalInput, key: str, default: bool) -> bool:
    raw = inp.args.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    videos = _resolve_paths(ctx, inp.bundle.videos)
    sounds = _resolve_paths(ctx, inp.bundle.sounds)
    if not videos or not sounds:
        raise RuntimeError(_HELP.strip())

    shortest = _bool_arg(inp, "shortest", True)
    audio_codec = inp.args.get("audio_codec", "aac").strip() or "aac"

    out.sounds.clear()
    out.images.clear()
    out.texts.clear()
    out.prompts.clear()
    out.files.clear()
    out.embeddings.clear()
    out.labels.clear()
    out.changes.clear()

    if os.environ.get("AH_EMULATE_ATTACH_AUDIO", "").lower() in ("1", "true", "yes"):
        from externals.video_audio.ffmpeg_io import pair_videos_and_sounds

        try:
            pairs = pair_videos_and_sounds(videos, sounds)
        except ValueError as exc:
            raise RuntimeError(f"$attach_audio: {exc}") from exc
        out.videos.clear()
        for video_path, sound_path in pairs:
            content = (
                f"[emulated $attach_audio shortest={shortest}]\n"
                f"video: {video_path.name}\n"
                f"sound: {sound_path.name}\n"
            )
            out.videos.append(ctx.new_link("videos", ".mp4", content))
        return out

    from externals.video_audio.ffmpeg_io import (
        attach_audio,
        pair_videos_and_sounds,
        require_ffmpeg,
    )

    require_ffmpeg()

    try:
        pairs = pair_videos_and_sounds(videos, sounds)
    except ValueError as exc:
        raise RuntimeError(f"$attach_audio: {exc}") from exc

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    out.videos.clear()

    for index, (video_path, sound_path) in enumerate(pairs):
        dest = videos_dir / _output_name(index)
        attach_audio(
            video_path,
            sound_path,
            dest,
            shortest=shortest,
            audio_codec=audio_codec,
        )
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.videos.append(link)

    return out
