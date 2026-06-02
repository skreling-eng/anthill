"""$detach_audio — extract audio from videos[] into sounds[]."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$detach_audio needs videos[] in the input bundle.

Example:
  @track: $file('clip.mp4') -> $detach_audio -> $save('clip.wav')

Optional: format=wav (default) | aac | mp3 | copy

Requires ffmpeg: uv run python tools/download_ffmpeg.py (tools/ffmpeg/) or PATH.
Set AH_EMULATE_DETACH_AUDIO=1 for a stub.
"""

_FORMAT_EXT = {
    "wav": ".wav",
    "wave": ".wav",
    "aac": ".m4a",
    "mp3": ".mp3",
    "copy": ".m4a",
}


def _resolve_paths(ctx: ExternalContext, links: list[str]) -> list[Path]:
    paths: list[Path] = []
    for link in links:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _output_name(index: int, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_detach_{index}{ext}"


def _format_arg(inp: ExternalInput) -> str:
    raw = inp.args.get("format", "wav").strip().lower() or "wav"
    if raw not in _FORMAT_EXT:
        raise ValueError(
            f"$detach_audio: format= must be wav, aac, mp3, or copy; got {raw!r}"
        )
    return raw


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    videos = _resolve_paths(ctx, inp.bundle.videos)
    if not videos:
        raise RuntimeError(_HELP.strip())

    fmt = _format_arg(inp)
    ext = _FORMAT_EXT[fmt]

    out.videos.clear()

    if os.environ.get("AH_EMULATE_DETACH_AUDIO", "").lower() in ("1", "true", "yes"):
        out.sounds.clear()
        for index, path in enumerate(videos):
            content = f"[emulated $detach_audio format={fmt}] {path.name}\n"
            out.sounds.append(ctx.new_link("sounds", ext, content))
        return out

    from externals.video_audio.ffmpeg_io import detach_audio, require_ffmpeg

    require_ffmpeg()

    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    out.sounds.clear()

    for index, video_path in enumerate(videos):
        dest = sounds_dir / _output_name(index, ext)
        detach_audio(video_path, dest, fmt=fmt)
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.sounds.append(link)

    return out
