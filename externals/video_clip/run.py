"""$video_clip — join input videos to match a sound track."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$video_clip needs videos[] and sounds[] in the input bundle.

Example:
  @clips: ... -> $image2video(...)
  @music: $file('track.wav')
  @out: (@clips, @music) -> $video_clip -> $save('output.mp4')

Optional args: fps=25, frames_per_chunk=81, delete_last_frames=0, repeat=120, flip=1

Reads frames directly from each MP4 (no PNG intermediate). Requires ffmpeg/ffprobe
and moviepy. Set AH_EMULATE_VIDEO_CLIP=1 for a stub.
"""


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_video_clip_{index}.mp4"


def _float_arg(inp: ExternalInput, key: str, default: float) -> float:
    raw = inp.args.get(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"$video_clip: invalid {key}={raw!r}") from exc
    if value <= 0:
        raise ValueError(f"$video_clip: {key} must be positive, got {value}")
    return value


def _int_arg(inp: ExternalInput, key: str, default: int, *, min_value: int = 0) -> int:
    raw = inp.args.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"$video_clip: invalid {key}={raw!r}") from exc
    if value < min_value:
        raise ValueError(f"$video_clip: {key} must be >= {min_value}, got {value}")
    return value


def _bool_arg(inp: ExternalInput, key: str, default: bool) -> bool:
    raw = inp.args.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    videos: list[str],
    sounds: list[str],
    fps: float,
    frames_per_chunk: int,
    delete_last_frames: int,
) -> None:
    content = (
        f"[emulated $video_clip fps={fps} "
        f"frames_per_chunk={frames_per_chunk} delete_last_frames={delete_last_frames}]\n"
        f"videos: {videos}\n"
        f"sounds: {sounds}\n"
    )
    link = ctx.new_link("videos", ".mp4", content)
    out.videos.append(link)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    videos = list(inp.bundle.videos)
    sounds = list(inp.bundle.sounds)
    fps = _float_arg(inp, "fps", 25.0)
    frames_per_chunk = _int_arg(inp, "frames_per_chunk", 81, min_value=1)
    delete_last_frames = _int_arg(inp, "delete_last_frames", 0, min_value=0)
    repeat = _int_arg(inp, "repeat", 120, min_value=1)
    alternate_flip = _bool_arg(inp, "flip", True)

    if not videos or not sounds:
        raise RuntimeError(_HELP.strip())

    if os.environ.get("AH_EMULATE_VIDEO_CLIP", "").lower() in ("1", "true", "yes"):
        out.videos = []
        _emulate(
            ctx,
            out,
            videos,
            sounds,
            fps,
            frames_per_chunk,
            delete_last_frames,
        )
        return out

    from externals.video_clip.encode import (
        build_joined_video_mp4,
        require_ffmpeg,
        require_moviepy,
    )

    require_ffmpeg()
    require_moviepy()

    video_paths = [ctx.resolve_link_path(link) for link in videos]
    for path in video_paths:
        if not path.is_file():
            raise FileNotFoundError(f"$video_clip: video not found: {path}")

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    out.videos = []
    for si, sound_link in enumerate(sounds):
        audio_path = ctx.resolve_link_path(sound_link)
        if not audio_path.is_file():
            raise FileNotFoundError(f"$video_clip: sound not found: {audio_path}")

        dest = videos_dir / _output_name(si)
        build_joined_video_mp4(
            video_paths,
            audio_path,
            dest,
            fps=fps,
            frames_per_chunk=frames_per_chunk,
            delete_last_frames=delete_last_frames,
            repeat=repeat,
            alternate_flip=alternate_flip,
        )
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.videos.append(link)

    return out
