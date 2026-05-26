"""$image_clip — slideshow video from images, timed to a sound file."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$image_clip needs images[] and sounds[] in the input bundle.

Example:
  @images: ... -> $image(...)
  @music: $file('track.wav')
  @clip: (@images, @music) -> $image_clip -> $save('output.mp4')

Optional args: fps=25

Requires ffmpeg/ffprobe (slideshow) and moviepy (audio mux). Set AH_EMULATE_IMAGE_CLIP=1 for a stub.
"""


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_image_clip_{index}.mp4"


def _fps(inp: ExternalInput) -> float:
    raw = inp.args.get("fps", "25").strip()
    try:
        fps = float(raw)
    except ValueError as exc:
        raise ValueError(f"$image_clip: invalid fps={raw!r}") from exc
    if fps <= 0:
        raise ValueError(f"$image_clip: fps must be positive, got {fps}")
    return fps


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    images: list[str],
    sounds: list[str],
    fps: float,
) -> None:
    content = (
        f"[emulated $image_clip fps={fps}]\n"
        f"images: {images}\n"
        f"sounds: {sounds}\n"
    )
    link = ctx.new_link("videos", ".mp4", content)
    out.videos.append(link)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    images = list(inp.bundle.images)
    sounds = list(inp.bundle.sounds)
    fps = _fps(inp)

    if not images or not sounds:
        raise RuntimeError(_HELP.strip())

    if os.environ.get("AH_EMULATE_IMAGE_CLIP", "").lower() in ("1", "true", "yes"):
        _emulate(ctx, out, images, sounds, fps)
        return out

    from externals.image_clip.encode import (
        build_slideshow_mp4,
        require_ffmpeg,
        require_moviepy,
    )

    require_ffmpeg()
    require_moviepy()

    image_paths = [(ctx.base_dir / link).resolve() for link in images]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(f"$image_clip: image not found: {path}")

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for si, sound_link in enumerate(sounds):
        audio_path = (ctx.base_dir / sound_link).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"$image_clip: sound not found: {audio_path}")

        with tempfile.TemporaryDirectory(prefix="image_clip_") as tmp:
            work = Path(tmp)
            mp4_path = build_slideshow_mp4(
                image_paths,
                audio_path,
                fps=fps,
                work_dir=work,
            )
            dest = videos_dir / _output_name(si)
            dest.write_bytes(mp4_path.read_bytes())

        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.videos.append(link)

    return out
