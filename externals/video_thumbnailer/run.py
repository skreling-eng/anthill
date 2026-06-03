"""$video_thumbnailer — contact-sheet preview image per video."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.video_thumbnailer.preview import PreviewBuildError, build_preview_image
from externals.video_thumbnailer.settings import ThumbnailOptions, options_from_input
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$video_thumbnailer needs videos[] in the input bundle.

Example:
  @previews: $file('clip.mp4') -> $video_thumbnailer -> $save('clip_preview.jpg')

Builds one JPEG contact sheet per input video (grid of frames + optional metadata header).
Set AH_EMULATE_VIDEO_THUMBNAILER=1 for a stub without PyAV/MediaInfo.

Requires: uv sync --extra video_thumbnailer
"""


def _resolve_videos(ctx: ExternalContext, links: list[str]) -> list[Path]:
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
    return f"{ts}_video_thumbnailer_{index}.jpg"


def _require_deps() -> None:
    missing: list[str] = []
    try:
        import av  # noqa: F401
    except ImportError:
        missing.append("av")
    try:
        import pymediainfo  # noqa: F401
    except ImportError:
        missing.append("pymediainfo")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    if missing:
        raise ImportError(
            f"$video_thumbnailer needs {', '.join(missing)}. "
            "Install with: uv sync --extra video_thumbnailer "
            "or set AH_EMULATE_VIDEO_THUMBNAILER=1"
        )


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    videos: list[Path],
    options: ThumbnailOptions,
) -> ArrayBundle:
    out.videos.clear()
    out.images.clear()
    for index, path in enumerate(videos):
        marker = (
            f"[emulated $video_thumbnailer]\n"
            f"source: {path.name}\n"
            f"grid: {options.columns}x{options.rows} width={options.width}\n"
        ).encode("utf-8")
        out.images.append(ctx.new_link("images", ".jpg", marker))
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    videos = _resolve_videos(ctx, inp.bundle.videos)
    if not videos:
        raise RuntimeError(_HELP.strip())

    options = options_from_input(inp)
    emulate = os.environ.get("AH_EMULATE_VIDEO_THUMBNAILER", "").lower() in (
        "1",
        "true",
        "yes",
    )

    out.videos.clear()

    if emulate:
        return _emulate(ctx, out, videos, options)

    _require_deps()

    images_dir = ctx.op_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out.images.clear()

    for index, video_path in enumerate(videos):
        try:
            sheet = build_preview_image(video_path, options)
        except PreviewBuildError as exc:
            raise RuntimeError(f"$video_thumbnailer: {exc}") from exc

        dest = images_dir / _output_name(index)
        sheet.save(dest, format="JPEG", quality=options.jpeg_quality)
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.images.append(link)

    return out
