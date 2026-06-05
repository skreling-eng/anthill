"""$add_soft_subtitles — burn subtitles from texts[] onto each videos[] clip."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$add_soft_subtitles needs videos[] and texts[] in the input bundle.

Example:
  @subs: $file('episode.ass', source_path=True)
  @titled: (@clip, @subs) -> $add_soft_subtitles -> $save('episode_subs.mp4')

Pairing: equal count (zip), one text for all videos, or pad shorter texts[] with last entry.

Optional: font=, size=, bottom=  (font required only for plain-text drawtext mode)

Requires ffmpeg: uv run python tools/download_ffmpeg.py (tools/ffmpeg/) or PATH.
Set AH_EMULATE_ADD_SOFT_SUBTITLES=1 for a stub.
"""

_DEFAULT_FONT = "ttf/MuseoSansCyrl-700.ttf"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_paths(ctx: ExternalContext, links: list[str]) -> list[Path]:
    paths: list[Path] = []
    for link in links:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _resolve_link_path(ctx: ExternalContext, link: str) -> Path | None:
    if not link.strip():
        return None
    path = Path(link)
    if not path.is_absolute():
        path = (ctx.base_dir / link).resolve()
    return path if path.is_file() else None


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_soft_subs_{index}.mp4"


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _prepare_text(text: str) -> str:
    if len(text) > 10:
        text = re.sub(r"\^", "\n", text)
    return text.lstrip("\ufeff")


def _text_links_for_videos(
    ctx: ExternalContext, inp: ExternalInput, video_count: int
) -> list[str]:
    if inp.args.get("text", "").strip():
        return [""] * video_count
    links = list(inp.bundle.texts)
    if not links:
        return [""] * video_count
    if len(links) == 1:
        return [links[0]] * video_count
    if len(links) >= video_count:
        return links[:video_count]
    padded = list(links)
    padded.extend([links[-1]] * (video_count - len(padded)))
    return padded


def _resolve_font_path(font_arg: str) -> Path | None:
    for base in (_REPO_ROOT, Path.cwd()):
        path = (base / font_arg).resolve()
        if path.is_file():
            return path
    path = Path(font_arg)
    if path.is_file():
        return path.resolve()
    return None


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    videos = _resolve_paths(ctx, inp.bundle.videos)
    if not videos:
        raise RuntimeError(_HELP.strip())

    inline_text = inp.args.get("text", "").strip()
    text_links = _text_links_for_videos(ctx, inp, len(videos))
    font_arg = inp.args.get("font", _DEFAULT_FONT).strip() or _DEFAULT_FONT
    font_size = _int_arg(inp.args, "size", 40)
    bottom_margin = _int_arg(inp.args, "bottom", 40)

    out.sounds.clear()
    out.images.clear()
    out.texts.clear()
    out.prompts.clear()
    out.files.clear()
    out.embeddings.clear()
    out.labels.clear()
    out.changes.clear()

    from externals.video_audio.ffmpeg_io import (
        is_subtitle_file,
        pair_videos_and_text_links,
        pair_videos_and_texts,
    )

    if os.environ.get("AH_EMULATE_ADD_SOFT_SUBTITLES", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        texts = (
            [_prepare_text(inline_text)] * len(videos)
            if inline_text
            else [
                _prepare_text(ctx.read_link_text(link)) if link else ""
                for link in text_links
            ]
        )
        out.videos.clear()
        for video_path, text in pair_videos_and_texts(videos, texts):
            content = (
                f"[emulated $add_soft_subtitles]\n"
                f"video: {video_path.name}\n"
                f"text:\n{text[:500]}\n"
            )
            out.videos.append(ctx.new_link("videos", ".mp4", content))
        return out

    from externals.video_audio.ffmpeg_io import add_soft_subtitles, require_ffmpeg

    require_ffmpeg()

    font_path = _resolve_font_path(font_arg)
    videos_dir = ctx.op_dir / "videos"
    texts_dir = ctx.op_dir / "texts"
    videos_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    out.videos.clear()

    for index, (video_path, text_link) in enumerate(
        pair_videos_and_text_links(videos, text_links)
    ):
        text_file = texts_dir / f"subs_{index}.txt"
        dest = videos_dir / _output_name(index)
        subtitle_src = _resolve_link_path(ctx, text_link)
        if inline_text and not text_link:
            text = _prepare_text(inline_text)
            subtitle_src = None
        elif subtitle_src is not None and is_subtitle_file(subtitle_src):
            text = ""
        else:
            text = _prepare_text(
                ctx.read_link_text(text_link) if text_link else ""
            )
            subtitle_src = None
        add_soft_subtitles(
            video_path,
            dest,
            text,
            work_dir=ctx.op_dir,
            subtitle_file=text_file,
            font_path=font_path,
            subtitle_src=subtitle_src,
            font_size=font_size,
            bottom_margin=bottom_margin,
        )
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.videos.append(link)

    return out
