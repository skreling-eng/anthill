"""$split_video_fast — scene split via ffmpeg-scaled frame hashes."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import make_label_entry


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_SPLIT_VIDEO_FAST", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _int_arg(args: dict[str, str], key: str, default: int, *, min_value: int = 1) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(
            f"$split_video_fast: {key}= must be >= {min_value}, got {value}"
        )
    return value


def _video_links(ctx: ExternalContext, bundle: ArrayBundle) -> list[str]:
    links: list[str] = []
    for link in bundle.videos:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            links.append(link)
    return links


def _output_name(source_stem: str, fragment_index: int, start_frame: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_stem)
    return f"{ts}_{safe}_frag{fragment_index:04d}_f{start_frame}.mp4"


def _help() -> str:
    return (
        "$split_video_fast needs videos[] in the input bundle.\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra split_video_fast)\n"
        "  uv run python tools/download_ffmpeg.py\n"
        "  threshold=10  hash_size=8  min_frames=100  width=320\n"
        "Test without ffmpeg: AH_EMULATE_SPLIT_VIDEO_FAST=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    video_links = _video_links(ctx, inp.bundle)
    if not video_links:
        raise RuntimeError(_help().strip())

    threshold = _int_arg(inp.args, "threshold", 10, min_value=0)
    hash_size = _int_arg(inp.args, "hash_size", 8, min_value=1)
    min_frames = _int_arg(inp.args, "min_frames", 100, min_value=1)
    frame_width = _int_arg(inp.args, "width", 320, min_value=1)

    out.videos.clear()
    out.labels.clear()

    if _emulate_enabled():
        for index, link in enumerate(video_links):
            src_path = ctx.resolve_link_path(link)
            content = (
                f"[emulated $split_video_fast threshold={threshold} "
                f"hash_size={hash_size} min_frames={min_frames} width={frame_width}] "
                f"{src_path.name}\n"
            )
            out_link = ctx.new_link("videos", ".mp4", content)
            out.videos.append(out_link)
            out.labels.append(
                make_label_entry(
                    "fragment",
                    [("videos", out_link)],
                    {"src": link.replace("\\", "/"), "frame": 0},
                )
            )
        return out

    from externals.split_video.cut import cut_fragment
    from externals.split_video_fast.split import _require_deps, detect_fragments
    from externals.video_audio.ffmpeg_io import require_ffmpeg

    _require_deps()
    require_ffmpeg()

    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    fragment_counter = 0
    for link in video_links:
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$split_video_fast cancelled")

        src_path = ctx.resolve_link_path(link)
        src_posix = link.replace("\\", "/")
        fragments, fps = detect_fragments(
            src_path,
            threshold=threshold,
            hash_size=hash_size,
            min_frames=min_frames,
            frame_width=frame_width,
        )
        if not fragments:
            print(
                f"$split_video_fast: {src_path.name} — no frames, skipped",
                flush=True,
            )
            continue

        print(
            f"$split_video_fast: {src_path.name} — {len(fragments)} fragment(s), "
            f"threshold={threshold}, min_frames={min_frames}, width={frame_width}",
            flush=True,
        )

        for frag in fragments:
            dest = videos_dir / _output_name(
                src_path.stem, fragment_counter, frag.start
            )
            cut_fragment(
                src_path,
                dest,
                start_frame=frag.start,
                end_frame=frag.end,
                fps=fps,
            )
            out_link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
            out.videos.append(out_link)
            out.labels.append(
                make_label_entry(
                    "fragment",
                    [("videos", out_link)],
                    {"src": src_posix, "frame": frag.start},
                )
            )
            fragment_counter += 1

    return out
