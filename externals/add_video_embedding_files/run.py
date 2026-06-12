"""$add_video_embedding_files — write per-fragment .ahvemb sidecars next to source videos."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.image2embedding.embedding_format import emulated_siglip_embedding
from externals.video_index.ahvemb import ahvemb_path_for_video
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _resolve_use_gpu(inp: ExternalInput) -> bool:
    raw = inp.args.get("gpu", "").strip()
    if raw:
        return _truthy(raw)
    env = os.environ.get("AH_ADD_VIDEO_EMBEDDING_FILES_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _int_arg(
    args: dict[str, str], key: str, default: int, *, min_value: int = 1
) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(
            f"$add_video_embedding_files: {key}= must be >= {min_value}, got {value}"
        )
    return value


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _video_links(ctx: ExternalContext, bundle: ArrayBundle) -> list[str]:
    links: list[str] = []
    for link in bundle.videos:
        path = ctx.resolve_link_path(link)
        if path.is_file():
            links.append(link)
    return links


def _resolve_model(model_name: str):
    from externals.image2embedding.model_list import get_image2embedding_model

    raw = (model_name or "default").strip() or "default"
    try:
        return get_image2embedding_model(raw)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc


def _help() -> str:
    return (
        "$add_video_embedding_files needs videos[] in the input bundle.\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "  Splits each video in memory, picks random sample frames per fragment,\n"
        "  averages SigLIP embeddings, writes .ahvemb/<video>.ahvemb beside the source.\n"
        "  samples=5  threshold=10  hash_size=8  min_frames=100  width=320\n"
        "  overwrite=True — rewrite existing .ahvemb (default: skip if present)\n"
        "Test without models: AH_EMULATE_ADD_VIDEO_EMBEDDING_FILES=1"
    )


def _optional_int_arg(
    args: dict[str, str], key: str, *, min_value: int = 1
) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < min_value:
        raise ValueError(
            f"$add_video_embedding_files: {key}= must be >= {min_value}, got {value}"
        )
    return value


def _write_ahvemb(dest: Path, lines: list[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    video_links = _video_links(ctx, inp.bundle)
    if not video_links:
        return inp.bundle.copy()

    threshold = _int_arg(inp.args, "threshold", 10, min_value=0)
    hash_size = _int_arg(inp.args, "hash_size", 8, min_value=1)
    min_frames = _int_arg(inp.args, "min_frames", 100, min_value=1)
    sample_count = _int_arg(inp.args, "samples", 5, min_value=1)
    every_nth = _optional_int_arg(inp.args, "every", min_value=1)
    frame_width = _int_arg(inp.args, "width", 320, min_value=1)
    overwrite = _truthy(inp.args.get("overwrite", ""))
    use_gpu = _resolve_use_gpu(inp)
    model_name = read_arg_list(inp, "model", "default")[0]
    profile = _resolve_model(model_name)
    emulate = _emulate_enabled()

    from externals.split_video_fast.split import _require_deps, detect_fragments
    from externals.video2embedding.frames import sample_fragment_frames
    from externals.video_audio.ffmpeg_io import require_ffmpeg

    _require_deps()
    require_ffmpeg()

    model = processor = None
    if not emulate:
        try:
            from externals.image2embedding.model_paths import ensure_model
            from externals.image2embedding.siglip2 import (
                encode_pil_images_averaged,
                load_model,
            )
        except ImportError as exc:
            raise RuntimeError(_help()) from exc
        model_dir = ensure_model(profile)
        model, processor = load_model(profile, model_dir, use_gpu=use_gpu)

    _log(f"$add_video_embedding_files: processing {len(video_links)} video(s)")

    for video_index, link in enumerate(video_links, start=1):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$add_video_embedding_files cancelled")

        src_path = ctx.resolve_link_path(link)
        dest = ahvemb_path_for_video(src_path)
        video_note = f"[{video_index}/{len(video_links)}] {src_path.name}"
        if dest.is_file() and not overwrite:
            _log(
                f"$add_video_embedding_files: {video_note} — "
                f"{dest.name} exists, skipped"
            )
            continue

        _log(f"$add_video_embedding_files: {video_note} — detecting fragments")
        fragments, _fps = detect_fragments(
            src_path,
            threshold=threshold,
            hash_size=hash_size,
            min_frames=min_frames,
            sample_count=0 if every_nth is not None else sample_count,
            frame_width=frame_width,
        )
        if not fragments:
            _log(f"$add_video_embedding_files: {video_note} — no frames, skipped")
            continue

        _log(
            f"$add_video_embedding_files: {video_note} — "
            f"{len(fragments)} fragment(s), embedding"
        )
        lines: list[str] = []
        for frag_index, frag in enumerate(fragments, start=1):
            _log(
                f"$add_video_embedding_files: {video_note} — "
                f"fragment {frag_index}/{len(fragments)} "
                f"frames {frag.start}-{frag.end}"
                + (
                    f" samples={len(frag.sample_images)}"
                    if frag.sample_images
                    else ""
                )
            )
            if emulate:
                encoded = emulated_siglip_embedding(
                    f"{src_path}:{frag.start}:{frag.end}"
                )
            else:
                if frag.sample_images:
                    frames = list(frag.sample_images)
                else:
                    frames = sample_fragment_frames(
                        src_path,
                        frag.start,
                        frag.end,
                        every_nth=every_nth,
                    )
                if not frames:
                    continue
                encoded = encode_pil_images_averaged(
                    profile,
                    model,
                    processor,
                    frames,
                    use_gpu=use_gpu,
                )
            lines.append(f"{frag.start} {frag.end} {encoded}")

        _write_ahvemb(dest, lines)
        _log(
            f"$add_video_embedding_files: {video_note} — "
            f"wrote {dest.name} ({len(lines)} fragment(s))"
        )

    return inp.bundle.copy()
