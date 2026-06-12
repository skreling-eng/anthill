"""$create_video_index — build one FAISS HNSW index from all .ahvemb sidecars."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.video_index.ahvemb import ahvemb_path_for_video, parse_ahvemb_file
from externals.video_index.store import (
    FAISS_SUFFIX,
    INDEX_STEM,
    MAP_SUFFIX,
    build_combined_mapping,
    build_faiss_index,
    read_video_fps,
    save_index_pair,
    save_mapping_only,
)
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_CREATE_VIDEO_INDEX", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _video_links(ctx: ExternalContext, bundle: ArrayBundle) -> list[str]:
    links: list[str] = []
    for link in bundle.videos:
        path = ctx.resolve_link_path(link)
        if path.is_file():
            links.append(link)
    return links


def _help() -> str:
    return (
        "$create_video_index needs videos[] with matching .ahvemb sidecars.\n"
        "  tools\\setup_external_venvs.ps1   (or UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media --extra video_index)\n"
        "  Writes one videos.faiss (+ sibling videos.ahvmap.json on disk) into files[] only.\n"
        "Test without faiss: AH_EMULATE_CREATE_VIDEO_INDEX=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    video_links = _video_links(ctx, inp.bundle)
    if not video_links:
        return ArrayBundle()

    emulate = _emulate_enabled()
    rows: list[tuple[Path, Path, float, int, int, list[float]]] = []

    for link in video_links:
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$create_video_index cancelled")

        src_path = ctx.resolve_link_path(link)
        sidecar = ahvemb_path_for_video(src_path)
        if not sidecar.is_file():
            print(
                f"$create_video_index: {src_path.name} — "
                f"{sidecar.name} missing, skipped",
                flush=True,
            )
            continue

        fragments = parse_ahvemb_file(sidecar)
        if not fragments:
            print(
                f"$create_video_index: {src_path.name} — "
                f"empty {sidecar.name}, skipped",
                flush=True,
            )
            continue

        fps = read_video_fps(src_path)
        for start, end, vec in fragments:
            rows.append((src_path, sidecar, fps, start, end, vec))
        print(
            f"$create_video_index: {src_path.name} — "
            f"{len(fragments)} fragment(s)",
            flush=True,
        )

    if not rows:
        return ArrayBundle()

    meta = build_combined_mapping(rows)
    files_dir = ctx.op_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    index_path = files_dir / f"{INDEX_STEM}{FAISS_SUFFIX}"
    map_path = files_dir / f"{INDEX_STEM}{MAP_SUFFIX}"

    if emulate:
        meta["vectors"] = [vec for *_rest, vec in rows]
        save_mapping_only(map_path, meta)
        out_link = str(map_path.relative_to(ctx.base_dir)).replace("\\", "/")
    else:
        try:
            index = build_faiss_index(rows)
        except RuntimeError as exc:
            raise RuntimeError(_help()) from exc
        save_index_pair(index_path, index, meta)
        out_link = str(index_path.relative_to(ctx.base_dir)).replace("\\", "/")

    out = ArrayBundle()
    out.files.append(out_link)
    print(
        f"$create_video_index: {len(rows)} fragment(s) from "
        f"{len({row[0] for row in rows})} video(s) -> {Path(out_link).name}",
        flush=True,
    )
    return out
