"""$search_local_video — search one FAISS index and return matching video fragments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list, read_prompt_texts
from externals.image2embedding.embedding_format import (
    emulated_siglip_embedding,
    unpack_siglip_embedding,
)
from externals.video_index.store import (
    FAISS_SUFFIX,
    MAP_SUFFIX,
    brute_force_search,
    index_path_for_map,
    load_index_pair,
    load_mapping,
    map_path_for_index,
    search_index,
)
from ahlib.ah_runtime import ArrayBundle
from ahlib.label_utils import make_label_entry


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_SEARCH_LOCAL_VIDEO", "").lower() in (
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
    env = os.environ.get("AH_SEARCH_LOCAL_VIDEO_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _int_arg(args: dict[str, str], key: str, default: int, *, min_value: int = 1) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(
            f"$search_local_video: {key}= must be >= {min_value}, got {value}"
        )
    return value


def _resolve_model(model_name: str):
    from externals.image2embedding.model_list import get_image2embedding_model

    raw = (model_name or "default").strip() or "default"
    try:
        return get_image2embedding_model(raw)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc


def _optional_int_arg(args: dict[str, str], key: str, *, min_value: int = 1) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < min_value:
        raise ValueError(
            f"$search_local_video: {key}= must be >= {min_value}, got {value}"
        )
    return value


def _help() -> str:
    return (
        "$search_local_video needs files[] (videos.faiss or videos.ahvmap.json) "
        "and a text prompt and/or videos[] query clip(s).\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media --extra video_index)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "  n=10 (default) — top matching fragments per query\n"
        "  every=N — sample every Nth frame for videos[] queries (default ~5 frames)\n"
        "  Multiple prompts[] or videos[] entries → one search each; results combined.\n"
        "Test without models/ffmpeg: AH_EMULATE_SEARCH_LOCAL_VIDEO=1"
    )


def _discover_index_source(
    ctx: ExternalContext, bundle: ArrayBundle
) -> tuple[Path | None, Path] | None:
    faiss_path: Path | None = None
    map_path: Path | None = None

    for link in bundle.files:
        path = ctx.resolve_link_path(link)
        if not path.is_file():
            continue
        if path.name.endswith(MAP_SUFFIX):
            map_path = path
            sibling = index_path_for_map(path)
            if sibling.is_file():
                faiss_path = sibling
        elif path.suffix == ".faiss" or link.endswith(FAISS_SUFFIX):
            faiss_path = path
            sibling = map_path_for_index(path)
            if sibling.is_file():
                map_path = sibling

    if map_path is None:
        return None
    return faiss_path, map_path


def _load_index_bundle(
    faiss_path: Path | None, map_path: Path
) -> tuple[object | None, dict, list[list[float]] | None]:
    meta = load_mapping(map_path)
    vectors = meta.get("vectors")
    if faiss_path is None:
        if not isinstance(vectors, list):
            vectors = None
        return None, meta, vectors
    index, meta = load_index_pair(faiss_path)
    return index, meta, None


@dataclass(frozen=True)
class _SearchQuery:
    label: str
    video_path: Path | None = None


def _search_queries(ctx: ExternalContext, inp: ExternalInput) -> list[_SearchQuery]:
    queries: list[_SearchQuery] = []
    for text in read_prompt_texts(ctx, inp):
        prompt = text.strip()
        if prompt:
            queries.append(_SearchQuery(label=prompt))
    for link in inp.bundle.videos:
        path = ctx.resolve_link_path(link)
        if path.is_file():
            label = str(path.resolve()).replace("\\", "/")
            queries.append(_SearchQuery(label=label, video_path=path))
    if not queries:
        raise RuntimeError(_help().strip())
    return queries


def _search_hits(
    index: object | None,
    meta: dict,
    vectors: list[list[float]] | None,
    query_vec: list[float],
    *,
    top_n: int,
) -> list[dict]:
    if index is not None:
        return search_index(index, meta, query_vec, k=top_n)
    if vectors is not None:
        return brute_force_search(meta, vectors, query_vec, k=top_n)
    return []


def _encode_text_query(
    query: str,
    *,
    emulate: bool,
    profile: object | None,
    model: object | None,
    processor: object | None,
    use_gpu: bool,
) -> list[float]:
    if emulate:
        return unpack_siglip_embedding(emulated_siglip_embedding(query))

    assert profile is not None and model is not None and processor is not None
    from externals.image2embedding.siglip2 import encode_text

    encoded = encode_text(profile, model, processor, query, use_gpu=use_gpu)
    return unpack_siglip_embedding(encoded)


def _encode_video_query(
    video_path: Path,
    *,
    emulate: bool,
    profile: object | None,
    model: object | None,
    processor: object | None,
    use_gpu: bool,
    every_nth: int | None,
) -> list[float]:
    if emulate:
        return unpack_siglip_embedding(emulated_siglip_embedding(video_path.name))

    assert profile is not None and model is not None and processor is not None
    from externals.image2embedding.siglip2 import encode_pil_images_averaged
    from externals.video2embedding.frames import sample_video_frames

    frames = sample_video_frames(video_path, every_nth=every_nth)
    if not frames:
        raise RuntimeError(f"$search_local_video: no frames in {video_path.name}")
    encoded = encode_pil_images_averaged(
        profile,
        model,
        processor,
        frames,
        use_gpu=use_gpu,
    )
    return unpack_siglip_embedding(encoded)


def _encode_search_query(
    query: _SearchQuery,
    *,
    emulate: bool,
    profile: object | None,
    model: object | None,
    processor: object | None,
    use_gpu: bool,
    every_nth: int | None,
) -> list[float]:
    if query.video_path is not None:
        return _encode_video_query(
            query.video_path,
            emulate=emulate,
            profile=profile,
            model=model,
            processor=processor,
            use_gpu=use_gpu,
            every_nth=every_nth,
        )
    return _encode_text_query(
        query.label,
        emulate=emulate,
        profile=profile,
        model=model,
        processor=processor,
        use_gpu=use_gpu,
    )


_MAX_FILENAME_LEN = 260
_MAX_SOURCE_STEM_LEN = 100
_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def _sanitize_stem(source_stem: str) -> str:
    cleaned = "".join(
        c if c not in _INVALID_FILENAME_CHARS and ord(c) >= 32 else "_"
        for c in source_stem
    )
    return cleaned[:_MAX_SOURCE_STEM_LEN]


def _output_name(source_stem: str, rank: int, start_frame: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = _sanitize_stem(source_stem)
    suffix = f"_search{rank:03d}_f{start_frame}.mp4"
    name = f"{ts}_{safe}{suffix}"
    if len(name) > _MAX_FILENAME_LEN:
        budget = _MAX_FILENAME_LEN - len(f"{ts}_{suffix}")
        safe = safe[: max(budget, 0)]
        name = f"{ts}_{safe}{suffix}"
    return name


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    source = _discover_index_source(ctx, inp.bundle)
    if source is None:
        raise RuntimeError(_help().strip())

    queries = _search_queries(ctx, inp)
    top_n = _int_arg(inp.args, "n", 10)
    every_nth = _optional_int_arg(inp.args, "every", min_value=1)
    use_gpu = _resolve_use_gpu(inp)
    model_name = read_arg_list(inp, "model", "default")[0]
    emulate = _emulate_enabled()

    faiss_path, map_path = source
    index, meta, vectors = _load_index_bundle(faiss_path, map_path)

    profile = model = processor = None
    if not emulate:
        try:
            from externals.image2embedding.model_paths import ensure_model
            from externals.image2embedding.siglip2 import load_model
        except ImportError as exc:
            raise RuntimeError(_help()) from exc
        profile = _resolve_model(model_name)
        model_dir = ensure_model(profile)
        model, processor = load_model(profile, model_dir, use_gpu=use_gpu)

    out = ArrayBundle()
    out.prompts.clear()

    all_hits: list[tuple[str, dict]] = []
    for query in queries:
        query_vec = _encode_search_query(
            query,
            emulate=emulate,
            profile=profile,
            model=model,
            processor=processor,
            use_gpu=use_gpu,
            every_nth=every_nth,
        )
        for hit in _search_hits(index, meta, vectors, query_vec, top_n=top_n):
            all_hits.append((query.label, hit))

    if not all_hits:
        print("$search_local_video: no matches", flush=True)
        return out

    if emulate:
        for output_index, (query, hit) in enumerate(all_hits):
            content = (
                f"[emulated $search_local_video n={top_n}]\n"
                f"query: {query}\n"
                f"video: {hit['video']}\n"
                f"start: {hit['start']}\n"
                f"end: {hit['end']}\n"
                f"closeness: {hit['closeness']}\n"
            )
            out_link = ctx.new_link("videos", ".mp4", content)
            out.videos.append(out_link)
            out.labels.append(
                make_label_entry(
                    "search_results",
                    [("videos", out_link)],
                    {
                        "query": query,
                        "src": hit["video"].replace("\\", "/"),
                        "ahvemb": hit["ahvemb"].replace("\\", "/"),
                        "start": hit["start"],
                        "end": hit["end"],
                        "closeness": hit["closeness"],
                    },
                )
            )
        return out

    from externals.split_video.cut import cut_fragment
    from externals.video_audio.ffmpeg_io import require_ffmpeg

    require_ffmpeg()
    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for output_index, (query, hit) in enumerate(all_hits):
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$search_local_video cancelled")

        src_path = Path(hit["video"])
        if not src_path.is_file():
            raise FileNotFoundError(f"$search_local_video: video not found: {src_path}")

        dest = videos_dir / _output_name(src_path.stem, output_index, hit["start"])
        cut_fragment(
            src_path,
            dest,
            start_frame=hit["start"],
            end_frame=hit["end"],
            fps=hit["fps"],
        )
        out_link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.videos.append(out_link)
        out.labels.append(
            make_label_entry(
                "search_results",
                [("videos", out_link)],
                {
                    "query": query,
                    "src": hit["video"].replace("\\", "/"),
                    "ahvemb": hit["ahvemb"].replace("\\", "/"),
                    "start": hit["start"],
                    "end": hit["end"],
                    "closeness": hit["closeness"],
                },
            )
        )

    print(
        f"$search_local_video: {len(out.videos)} fragment(s) "
        f"from {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}",
        flush=True,
    )
    return out
