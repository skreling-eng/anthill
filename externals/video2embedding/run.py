"""$video2embedding — averaged SigLIP 2 embedding from sampled video frames."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.image2embedding.embedding_format import emulated_siglip_embedding
from externals.video2embedding.frames import sample_video_frames
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_VIDEO2EMBEDDING", "").lower() in (
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
    env = os.environ.get("AH_VIDEO2EMBEDDING_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _optional_int_arg(args: dict[str, str], key: str, *, min_value: int = 1) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < min_value:
        raise ValueError(f"$video2embedding: {key}= must be >= {min_value}, got {value}")
    return value


def _video_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.videos:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _resolve_model(model_name: str):
    from externals.image2embedding.model_list import get_image2embedding_model

    raw = (model_name or "default").strip() or "default"
    try:
        return get_image2embedding_model(raw)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc


def _help() -> str:
    return (
        "$video2embedding requires torch, transformers, Pillow, and opencv-python.\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media / video2embedding)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "  Default: ~5 frames per video; every=N overrides (sample every Nth frame)\n"
        "Test without models: AH_EMULATE_VIDEO2EMBEDDING=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.embeddings.clear()

    videos = _video_paths(ctx, inp.bundle)
    if not videos:
        return out

    use_gpu = _resolve_use_gpu(inp)
    every_nth = _optional_int_arg(inp.args, "every", min_value=1)
    model_name = read_arg_list(inp, "model", "default")[0]
    profile = _resolve_model(model_name)

    if _emulate_enabled():
        for video_path in videos:
            out.embeddings.append(emulated_siglip_embedding(video_path.name))
        return out

    try:
        from externals.image2embedding.model_paths import ensure_model
        from externals.image2embedding.siglip2 import encode_pil_images_averaged, load_model
    except ImportError as exc:
        raise RuntimeError(_help()) from exc

    model_dir = ensure_model(profile)
    model, processor = load_model(profile, model_dir, use_gpu=use_gpu)

    for video_path in videos:
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$video2embedding cancelled")
        frames = sample_video_frames(video_path, every_nth=every_nth)
        if not frames:
            continue
        if every_nth is not None:
            sample_note = f"every={every_nth}"
        else:
            sample_note = f"auto ~5 frames"
        print(
            f"$video2embedding: {video_path.name} — {len(frames)} frame(s), {sample_note}",
            flush=True,
        )
        encoded = encode_pil_images_averaged(
            profile,
            model,
            processor,
            frames,
            use_gpu=use_gpu,
        )
        out.embeddings.append(encoded)

    return out
