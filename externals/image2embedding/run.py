"""$image2embedding — SigLIP 2 image vectors as base64 float16 text."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.image2embedding.embedding_format import emulated_siglip_embedding
from ahlib.ah_runtime import ArrayBundle


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE2EMBEDDING", "").lower() in (
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
    env = os.environ.get("AH_IMAGE2EMBEDDING_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _image_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.images:
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
        "$image2embedding requires torch, transformers, and Pillow (media venv).\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media / image2embedding)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "  model=default (google/siglip2-base-patch16-384) | model=224\n"
        "Test without models: AH_EMULATE_IMAGE2EMBEDDING=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.embeddings.clear()

    images = _image_paths(ctx, inp.bundle)
    if not images:
        return out

    use_gpu = _resolve_use_gpu(inp)
    model_name = read_arg_list(inp, "model", "default")[0]
    profile = _resolve_model(model_name)

    if _emulate_enabled():
        for image_path in images:
            out.embeddings.append(emulated_siglip_embedding(image_path.name))
        return out

    try:
        from externals.image2embedding.model_paths import ensure_model
        from externals.image2embedding.siglip2 import encode_image, load_model
    except ImportError as exc:
        raise RuntimeError(_help()) from exc

    model_dir = ensure_model(profile)
    model, processor = load_model(profile, model_dir, use_gpu=use_gpu)

    for image_path in images:
        if ctx.cancel_event is not None and ctx.cancel_event.is_set():
            from ahlib.ah_runtime import RuntimeCancelled

            raise RuntimeCancelled("$image2embedding cancelled")
        hex_vec = encode_image(
            profile,
            model,
            processor,
            image_path,
            use_gpu=use_gpu,
        )
        out.embeddings.append(hex_vec)

    return out
