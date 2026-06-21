"""Local paths and downloads for $image2text vision-language models."""

from __future__ import annotations

import json
from pathlib import Path

from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled
from externals.image.model_paths import models_roots
from externals.image2text.model_list import Image2TextModel, get_image2text_model


def _resolve_profile(profile: Image2TextModel | str | None) -> Image2TextModel:
    if isinstance(profile, Image2TextModel):
        return profile
    return get_image2text_model(profile or "qwen2")


def model_dir(profile: Image2TextModel | str | None = None) -> Path:
    m = _resolve_profile(profile)
    for root in models_roots():
        candidate = root / m.subdir
        if (candidate / "config.json").is_file():
            return candidate
    return models_roots()[0] / m.subdir


def _has_weights(path: Path) -> bool:
    if (path / "model.safetensors").is_file():
        return True
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shards = {str(v) for v in weight_map.values()}
        return bool(shards) and all((path / name).is_file() for name in shards)
    return any(path.glob("model-*.safetensors"))


def model_ready(profile: Image2TextModel | str | None = None) -> bool:
    path = model_dir(profile)
    return (path / "config.json").is_file() and _has_weights(path)


def ensure_model(profile: Image2TextModel | str | None = None, *, force: bool = False) -> Path:
    """Resolve VL weights under models/ (anthill bundle, optional upstream)."""
    m = _resolve_profile(profile)
    path = model_dir(m)
    path.mkdir(parents=True, exist_ok=True)
    if model_ready(m) and not force:
        return path

    try:
        ensure_anthill_tree(
            m.subdir.as_posix(),
            ready=lambda: model_ready(m),
            label="$image2text",
            force=force,
        )
    except FileNotFoundError:
        if not upstream_fallback_enabled():
            raise
    if model_ready(m):
        return model_dir(m)

    if upstream_fallback_enabled():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "$image2text needs huggingface-hub to download the model: "
                "uv sync --extra image2text"
            ) from exc

        print(f"$image2text: downloading {m.hf_repo} -> {path}", flush=True)
        snapshot_download(m.hf_repo, local_dir=str(path))
        if model_ready(m):
            return path

    if not model_ready(m):
        raise FileNotFoundError(
            f"Model not ready under {path}. "
            f"Run: uv run python tools/download_models.py "
            f"(or set AH_MODEL_UPSTREAM_FALLBACK=1)"
        )
    return path
