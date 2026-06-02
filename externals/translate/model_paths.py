"""Local paths and downloads for $translate (M2M100)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled
from externals.image.model_paths import models_roots
from externals.translate.model_list import TranslateModel, get_translate_model


def _resolve_profile(profile: TranslateModel | str | None) -> TranslateModel:
    if isinstance(profile, TranslateModel):
        return profile
    return get_translate_model(profile or "default")


def model_dir(profile: TranslateModel | str | None = None) -> Path:
    m = _resolve_profile(profile)
    for root in models_roots():
        candidate = root / m.subdir
        if (candidate / "config.json").is_file():
            return candidate
    return models_roots()[0] / m.subdir


def _has_weights(path: Path) -> bool:
    if (path / "pytorch_model.bin").is_file():
        return True
    if (path / "model.safetensors").is_file():
        return True
    return any(path.glob("model-*.safetensors"))


def model_ready(profile: TranslateModel | str | None = None) -> bool:
    path = model_dir(profile)
    return (
        (path / "config.json").is_file()
        and _has_weights(path)
        and (path / "sentencepiece.bpe.model").is_file()
    )


def ensure_model(profile: TranslateModel | str | None = None, *, force: bool = False) -> Path:
    """Resolve M2M100 weights under models/m2m100_1.2B/."""
    m = _resolve_profile(profile)
    path = model_dir(m)
    path.mkdir(parents=True, exist_ok=True)
    if model_ready(m) and not force:
        return path

    ensure_anthill_tree(
        m.subdir.as_posix(),
        ready=lambda: model_ready(m),
        label="$translate",
        force=force,
    )
    if model_ready(m):
        return model_dir(m)

    if upstream_fallback_enabled():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "$translate needs huggingface-hub to download the model: "
                "uv sync --extra media"
            ) from exc

        print(f"$translate: downloading {m.hf_repo} -> {path}", flush=True)
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
