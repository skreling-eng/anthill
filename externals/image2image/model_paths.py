"""Qwen-Rapid-AIO checkpoint under models/qwen-rapid/."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import require_models_file
from externals.image.model_paths import models_roots

MODEL_ALIASES: dict[str, str] = {
    "sfw-v23": "Qwen-Rapid-AIO-SFW-v23.safetensors",
    "nsfw-v23": "Qwen-Rapid-AIO-NSFW-v23.safetensors",
}

DEFAULT_MODEL = "sfw-v23"
DEFAULT_CKPT = MODEL_ALIASES[DEFAULT_MODEL]
MODEL_SUBDIR = Path("qwen-rapid")


def available_models() -> str:
    return ", ".join(sorted(MODEL_ALIASES))


def _repo_model_path(name: str) -> Path:
    for root in models_roots():
        candidate = root / MODEL_SUBDIR / name
        if candidate.is_file():
            return candidate.resolve()
    return (models_roots()[0] / MODEL_SUBDIR / name).resolve()


def _resolve_model_ref(model_arg: str) -> str:
    raw = (model_arg or DEFAULT_MODEL).strip()
    if not raw:
        raw = DEFAULT_MODEL
    return MODEL_ALIASES.get(raw, raw)


def resolve_checkpoint(model_arg: str = "") -> Path:
    """Resolve model= to an absolute .safetensors path."""
    raw = _resolve_model_ref(model_arg)

    path = Path(raw)
    if path.is_file():
        return path.resolve()

    for root in models_roots():
        for candidate in (
            root / MODEL_SUBDIR / raw,
            root / raw,
            root / MODEL_SUBDIR / Path(raw).name,
        ):
            if candidate.is_file():
                return candidate.resolve()

    if not raw.endswith(".safetensors"):
        named = _repo_model_path(raw + ".safetensors")
        if named.is_file():
            return named

    named = _repo_model_path(Path(raw).name)
    if named.is_file():
        return named

    # Unified fetch: local models/ then skreling-eng/anthill on demand.
    stem = Path(raw).stem if raw.endswith(".safetensors") else raw
    for rel in (
        f"qwen-rapid/{stem}.safetensors",
        f"qwen-rapid/{Path(raw).name}",
        f"qwen-rapid/{DEFAULT_CKPT}",
    ):
        try:
            return require_models_file(rel, label="$image2image")
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"$image2image checkpoint not found: {model_arg or DEFAULT_MODEL!r}. "
        f"Expected under models/qwen-rapid/ "
        f"(models: {available_models()}). "
        f"Run: uv run python tools/download_models.py"
    )


def model_ready(model_arg: str = "") -> bool:
    try:
        return resolve_checkpoint(model_arg).is_file()
    except FileNotFoundError:
        return False
