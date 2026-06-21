"""Wan checkpoint resolution for $image2video (comfy_lib in-process)."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import require_models_file
from externals.image.model_paths import models_roots

MODEL_ALIASES: dict[str, str] = {
    "default": "wan2.2-rapid-mega-aio-v12.safetensors",
    "mega": "wan2.2-rapid-mega-aio-v12.safetensors",
    "mega-nsfw": "wan2.2-rapid-mega-aio-nsfw-v12.2.safetensors",
    "wan": "wan2.2-i2v-rapid-aio-v10.safetensors",
    "rapid": "wan2.2-i2v-rapid-aio-v10.safetensors",
    "i2v": "wan2.2-i2v-rapid-aio-v10.safetensors",
}

MEGA_WORKFLOW = "Rapid-AIO-Mega__3_start_image.json"
DEFAULT_CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"


def is_mega_model(model_arg: str) -> bool:
    name = _resolve_filename(model_arg).lower()
    return "mega" in name

DEFAULT_MODEL = "mega"
WAN_SUBDIR = Path("WAN")


def available_models() -> str:
    return ", ".join(sorted(MODEL_ALIASES))


def _resolve_filename(model_arg: str) -> str:
    raw = (model_arg or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return MODEL_ALIASES.get(raw, raw)


def _search_roots() -> list[Path]:
    roots: list[Path] = []
    for root in models_roots():
        roots.extend(
            [
                root / "wan",
                root / "checkpoints" / WAN_SUBDIR,
                root / "checkpoints",
            ]
        )
    return roots


def resolve_checkpoint(model_arg: str = "") -> Path:
    """Resolve model= to an absolute .safetensors path."""
    name = _resolve_filename(model_arg)
    path = Path(name)
    if path.is_file():
        return path.resolve()

    if not name.endswith(".safetensors"):
        name = f"{name}.safetensors"

    for root in _search_roots():
        for candidate in (root / name, root / Path(name).name):
            if candidate.is_file():
                return candidate.resolve()

    wan_rel = f"wan/{name}"
    try:
        return require_models_file(wan_rel, label="$image2video")
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"$image2video checkpoint not found: {model_arg or DEFAULT_MODEL!r}. "
        f"Place under models/wan/ "
        f"(models: {available_models()}). "
        f"Run: uv run python tools/download_models.py"
    )


def comfy_ckpt_name(model_arg: str = "") -> str:
    """Checkpoint name for CheckpointLoaderSimple (WAN\\file.safetensors)."""
    path = resolve_checkpoint(model_arg)
    for root in models_roots():
        wan_dir = root / "wan"
        if wan_dir.is_dir():
            try:
                path.relative_to(wan_dir.resolve())
                return path.name
            except ValueError:
                pass
        ckpt_wan = root / "checkpoints" / WAN_SUBDIR
        if ckpt_wan.is_dir():
            try:
                rel = path.relative_to(ckpt_wan.resolve())
                return str(WAN_SUBDIR / rel).as_posix().replace("/", "\\")
            except ValueError:
                pass
    return f"{WAN_SUBDIR}\\{path.name}"
