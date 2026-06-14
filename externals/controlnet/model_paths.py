"""Qwen-Image + InstantX Union ControlNet weights under models/qwen-image/."""

from __future__ import annotations

from pathlib import Path

from externals.anthill_models import ensure_anthill_files
from externals.image.model_paths import models_roots

QWEN_IMAGE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
CONTROLNET_REPO = "InstantX/Qwen-Image-ControlNet-Union"

MODEL_SUBDIR = Path("qwen-image")

UNET_NAME = "qwen_image_fp8_e4m3fn.safetensors"
CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
CONTROLNET_NAME = "Qwen-Image-InstantX-ControlNet-Union.safetensors"

_UPSTREAM_FILES = {
    "diffusion_models": (QWEN_IMAGE_REPO, f"split_files/diffusion_models/{UNET_NAME}"),
    "text_encoders": (QWEN_IMAGE_REPO, f"split_files/text_encoders/{CLIP_NAME}"),
    "vae": (QWEN_IMAGE_REPO, f"split_files/vae/{VAE_NAME}"),
    "controlnet": (
        CONTROLNET_REPO,
        "diffusion_pytorch_model.safetensors",
        CONTROLNET_NAME,
    ),
}


def qwen_image_root() -> Path:
    for root in models_roots():
        candidate = root / MODEL_SUBDIR
        if candidate.is_dir():
            return candidate
    return (models_roots()[0] / MODEL_SUBDIR).resolve()


def model_file(kind: str) -> Path:
    names = {
        "unet": UNET_NAME,
        "clip": CLIP_NAME,
        "vae": VAE_NAME,
        "controlnet": CONTROLNET_NAME,
    }
    subdirs = {
        "unet": "diffusion_models",
        "clip": "text_encoders",
        "vae": "vae",
        "controlnet": "controlnet",
    }
    return qwen_image_root() / subdirs[kind] / names[kind]


def models_ready() -> bool:
    return all(model_file(kind).is_file() for kind in ("unet", "clip", "vae", "controlnet"))


def _anthill_rels() -> list[str]:
    return [
        f"qwen-image/diffusion_models/{UNET_NAME}",
        f"qwen-image/text_encoders/{CLIP_NAME}",
        f"qwen-image/vae/{VAE_NAME}",
        f"qwen-image/controlnet/{CONTROLNET_NAME}",
    ]


def _download_upstream() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "$controlnet needs huggingface-hub: uv sync --extra media"
        ) from exc

    root = qwen_image_root()
    for kind, spec in _UPSTREAM_FILES.items():
        dest_dir = root / kind
        dest_dir.mkdir(parents=True, exist_ok=True)
        if len(spec) == 3:
            repo, remote_name, local_name = spec
            dest = dest_dir / local_name
        else:
            repo, remote_name = spec
            dest = dest_dir / Path(remote_name).name
        if dest.is_file():
            continue
        print(f"$controlnet: downloading {repo}/{remote_name} -> {dest}", flush=True)
        cached = hf_hub_download(repo, remote_name)
        if not dest.is_file():
            dest.write_bytes(Path(cached).read_bytes())


def ensure_models(*, force: bool = False) -> Path:
    """Resolve Qwen-Image FP8 + InstantX Union ControlNet; fetch on demand."""
    if models_ready() and not force:
        return qwen_image_root()

    try:
        ensure_anthill_files(_anthill_rels(), label="$controlnet", force=force)
    except Exception:
        pass
    if models_ready():
        return qwen_image_root()

    _download_upstream()

    if not models_ready():
        missing = [kind for kind in ("unet", "clip", "vae", "controlnet") if not model_file(kind).is_file()]
        raise FileNotFoundError(
            f"$controlnet models missing under {qwen_image_root()}: {', '.join(missing)}. "
            "Run: uv run python tools/download_models.py --upstream-fallback"
        )
    return qwen_image_root()
