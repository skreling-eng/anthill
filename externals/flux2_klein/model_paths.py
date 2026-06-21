"""Flux.2 Klein 9B FP8 assets under models/flux2klein/ + shared TE/VAE."""

from __future__ import annotations

import os
from pathlib import Path

from externals.anthill_models import require_models_file, upstream_fallback_enabled
from externals.image.model_paths import models_roots

MODEL_SUBDIR = Path("flux2klein")
UNET_FILENAME = "flux2Klein9bFp8_fp8.safetensors"
TEXT_ENCODER = "qwen_3_8b_fp8mixed.safetensors"
VAE_FILENAME = "flux2-vae.safetensors"

DEFAULT_MODEL = "klein-fp8"
MODEL_ALIASES: dict[str, str] = {
    "klein-fp8": UNET_FILENAME,
    "flux2_klein_fp8": UNET_FILENAME,
    "flux2klein9bfp8": UNET_FILENAME,
    "flux2Klein9bFp8_fp8": UNET_FILENAME,
}

HF_ASSET_URLS: dict[str, str] = {
    f"text_encoders/{TEXT_ENCODER}": (
        "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/"
        "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"
    ),
    f"vae/{VAE_FILENAME}": (
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/"
        "split_files/vae/flux2-vae.safetensors"
    ),
}


def available_models() -> str:
    return ", ".join(sorted(MODEL_ALIASES))


def _resolve_model_ref(model_arg: str) -> str:
    raw = (model_arg or DEFAULT_MODEL).strip()
    if not raw:
        raw = DEFAULT_MODEL
    return MODEL_ALIASES.get(raw, raw)


def is_klein_model(model_arg: str) -> bool:
    raw = (model_arg or "").strip()
    if not raw:
        return False
    if raw in MODEL_ALIASES:
        return True
    name = Path(raw).name.lower()
    return "klein" in name.lower() and name.endswith(".safetensors")


def klein_is_distilled(model_arg: str) -> bool:
    """True for flux-2-klein-*-fp8 (4-step); false for *base* checkpoints."""
    try:
        name = resolve_unet(model_arg).name.lower()
    except FileNotFoundError:
        name = _resolve_model_ref(model_arg).lower()
    return "base" not in name


def klein_recommended_steps_cfg(model_arg: str) -> tuple[int, float]:
    if klein_is_distilled(model_arg):
        return 4, 1.0
    return 20, 4.0


def normalize_klein_steps_cfg(
    model_arg: str,
    steps: int,
    cfg: float,
) -> tuple[int, float]:
    """Map user/base defaults onto distilled Klein settings unless explicitly overridden."""
    if not klein_is_distilled(model_arg):
        return steps, cfg
    if os.environ.get("AH_FLUX2_KLEIN_ALLOW_LONG_STEPS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return steps, cfg
    rec_steps, rec_cfg = klein_recommended_steps_cfg(model_arg)
    if steps <= 8 and cfg <= 2.0:
        return steps, cfg
    print(
        f"$flux2_klein: distilled checkpoint — using steps={rec_steps} cfg={rec_cfg} "
        f"(not steps={steps} cfg={cfg}; long/high cfg causes blur and artifacts). "
        f"Set AH_FLUX2_KLEIN_ALLOW_LONG_STEPS=1 to keep your values.",
        flush=True,
    )
    return rec_steps, rec_cfg


def _repo_unet_path(name: str) -> Path:
    for root in models_roots():
        candidate = root / MODEL_SUBDIR / name
        if candidate.is_file():
            return candidate.resolve()
    return (models_roots()[0] / MODEL_SUBDIR / name).resolve()


def resolve_unet(model_arg: str = "") -> Path:
    """Resolve model= to the Klein diffusion .safetensors path."""
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
        named = _repo_unet_path(raw + ".safetensors")
        if named.is_file():
            return named

    named = _repo_unet_path(Path(raw).name)
    if named.is_file():
        return named

    stem = Path(raw).stem if raw.endswith(".safetensors") else raw
    for rel in (
        f"flux2klein/{stem}.safetensors",
        f"flux2klein/{UNET_FILENAME}",
    ):
        try:
            return require_models_file(rel, label="$flux2_klein")
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Flux.2 Klein checkpoint not found: {model_arg or DEFAULT_MODEL!r}. "
        f"Expected under models/flux2klein/ ({available_models()}). "
        f"Run: uv run python tools/download_models.py --upstream-fallback"
    )


def companion_path(rel: str) -> Path:
    for root in models_roots():
        candidate = root / rel.replace("/", os.sep)
        if candidate.is_file():
            return candidate.resolve()
    return (models_roots()[0] / rel.replace("/", os.sep)).resolve()


def companion_ready() -> bool:
    return (
        companion_path(f"text_encoders/{TEXT_ENCODER}").is_file()
        and companion_path(f"vae/{VAE_FILENAME}").is_file()
    )


def ensure_companion_assets(*, force: bool = False) -> None:
    """Fetch Qwen3 TE + flux2 VAE when missing (anthill bundle or upstream HF)."""
    if companion_ready() and not force:
        return

    for rel in HF_ASSET_URLS:
        path = companion_path(rel)
        if path.is_file() and not force:
            continue
        try:
            require_models_file(rel, label="$flux2_klein")
            continue
        except FileNotFoundError:
            pass

        if not upstream_fallback_enabled():
            raise FileNotFoundError(
                f"Missing {rel} for Flux.2 Klein. "
                f"Place under models/{rel} or run: "
                f"uv run python tools/download_models.py --upstream-fallback"
            )

        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        url = HF_ASSET_URLS[rel]
        print(f"$flux2_klein: downloading {rel} from Hugging Face…", flush=True)
        urllib.request.urlretrieve(url, path)


def model_ready(model_arg: str = "") -> bool:
    try:
        resolve_unet(model_arg)
        return companion_ready()
    except FileNotFoundError:
        return False
