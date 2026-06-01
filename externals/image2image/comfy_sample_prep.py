"""GPU residency for Qwen image2image: CLIP/VAE for encode, UNet only for KSampler."""

from __future__ import annotations

import time
from typing import Any

def reset_unet_prepared() -> None:
    """Back-compat no-op; UNet is now refreshed before each sample."""
    return


def preload_for_encode(out: tuple[Any, ...]) -> None:
    """Load CLIP (+ VAE) for TextEncode; keep diffusion UNet off GPU until sample."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    _clip, vae = out[1], out[2]
    patchers: list[Any] = []
    for part in (_clip, vae):
        patcher = getattr(part, "patcher", None)
        if patcher is not None:
            patchers.append(patcher)
    if not patchers:
        return
    mm.load_models_gpu(patchers, force_full_load=False)


def prepare_for_ksampler(model: Any, *, reuse_if_same: bool = True) -> None:
    """Free CLIP/VAE VRAM, fully load UNet before KSampler (avoids 100s+/step CPU thrash)."""
    _ = reuse_if_same
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    t0 = time.perf_counter()
    # Always evict encoders / other patchers so the diffusion model gets full GPU.
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    # Reserve headroom for activations; <2GB free can make steps crawl due to paging.
    mm.load_models_gpu(
        [model],
        force_full_load=False,
        minimum_memory_required=3 * (1024**3),
    )
    device = getattr(model, "load_device", "?")
    weight_dev = "?"
    loaded = total = "?"
    try:
        weight_dev = next(model.model.parameters()).device
    except (StopIteration, AttributeError):
        pass
    try:
        loaded = getattr(model, "loaded_size", lambda: 0)()
        total = getattr(model, "model_size", lambda: 0)()
    except Exception:
        pass
    free_bytes = 0
    try:
        free_bytes = mm.get_free_memory(device)
        free_gb = f"{free_bytes / (1024**3):.1f}GB"
    except Exception:
        free_gb = "?"
    try:
        import comfy.model_management as mm_mod

        vram_mode = mm_mod.vram_state.name
    except Exception:
        vram_mode = "?"
    print(
        f"$image2image: UNet ready vram_state={vram_mode} load_device={device} "
        f"weights={weight_dev} loaded={loaded}/{total} vram_free={free_gb} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    if free_bytes and free_bytes < 2 * (1024**3):
        print(
            "$image2image: WARNING <2GB VRAM free after UNet load — sampling may crawl. "
            "Try lower resolution, set AH_IMAGE2IMAGE_SAMPLER=euler, or use AH_IMAGE2IMAGE_VRAM=low.",
            flush=True,
        )
