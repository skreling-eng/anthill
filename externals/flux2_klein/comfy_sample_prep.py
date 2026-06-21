"""GPU residency for Flux.2 Klein: encoders for encode, UNet only for KSampler."""

from __future__ import annotations

import gc
import os
import time
from typing import Any

# Headroom Comfy needs for Flux edit activations at capped resolution (~576×1024).
_ACTIVATION_HEADROOM_BYTES = int(3 * (1024**3))


def gpu_total_gib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


def apply_flux2_klein_vram_settings() -> None:
    """Optional VRAM override via AH_FLUX2_KLEIN_VRAM (default: NORMAL_VRAM)."""
    raw = os.environ.get(
        "AH_FLUX2_KLEIN_VRAM",
        os.environ.get("AH_IMAGE2IMAGE_VRAM", ""),
    ).strip().lower()
    if not raw:
        return
    from externals.comfy_inprocess.vram_config import _apply_vram_mode

    _apply_vram_mode(raw)
    try:
        import comfy.model_management as mm

        print(
            f"$flux2_klein: vram profile {raw!r} -> {mm.vram_state.name}",
            flush=True,
        )
    except ImportError:
        pass


def _on_gpu(part: Any) -> bool:
    patcher = getattr(part, "patcher", part)
    device = getattr(patcher, "load_device", None)
    if device is None:
        return False
    try:
        import torch

        return torch.device(device).type != "cpu"
    except Exception:
        return str(device) != "cpu"


def _detach_loader_cache() -> None:
    try:
        from externals.flux2_klein import comfy_executor as ex

        for _key, val in ex._LOADER_CACHE.items():
            obj = val[0] if isinstance(val, tuple) and val else val
            patcher = getattr(obj, "patcher", obj)
            if patcher is not None and hasattr(patcher, "detach"):
                try:
                    patcher.detach(unpatch_all=False)
                except Exception:
                    pass
    except ImportError:
        pass


def release_klein_gpu_cache(*, label: str = "$flux2_klein") -> None:
    """Detach all cached loaders and empty CUDA before UNet sampling."""
    _detach_loader_cache()

    try:
        import comfy.model_management as mm
        import torch
    except ImportError:
        return

    mm.unload_all_models()
    mm.cleanup_models_gc()
    mm.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"{label}: GPU after evict allocated={alloc:.1f}GiB reserved={reserved:.1f}GiB",
            flush=True,
        )


def park_unet_off_gpu(model: Any) -> None:
    """Keep UNet off GPU during VAE/CLIP encode (split-loader workflows)."""
    if model is None:
        return
    patcher = getattr(model, "patcher", model)
    if hasattr(patcher, "detach"):
        try:
            patcher.detach(unpatch_all=False)
        except Exception:
            pass
    release_klein_gpu_cache()


def prepare_for_klein_encode(*, clip: Any = None, vae: Any = None) -> None:
    """Unload diffusion UNet; keep only GPU-resident CLIP/VAE for the encode node."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    patchers: list[Any] = []
    for part in (clip, vae):
        if part is None or not _on_gpu(part):
            continue
        patcher = getattr(part, "patcher", None)
        if patcher is not None:
            patchers.append(patcher)
    mm.unload_all_models()
    mm.soft_empty_cache(force=True)
    gc.collect()
    if not patchers:
        return
    mm.load_models_gpu(patchers, force_full_load=False)


def _klein_force_full_unet() -> bool:
    raw = os.environ.get("AH_FLUX2_KLEIN_FULL_UNET", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # Full 9B UNet + edit activations does not fit on 16–17 GiB cards.
    return gpu_total_gib() > 20


def _load_unet_patcher(
    patcher: Any,
    *,
    force_full: bool,
    min_mem_bytes: int,
) -> tuple[int, int, str, int]:
    import comfy.model_management as mm

    mm.load_models_gpu(
        [patcher],
        force_full_load=force_full,
        minimum_memory_required=min_mem_bytes,
    )
    device = getattr(patcher, "load_device", "?")
    weight_dev = "?"
    loaded = total = 0
    try:
        weight_dev = next(patcher.model.parameters()).device
    except (StopIteration, AttributeError):
        pass
    try:
        loaded = patcher.loaded_size()
        total = patcher.model_size()
    except Exception:
        pass
    free_bytes = 0
    try:
        free_bytes = mm.get_free_memory(device)
    except Exception:
        pass
    return loaded, total, str(weight_dev), free_bytes


def offload_klein_conditioning(
    conditioning: Any,
    *,
    keep_reference_latents: bool = False,
) -> Any:
    """Move conditioning payloads to CPU so UNet load has room for activations."""
    try:
        import torch
    except ImportError:
        return conditioning
    if not isinstance(conditioning, list):
        return conditioning
    out: list[Any] = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            out.append(item)
            continue
        cond, meta = item[0], item[1]
        if isinstance(cond, dict):
            cond = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in cond.items()
            }
        elif torch.is_tensor(cond):
            cond = cond.detach().cpu()
        if isinstance(meta, dict):
            meta = dict(meta)
            refs = meta.get("reference_latents")
            if isinstance(refs, list) and not keep_reference_latents:
                meta["reference_latents"] = [
                    t.detach().cpu() if torch.is_tensor(t) else t for t in refs
                ]
        out.append([cond, meta])
    return out


def offload_klein_latent(latent: Any) -> Any:
    """Move latent sample tensors to CPU when still referenced in the graph."""
    try:
        import torch
    except ImportError:
        return latent
    if not isinstance(latent, dict):
        return latent
    samples = latent.get("samples")
    if torch.is_tensor(samples):
        return {**latent, "samples": samples.detach().cpu()}
    return latent


def offload_klein_image_tensor(image_out: Any) -> Any:
    """Move LoadImage outputs to CPU."""
    try:
        import torch
    except ImportError:
        return image_out
    if not isinstance(image_out, tuple):
        return image_out
    moved: list[Any] = []
    for part in image_out:
        if torch.is_tensor(part):
            moved.append(part.detach().cpu())
        else:
            moved.append(part)
    return tuple(moved)


def prepare_for_ksampler(
    model: Any,
    *,
    label: str = "$flux2_klein",
    prefer_full_unet: bool = False,
) -> None:
    """Evict encoders; load UNet with activation headroom (Qwen-style partial load)."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    t0 = time.perf_counter()
    release_klein_gpu_cache(label=label)

    patcher = getattr(model, "patcher", model)
    try:
        patcher.detach(unpatch_all=False)
    except Exception:
        pass

    full_load = _klein_force_full_unet() or prefer_full_unet
    min_mem = _ACTIVATION_HEADROOM_BYTES
    loaded, total, weight_dev, free_bytes = _load_unet_patcher(
        patcher,
        force_full=full_load,
        min_mem_bytes=min_mem,
    )

    if loaded == 0 and not full_load:
        print(
            f"{label}: partial UNet load got 0 bytes — retrying with full weights",
            flush=True,
        )
        release_klein_gpu_cache(label=label)
        loaded, total, weight_dev, free_bytes = _load_unet_patcher(
            patcher,
            force_full=True,
            min_mem_bytes=min_mem,
        )

    free_gb = f"{free_bytes / (1024**3):.1f}GB" if free_bytes else "?"
    try:
        vram_mode = mm.vram_state.name
    except Exception:
        vram_mode = "?"
    print(
        f"{label}: UNet ready vram_state={vram_mode} load_device="
        f"{getattr(patcher, 'load_device', '?')} weights={weight_dev} "
        f"loaded={loaded}/{total} vram_free={free_gb} "
        f"({'full' if full_load else 'partial'}) ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    if loaded == 0 or weight_dev == "cpu":
        print(
            f"{label}: ERROR UNet not on GPU — close other GPU apps, use "
            f"576x1024 or lower (AH_FLUX2_KLEIN_MAX_AREA), or set "
            f"AH_FLUX2_KLEIN_VRAM=low.",
            flush=True,
        )
    elif free_bytes and free_bytes < min_mem:
        print(
            f"{label}: WARNING <3GB VRAM free after UNet load — sampling may crawl. "
            f"Try smaller output (AH_FLUX2_KLEIN_MAX_AREA) or fewer steps.",
            flush=True,
        )
