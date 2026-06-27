"""Park Wan DiT weights off GPU before WanVAE encode (defer-load path)."""

from __future__ import annotations

import gc
from typing import Any

import torch


def gpu_allocated_mib() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
    except Exception:
        pass
    return 0.0


def _move_tensor_map_to_cpu(data: dict[str, Any]) -> dict[str, Any]:
    cpu = torch.device("cpu")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.device.type not in ("cpu", "meta"):
            out[key] = value.to(cpu, non_blocking=False)
        else:
            out[key] = value
    return out


def _model_pipeline(model_data: Any) -> dict[str, Any] | None:
    if isinstance(model_data, dict):
        return model_data
    pipeline = getattr(model_data, "pipeline", None)
    if isinstance(pipeline, dict):
        return pipeline
    return None


def _model_diffusion(model_data: Any) -> Any:
    if isinstance(model_data, dict):
        return model_data.get("diffusion_model")
    return getattr(model_data, "diffusion_model", None)


def _module_has_meta_params(module: Any) -> bool:
    try:
        for param in module.parameters():
            if param.device.type == "meta":
                return True
    except Exception:
        pass
    return False


def _move_module_to_cpu_if_materialized(module: Any) -> None:
    """Move a module to CPU only when it holds real weights (not meta placeholders)."""
    if module is None or not hasattr(module, "to"):
        return
    if getattr(module, "patched_linear", False):
        return
    if _module_has_meta_params(module):
        return
    module.to(torch.device("cpu"))


def park_dit_state_dict_to_cpu(patcher: Any) -> None:
    """Move Wan weight tensors in the patcher state dict to CPU (safe for defer-load)."""
    if patcher is None or not getattr(patcher, "model", None):
        return
    pipeline = _model_pipeline(patcher.model)
    if pipeline is None:
        return
    sd = pipeline.get("sd")
    if isinstance(sd, dict):
        pipeline["sd"] = _move_tensor_map_to_cpu(sd)
    scale_weights = pipeline.get("scale_weights")
    if isinstance(scale_weights, dict):
        pipeline["scale_weights"] = _move_tensor_map_to_cpu(scale_weights)


def park_dit_for_vae_encode(patcher: Any, transformer: Any) -> tuple[float, float]:
    """Free GPU for WanVAE encode; return (before, after) MiB allocated."""
    import comfy.model_management as mm

    before = gpu_allocated_mib()

    park_dit_state_dict_to_cpu(patcher)

    # FP8 scaled models use patched_linear + meta module shells until load_weights.
    # Weights live in patcher state dict — never call .to(cpu) on meta modules.
    diffusion_model = None
    if patcher is not None and getattr(patcher, "model", None):
        diffusion_model = _model_diffusion(patcher.model)
    if diffusion_model is not None and diffusion_model is not transformer:
        _move_module_to_cpu_if_materialized(diffusion_model)
    _move_module_to_cpu_if_materialized(transformer)

    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()

    return before, gpu_allocated_mib()


def summarize_transformer_devices(transformer: Any) -> dict[str, Any]:
    """Count parameter tensors by device and estimate on-GPU weight bytes."""
    counts: dict[str, int] = {"cuda": 0, "cpu": 0, "meta": 0, "other": 0}
    cuda_bytes = 0
    total = 0
    for param in transformer.parameters():
        total += 1
        dev = param.device.type
        if dev in counts:
            counts[dev] += 1
        else:
            counts["other"] += 1
        if dev == "cuda" and param.data is not None:
            cuda_bytes += param.numel() * param.element_size()
    return {
        "total_params": total,
        "counts": counts,
        "cuda_mib": cuda_bytes / (1024**2),
    }


def summarize_state_dict_devices(patcher: Any) -> dict[str, int]:
    """Count weight tensors in patcher sd by device (duplicate copy if still resident)."""
    counts: dict[str, int] = {"cuda": 0, "cpu": 0, "meta": 0, "other": 0}
    pipeline = _model_pipeline(getattr(patcher, "model", None)) if patcher else None
    if pipeline is None:
        return counts
    sd = pipeline.get("sd")
    if not isinstance(sd, dict):
        return counts
    for value in sd.values():
        if not isinstance(value, torch.Tensor):
            continue
        dev = value.device.type
        if dev in counts:
            counts[dev] += 1
        else:
            counts["other"] += 1
    return counts


def log_sampling_vram_readiness(
    transformer: Any,
    patcher: Any | None = None,
) -> None:
    """Log whether DiT weights look fully materialized on GPU before denoising."""
    summary = summarize_transformer_devices(transformer)
    counts = summary["counts"]
    cuda_n = counts["cuda"]
    meta_n = counts["meta"]
    cpu_n = counts["cpu"]
    total = summary["total_params"]
    alloc_mib = gpu_allocated_mib()
    sd_counts = summarize_state_dict_devices(patcher)

    materialized = cuda_n + cpu_n
    ready = meta_n == 0 and cuda_n > 0 and materialized >= max(1, total - 2)
    if meta_n == 0 and cuda_n > 0 and cpu_n > 0:
        status = "partial"
    elif ready:
        status = "ready"
    else:
        status = "INCOMPLETE"
    print(
        f"$avatar: sampling VRAM {status}: "
        f"params cuda={cuda_n} cpu={cpu_n} meta={meta_n} "
        f"({summary['cuda_mib']:.0f} MiB weights on GPU, "
        f"{alloc_mib:.0f} MiB total CUDA allocated)",
        flush=True,
    )
    if sd_counts.get("cuda", 0) or sd_counts.get("cpu", 0):
        print(
            f"$avatar: patcher sd tensors cuda={sd_counts.get('cuda', 0)} "
            f"cpu={sd_counts.get('cpu', 0)} "
            f"(CPU copy is normal after park; forward uses module weights)",
            flush=True,
        )
    if meta_n:
        print(
            "$avatar: WARNING: meta parameters remain — load_weights did not "
            "materialize all layers; sampling will be broken or extremely slow",
            flush=True,
        )
    elif cpu_n and cuda_n == 0:
        print(
            "$avatar: WARNING: all parameters on CPU — expect CPU inference or "
            "per-layer GPU uploads during sampling",
            flush=True,
        )
