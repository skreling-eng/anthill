"""Load Qwen-Rapid-AIO Comfy checkpoint into diffusers modules."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from safetensors import safe_open

from externals.image2image.aio_keymap import (
    convert_comfy_vae_state_dict,
    remap_text_encoder_state,
)

_TRANSFORMER_PREFIX = "model.diffusion_model."
_TEXT_PREFIX = "text_encoders.qwen25_7b.transformer."
_VAE_PREFIX = "vae."


def count_aio_tensors(aio_path: Path) -> tuple[int, int, int]:
    """Count tensors by component without loading weights."""
    tr = te = vae = 0
    with safe_open(str(aio_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(_TRANSFORMER_PREFIX):
                tr += 1
            elif key.startswith(_TEXT_PREFIX):
                te += 1
            elif key.startswith(_VAE_PREFIX):
                vae += 1
    return tr, te, vae


def _load_state(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Bulk load into a meta module; return (loaded, skipped)."""
    if not state_dict:
        return 0, 0
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    skipped = len(result.unexpected_keys)
    if skipped:
        sample = ", ".join(result.unexpected_keys[:5])
        extra = f" (+{skipped - 5} more)" if skipped > 5 else ""
        print(
            f"$image2image: {type(module).__name__} skipped keys: {sample}{extra}",
            flush=True,
        )
    loaded = len(state_dict) - skipped
    return loaded, skipped


def materialize_meta_parameters(
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
    label: str = "",
) -> int:
    """Fill any still-meta parameters/buffers so .to() / offload can run."""
    from accelerate.utils import set_module_tensor_to_device

    filled = 0
    for name, param in module.named_parameters():
        if not param.is_meta:
            continue
        filled += 1
        set_module_tensor_to_device(
            module,
            name,
            "cpu",
            value=torch.zeros(param.shape, dtype=dtype, device="cpu"),
        )
    for name, buf in module.named_buffers():
        if not buf.is_meta:
            continue
        filled += 1
        set_module_tensor_to_device(
            module,
            name,
            "cpu",
            value=torch.zeros(buf.shape, dtype=dtype, device="cpu"),
        )
    if filled:
        who = label or type(module).__name__
        print(
            f"$image2image: zero-filled {filled} missing {who} tensors",
            flush=True,
        )
        if filled > 8:
            raise RuntimeError(
                f"$image2image: {who} missing {filled} tensors after AIO load — "
                "checkpoint key mapping may be wrong"
            )
    return filled


def finalize_module(
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    label: str = "",
) -> torch.nn.Module:
    materialize_meta_parameters(module, dtype=dtype, label=label)
    target = torch.device(device)
    params = [p for p in module.parameters() if not p.is_meta]
    if not params:
        return module.to(device=target, dtype=dtype)
    needs_device = any(p.device != target for p in params)
    needs_dtype = any(p.dtype != dtype for p in params)
    if needs_device or needs_dtype:
        return module.to(device=target, dtype=dtype)
    return module


def _read_aio_tensors(
    handle,
    *,
    load_device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Single pass over safetensors; tensors land on load_device (cpu or cuda)."""
    transformer: dict[str, torch.Tensor] = {}
    text_raw: dict[str, torch.Tensor] = {}
    vae_raw: dict[str, torch.Tensor] = {}
    last_log = time.monotonic()
    for index, key in enumerate(handle.keys()):
        tensor = handle.get_tensor(key)
        if load_device != "cpu":
            tensor = tensor.to(load_device)
        if key.startswith(_TRANSFORMER_PREFIX):
            name = key[len(_TRANSFORMER_PREFIX) :]
            if not name.startswith("__"):
                transformer[name] = tensor
        elif key.startswith(_TEXT_PREFIX):
            text_raw[key[len(_TEXT_PREFIX) :]] = tensor
        elif key.startswith(_VAE_PREFIX):
            vae_raw[key[len(_VAE_PREFIX) :]] = tensor
        now = time.monotonic()
        if index % 400 == 0 or now - last_log >= 8.0:
            print(
                f"$image2image:   reading checkpoint... {index + 1} keys",
                flush=True,
            )
            last_log = now
    return transformer, text_raw, vae_raw


def apply_aio_checkpoint(
    *,
    aio_path: Path,
    transformer,
    text_encoder,
    vae,
    load_device: str = "cpu",
) -> None:
    """Map Comfy AIO tensors into diffusers transformer / text encoder / VAE."""
    where = "GPU" if load_device != "cpu" else "CPU"
    print(
        f"$image2image: loading weights from {aio_path.name} -> {where} "
        f"(GPU memory rises after load completes)",
        flush=True,
    )
    t0 = time.perf_counter()
    with safe_open(str(aio_path), framework="pt", device="cpu") as handle:
        print("$image2image:   scanning checkpoint...", flush=True)
        tr_state, te_raw, vae_raw = _read_aio_tensors(handle, load_device=load_device)
        t_read = time.perf_counter() - t0

        t1 = time.perf_counter()
        tr_n, _ = _load_state(transformer, tr_state)
        del tr_state
        print(
            f"$image2image:   transformer {tr_n} tensors ({time.perf_counter() - t1:.1f}s)",
            flush=True,
        )

        t1 = time.perf_counter()
        te_n = 0
        if te_raw:
            te_state = remap_text_encoder_state(te_raw)
            del te_raw
            te_n, _ = _load_state(text_encoder, te_state)
            del te_state
        print(
            f"$image2image:   text_encoder {te_n} tensors ({time.perf_counter() - t1:.1f}s)",
            flush=True,
        )

        t1 = time.perf_counter()
        vae_state = convert_comfy_vae_state_dict(vae_raw)
        del vae_raw
        vae_n, _ = _load_state(vae, vae_state)
        del vae_state
        print(
            f"$image2image:   vae {vae_n} tensors ({time.perf_counter() - t1:.1f}s)",
            flush=True,
        )

    print(
        f"$image2image: loaded transformer={tr_n} text={te_n} vae={vae_n} tensors "
        f"in {time.perf_counter() - t0:.1f}s (read {t_read:.1f}s)",
        flush=True,
    )
    if tr_n == 0:
        raise RuntimeError(f"No transformer weights found in {aio_path}")
