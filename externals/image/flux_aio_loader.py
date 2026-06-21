"""Load ComfyUI Flux Fusion AIO (.safetensors with model.diffusion_model.*) for $image flux_ext."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from bitsandbytes.functional import QuantState, dequantize_4bit
from bitsandbytes.nn import Params4bit
from diffusers import FluxTransformer2DModel
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
from safetensors import safe_open

from externals.image.model_paths import FLUX_CKPT_4BIT_ID, load_pretrained_sub, subfolder_path

_TRANSFORMER_PREFIX = "model.diffusion_model."
_BNB_SUFFIX = ".quant_state.bitsandbytes__nf4"


def is_comfy_flux_aio(path: str | Path) -> bool:
    """True when the file is a Comfy-style Flux AIO (not a plain transformer .safetensors)."""
    path = Path(path)
    if not path.is_file():
        return False
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(_TRANSFORMER_PREFIX + "double_blocks."):
                return True
    return False


def _dequantize_bnb_weight(handle, weight_key: str) -> torch.Tensor:
    fd = {
        "quant_state.bitsandbytes__nf4": handle.get_tensor(weight_key + _BNB_SUFFIX),
        "absmax": handle.get_tensor(weight_key + ".absmax"),
        "quant_map": handle.get_tensor(weight_key + ".quant_map"),
    }
    nested_absmax = weight_key + ".nested_absmax"
    if nested_absmax in handle.keys():
        fd["nested_absmax"] = handle.get_tensor(nested_absmax)
        fd["nested_quant_map"] = handle.get_tensor(weight_key + ".nested_quant_map")
    qs = QuantState.from_dict(fd, torch.device("cpu"))
    packed = handle.get_tensor(weight_key)
    return dequantize_4bit(packed, quant_state=qs).to(torch.float16)


def _dequantize_comfy_transformer(path: Path) -> dict[str, torch.Tensor]:
    """Comfy diffusion_model.* tensors -> fp16 state dict (Comfy key names)."""
    out: dict[str, torch.Tensor] = {}
    t0 = time.perf_counter()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = [
            k
            for k in handle.keys()
            if k.startswith(_TRANSFORMER_PREFIX) and not k[len(_TRANSFORMER_PREFIX) :].startswith("__")
        ]
        bnb_roots = sorted(
            {
                k[: -len(_BNB_SUFFIX)]
                for k in keys
                if k.endswith(_BNB_SUFFIX)
            }
        )
        print(
            f"$image: dequantizing {len(bnb_roots)} Flux AIO weight groups...",
            flush=True,
        )
        for index, weight_key in enumerate(bnb_roots):
            rel = weight_key[len(_TRANSFORMER_PREFIX) :]
            out[rel] = _dequantize_bnb_weight(handle, weight_key)
            bias_key = weight_key.replace(".weight", ".bias")
            if bias_key in keys:
                out[rel.replace(".weight", ".bias")] = handle.get_tensor(bias_key).to(torch.float16)
            if index and index % 50 == 0:
                print(f"$image:   dequant {index}/{len(bnb_roots)}", flush=True)

        for key in keys:
            rel = key[len(_TRANSFORMER_PREFIX) :]
            if rel in out or rel.endswith(_BNB_SUFFIX) or any(
                rel.endswith(s)
                for s in (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map")
            ):
                continue
            if rel.endswith(".scale"):
                out[rel] = handle.get_tensor(key).to(torch.float16)
                continue
            if rel.endswith(".bias") and rel not in out:
                out[rel] = handle.get_tensor(key).to(torch.float16)
    print(
        f"$image: dequantized {len(out)} tensors in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _apply_fp16_to_nf4_model(
    model: FluxTransformer2DModel,
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> None:
    """Quantize fp16 weights into an NF4 FluxTransformer2DModel skeleton."""
    model_keys = set(dict(model.named_parameters()).keys())
    t0 = time.perf_counter()
    applied = 0
    for name, tensor in state_dict.items():
        if name not in model_keys:
            continue
        module_name, param_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        param = getattr(module, param_name)
        if isinstance(param, Params4bit):
            quant = Params4bit(
                tensor.to(device),
                requires_grad=False,
                quant_type="nf4",
            ).to(device)
            setattr(module, param_name, quant)
        else:
            param.data.copy_(tensor.to(device))
        applied += 1
        if applied % 100 == 0:
            print(f"$image:   quantize/load {applied}/{len(state_dict)}", flush=True)
    print(
        f"$image: applied {applied} tensors to NF4 transformer "
        f"in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


def load_flux_aio_transformer(path: str | Path) -> FluxTransformer2DModel:
    """
    Load a Comfy Flux Fusion AIO into a diffusers NF4 FluxTransformer2DModel.

    The file bundles diffusion_model + text_encoders + vae; only the transformer
    is loaded here (encoders/VAE still come from FLUX.1-dev / flux.1-dev-nf4-pkg).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Flux AIO checkpoint not found: {path}")

    comfy_fp16 = _dequantize_comfy_transformer(path)
    converted = convert_flux_transformer_checkpoint_to_diffusers(comfy_fp16)
    if comfy_fp16:
        leftover = ", ".join(sorted(comfy_fp16.keys())[:8])
        raise RuntimeError(
            f"Flux AIO conversion left {len(comfy_fp16)} unmapped keys (e.g. {leftover})"
        )

    print(
        f"$image: loading NF4 transformer skeleton from {FLUX_CKPT_4BIT_ID}",
        flush=True,
    )
    model = load_pretrained_sub(FluxTransformer2DModel, FLUX_CKPT_4BIT_ID, "transformer")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(device)

    _apply_fp16_to_nf4_model(model, converted, device=device)
    return model
