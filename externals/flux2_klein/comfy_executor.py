"""Comfy executor hooks for Flux.2 Klein split loaders + ReferenceLatent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from externals.comfy_inprocess.executor import register_node_handler

_LOADER_CACHE: dict[str, tuple[Any, ...]] = {}


def _cache_loader(
    class_name: str,
    *,
    folder_key: str,
    name_key: str,
    load_fn,
    inputs: dict,
    label: str,
) -> tuple[Any, ...]:
    import folder_paths

    name = inputs[name_key]
    full = Path(folder_paths.get_full_path_or_raise(folder_key, name)).resolve()
    key = f"{class_name}:{full}"
    if key not in _LOADER_CACHE:
        print(f"{label}: loading {name} from disk…", flush=True)
        t0 = time.perf_counter()
        _LOADER_CACHE[key] = load_fn(**inputs)
        print(
            f"{label}: loaded {name} ({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    return _LOADER_CACHE[key]


def _unet_loader(inputs: dict) -> tuple[Any, ...]:
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS["UNETLoader"]

    def _load(**kwargs):
        return cls().load_unet(kwargs["unet_name"], kwargs.get("weight_dtype", "default"))

    return _cache_loader(
        "UNETLoader",
        folder_key="diffusion_models",
        name_key="unet_name",
        load_fn=_load,
        inputs=inputs,
        label="$flux2_klein",
    )


def _clip_loader(inputs: dict) -> tuple[Any, ...]:
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]

    def _load(**kwargs):
        device = kwargs.get("device", "default")
        if device == "cpu":
            print("$flux2_klein: Qwen TE on CPU (frees GPU for UNet sampling)", flush=True)
        return cls().load_clip(
            kwargs["clip_name"],
            kwargs.get("type", "flux2"),
            device,
        )

    return _cache_loader(
        "CLIPLoader",
        folder_key="text_encoders",
        name_key="clip_name",
        load_fn=_load,
        inputs=inputs,
        label="$flux2_klein",
    )


def _vae_loader(inputs: dict) -> tuple[Any, ...]:
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS["VAELoader"]

    def _load(**kwargs):
        return cls().load_vae(kwargs["vae_name"])

    return _cache_loader(
        "VAELoader",
        folder_key="vae",
        name_key="vae_name",
        load_fn=_load,
        inputs=inputs,
        label="$flux2_klein",
    )


def _reference_latent_handler(inputs: dict) -> tuple:
    import node_helpers

    conditioning = inputs["conditioning"]
    latent = inputs.get("latent")
    if latent is not None:
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [latent["samples"]]},
            append=True,
        )
    return (conditioning,)


def _empty_flux2_latent(inputs: dict) -> tuple:
    """Match Comfy EmptyFlux2LatentImage: 128ch latent, 16× pixel grid."""
    import comfy.model_management as mm
    import torch

    width = int(inputs["width"])
    height = int(inputs["height"])
    batch_size = int(inputs.get("batch_size", 1))
    latent = torch.zeros(
        [batch_size, 128, height // 16, width // 16],
        device=mm.intermediate_device(),
    )
    return ({"samples": latent},)


register_node_handler(
    "UNETLoader",
    _unet_loader,
    input_types={
        "required": {
            "unet_name": ("STRING",),
            "weight_dtype": ("STRING",),
        }
    },
)
register_node_handler(
    "CLIPLoader",
    _clip_loader,
    input_types={
        "required": {
            "clip_name": ("STRING",),
            "type": ("STRING",),
        },
        "optional": {"device": ("STRING",)},
    },
)
register_node_handler(
    "VAELoader",
    _vae_loader,
    input_types={"required": {"vae_name": ("STRING",)}},
)
register_node_handler(
    "ReferenceLatent",
    _reference_latent_handler,
    input_types={
        "required": {"conditioning": ("CONDITIONING",)},
        "optional": {"latent": ("LATENT",)},
    },
)
register_node_handler(
    "EmptyFlux2LatentImage",
    _empty_flux2_latent,
    input_types={
        "required": {
            "width": ("INT",),
            "height": ("INT",),
            "batch_size": ("INT",),
        }
    },
)
