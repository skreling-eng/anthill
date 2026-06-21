"""Bootstrap comfy_lib for $controlnet (Qwen-Image + Union ControlNet)."""

from __future__ import annotations

from pathlib import Path

from externals.comfy_inprocess.bootstrap import bootstrap_comfy, comfy_lib_root, get_nodes_module
from externals.controlnet.model_paths import qwen_image_root
from externals.image.model_paths import models_roots

_CONTROLNET_EXTRAS = (
    "nodes_model_advanced.py",
)


def _register_qwen_model_paths() -> None:
    import folder_paths

    root = qwen_image_root()
    for sub, key in (
        ("diffusion_models", "diffusion_models"),
        ("text_encoders", "text_encoders"),
        ("vae", "vae"),
        ("controlnet", "controlnet"),
    ):
        path = root / sub
        if path.is_dir():
            folder_paths.add_model_folder_path(key, str(path.resolve()))

    for models_root in models_roots():
        qwen = models_root / "qwen-image"
        if not qwen.is_dir():
            continue
        for sub in ("diffusion_models", "text_encoders", "vae", "controlnet"):
            path = qwen / sub
            if path.is_dir():
                folder_paths.add_model_folder_path(sub, str(path.resolve()))


def _load_controlnet_extras() -> None:
    import logging

    from nodes import load_custom_node

    extras_dir = comfy_lib_root() / "comfy_extras"
    failed: list[str] = []
    for name in _CONTROLNET_EXTRAS:
        path = extras_dir / name
        if not path.is_file():
            failed.append(name)
            continue
        if not load_custom_node(str(path), module_parent="comfy_extras"):
            failed.append(name)
    if failed:
        logging.warning("$controlnet: optional comfy_extras not loaded: %s", ", ".join(failed))


def bootstrap_controlnet_comfy(
    *,
    input_dir: Path,
    output_dir: Path,
) -> None:
    from externals.comfy_inprocess.vram_config import apply_controlnet_vram_settings

    apply_controlnet_vram_settings()
    bootstrap_comfy(input_dir=input_dir, output_dir=output_dir, vram_profile="default")
    _register_qwen_model_paths()
    _load_controlnet_extras()


__all__ = ["bootstrap_controlnet_comfy", "get_nodes_module"]
