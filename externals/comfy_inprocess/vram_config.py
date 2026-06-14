"""Configure comfy_lib VRAM mode before model_management initializes."""

from __future__ import annotations

import os


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _vram_mode_from_env() -> str | None:
    raw = os.environ.get("WAN_I2V_VRAM", "").strip().lower()
    if raw:
        return raw
    raw = os.environ.get("AH_COMFY_VRAM", "").strip().lower()
    return raw or None


def _apply_vram_mode(mode: str) -> None:
    if not mode or mode in ("normal", "default", "off", "0", "false", "no"):
        return

    try:
        import comfy.cli_args as cli_args
    except ImportError:
        return

    if mode in ("high", "highvram", "gpu", "gpu_only"):
        cli_args.args.highvram = True
        cli_args.args.lowvram = False
        cli_args.args.novram = False
        target = "HIGH_VRAM"
    elif mode in ("low", "lowvram", "low_vram") or _truthy(mode):
        cli_args.args.lowvram = True
        cli_args.args.novram = False
        cli_args.args.highvram = False
        target = "LOW_VRAM"
    elif mode in ("novram", "no_vram", "minimal", "min"):
        cli_args.args.novram = True
        cli_args.args.lowvram = False
        cli_args.args.highvram = False
        target = "NO_VRAM"
    else:
        return

    try:
        import comfy.model_management as mm
        from comfy.model_management import VRAMState

        if target == "HIGH_VRAM":
            mm.vram_state = VRAMState.HIGH_VRAM
            return
        mm.lowvram_available = True
        if target == "NO_VRAM":
            state = VRAMState.NO_VRAM
        else:
            state = VRAMState.LOW_VRAM
        mm.set_vram_to = state
        mm.vram_state = state
    except ImportError:
        pass


def apply_comfy_vram_settings() -> None:
    """Apply WAN_I2V_VRAM / AH_COMFY_VRAM to comfy cli args and model_management."""
    _apply_vram_mode(_vram_mode_from_env() or "")


def _default_image2image_vram() -> str:
    """Default VRAM mode for ~19GB FP8 UNet + activations.

    ``high`` only on large GPUs (32GB+). On 20–24GB cards, ``normal`` leaves headroom
    for KSampler; forcing full UNet load leaves <1GB free and steps take minutes.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gb >= 32:
                return "high"
    except Exception:
        pass
    return "normal"


def apply_image2image_vram_settings() -> None:
    """Qwen edit VRAM profile — do not inherit WAN_I2V_VRAM from $image2video."""
    raw = os.environ.get("AH_IMAGE2IMAGE_VRAM", os.environ.get("AH_COMFY_VRAM", "")).strip().lower()
    if not raw:
        raw = _default_image2image_vram()
    _apply_vram_mode(raw)


def _default_controlnet_vram() -> str:
    raw = os.environ.get("AH_CONTROLNET_VRAM", "").strip().lower()
    if raw:
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gb <= 19:
                return "low"
    except Exception:
        pass
    return "low"


def apply_controlnet_vram_settings() -> None:
    """Qwen-Image + Union ControlNet — low VRAM by default on ≤19GB GPUs."""
    _apply_vram_mode(_default_controlnet_vram())


def configure_comfy_vram_for_job(args: dict) -> None:
    """Map ``$image2video(..., vram='low')`` to env for this worker job."""
    if "vram" not in args and "lowvram" not in args:
        return
    raw = str(args.get("vram", args.get("lowvram", ""))).strip().lower()
    if raw in ("0", "false", "no", "off", "normal", "default"):
        os.environ["WAN_I2V_VRAM"] = "normal"
    elif raw in ("low", "lowvram", "low_vram") or _truthy(raw):
        os.environ["WAN_I2V_VRAM"] = "low"
    elif raw in ("novram", "no_vram", "minimal", "min"):
        os.environ["WAN_I2V_VRAM"] = "novram"
