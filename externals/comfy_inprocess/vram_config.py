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


def apply_comfy_vram_settings() -> None:
    """Apply WAN_I2V_VRAM / AH_COMFY_VRAM to comfy cli args and model_management."""
    mode = _vram_mode_from_env()
    if not mode or mode in ("normal", "default", "off", "0", "false", "no"):
        return

    try:
        import comfy.cli_args as cli_args
    except ImportError:
        return

    if mode in ("low", "lowvram", "low_vram") or _truthy(mode):
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

        mm.lowvram_available = True
        if target == "NO_VRAM":
            state = VRAMState.NO_VRAM
        else:
            state = VRAMState.LOW_VRAM
        mm.set_vram_to = state
        mm.vram_state = state
    except ImportError:
        pass


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
