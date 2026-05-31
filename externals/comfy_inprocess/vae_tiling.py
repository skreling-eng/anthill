"""Opt-in tiled VAE encode/decode for comfy_lib (lower VRAM, slower)."""

from __future__ import annotations

import os


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def force_tiled_vae() -> bool:
    """True when tiled VAE should run before the regular path (no OOM retry needed)."""
    raw = os.environ.get("WAN_I2V_TILED_VAE", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if _truthy(raw):
        return True
    return _truthy(os.environ.get("AH_COMFY_TILED_VAE", ""))


def tiled_vae_force_full_load() -> bool:
    """Opt-in: load full VAE weights before tiled encode (ComfyUI default does not)."""
    if not force_tiled_vae():
        return False
    for key in ("WAN_I2V_TILED_VAE_FULL", "AH_COMFY_TILED_VAE_FULL"):
        raw = os.environ.get(key, "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if _truthy(raw):
            return True
    return False


def configure_tiled_vae_for_job(args: dict) -> None:
    """Apply ``$image2video(..., tiled_vae=1)`` for the current op (worker subprocess)."""
    for key in ("tiled_vae", "vae_tiles", "tiled"):
        if key not in args:
            continue
        raw = str(args[key]).strip().lower()
        if raw in ("0", "false", "no", "off"):
            os.environ["WAN_I2V_TILED_VAE"] = "0"
        elif _truthy(raw):
            os.environ["WAN_I2V_TILED_VAE"] = "1"
        return
