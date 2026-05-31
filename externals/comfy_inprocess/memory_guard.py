"""VRAM-aware caps for Wan I2V (comfy_lib worker)."""

from __future__ import annotations

import os

_LATENT_ALIGN = 16
# Wan temporal VAE stride used in WanVaceToVideo (see comfy_nodes latent_length).
_TEMPORAL_STRIDE = 4


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def auto_cap_enabled() -> bool:
    """Opt-in only: set WAN_I2V_AUTO_CAP=1 to shrink resolution/frames on tight GPUs."""
    raw = os.environ.get("WAN_I2V_AUTO_CAP", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def frame_cap_disabled() -> bool:
    return _truthy(os.environ.get("WAN_I2V_NO_FRAME_CAP", ""))


def total_vram_mb() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception:
        pass
    return None


def estimate_wan_latent_tokens(*, width: int, height: int, num_frames: int) -> int:
    latent_length = ((num_frames - 1) // _TEMPORAL_STRIDE) + 1
    spatial = max(1, width // 8) * max(1, height // 8)
    return latent_length * spatial


def _default_max_area(vram_mb: int | None, *, mega: bool = False) -> int | None:
    raw = os.environ.get("WAN_I2V_MAX_AREA", "").strip()
    if raw:
        val = int(raw)
        return val if val > 0 else None
    if not auto_cap_enabled() or vram_mb is None:
        return None
    if vram_mb <= 18432:
        if mega:
            return 307_200  # ~480×640 — MEGA VACE needs tighter cap than I2V rapid
        return 399360  # ~480×832
    return None


def _fit_max_area(out_w: int, out_h: int, max_area: int) -> tuple[int, int]:
    from externals.image2video.comfy_workflow import snap_latent_size

    if out_w * out_h <= max_area:
        return out_w, out_h
    scale = (max_area / (out_w * out_h)) ** 0.5
    return snap_latent_size(round(out_w * scale), round(out_h * scale), multiple=_LATENT_ALIGN)


def cap_frames_for_vram(
    *,
    width: int,
    height: int,
    num_frames: int,
    vram_mb: int | None = None,
    mega: bool = False,
) -> int:
    if frame_cap_disabled():
        return num_frames
    vram_mb = vram_mb if vram_mb is not None else total_vram_mb()
    if vram_mb is None or vram_mb > 18432:
        return num_frames

    if mega and vram_mb <= 18432:
        max_frames, max_tokens = 17, 22_000
    elif vram_mb <= 12288:
        max_frames, max_tokens = 25, 40_000
    else:
        max_frames, max_tokens = 33, 52_000

    capped = min(num_frames, max_frames)
    tokens = estimate_wan_latent_tokens(width=width, height=height, num_frames=capped)
    if tokens > max_tokens:
        spatial = max(1, width // 8) * max(1, height // 8)
        max_latent_len = max(1, max_tokens // spatial)
        capped = min(capped, (max_latent_len - 1) * _TEMPORAL_STRIDE + 1)
        capped = max(capped, 5)

    if capped < num_frames:
        print(
            f"$image2video: capping frames {num_frames} -> {capped} "
            f"for ~{vram_mb}MB VRAM (WAN_I2V_NO_FRAME_CAP=1 to disable)",
            flush=True,
        )
    return capped


def configure_mega_runtime_defaults(model_name: str, args: dict | None = None) -> None:
    """MEGA on ≤18GB: novram, no node RAM cache, unless user overrides."""
    from externals.image2video.model_paths import is_mega_model

    if not is_mega_model(model_name):
        return
    vram_mb = total_vram_mb()
    if vram_mb is None or vram_mb > 18432:
        return

    arg_vram = str((args or {}).get("vram", "")).strip().lower()
    env_vram = os.environ.get("WAN_I2V_VRAM", "").strip().lower()
    if arg_vram not in ("normal", "default", "off", "0", "false", "no") and env_vram not in (
        "normal",
        "default",
        "off",
        "0",
        "false",
        "no",
    ):
        pass
    elif env_vram not in ("novram", "no_vram", "minimal", "min"):
        os.environ["WAN_I2V_VRAM"] = "novram"
        print(
            "$image2video: MEGA on ~16GB GPU → WAN_I2V_VRAM=novram",
            flush=True,
        )

    if not os.environ.get("AH_COMFY_CACHE", "").strip():
        os.environ["AH_COMFY_CACHE"] = "none"
        print(
            "$image2video: MEGA on ~16GB GPU → AH_COMFY_CACHE=none (less RAM held between nodes)",
            flush=True,
        )


def configure_mega_vram_defaults(model_name: str, args: dict | None = None) -> None:
    """Alias for configure_mega_runtime_defaults."""
    configure_mega_runtime_defaults(model_name, args)


def apply_wan_memory_limits(
    *,
    width: int,
    height: int,
    num_frames: int,
    mega: bool = False,
) -> tuple[int, int, int]:
    """Return (width, height, frames) after optional area/frame caps for tight GPUs."""
    vram_mb = total_vram_mb()
    if not auto_cap_enabled():
        return width, height, num_frames

    max_area = _default_max_area(vram_mb, mega=mega)
    out_w, out_h = width, height
    if max_area is not None and out_w * out_h > max_area:
        new_w, new_h = _fit_max_area(out_w, out_h, max_area)
        if (new_w, new_h) != (out_w, out_h):
            print(
                f"$image2video: scaling {out_w}x{out_h} -> {new_w}x{new_h} "
                f"(~{vram_mb or '?'}MB VRAM; set WAN_I2V_MAX_AREA to override)",
                flush=True,
            )
            out_w, out_h = new_w, new_h

    out_frames = cap_frames_for_vram(
        width=out_w,
        height=out_h,
        num_frames=num_frames,
        vram_mb=vram_mb,
        mega=mega,
    )
    return out_w, out_h, out_frames
