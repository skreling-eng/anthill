"""Model defaults for $avatar (SkyReels V3 A2V via ComfyUI-WanVideoWrapper)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WORKFLOW = "SkyReels-V3-Talking-Avatars_api.json"

DEFAULT_WAN_MODEL = os.environ.get(
    "AVATAR_WAN_MODEL",
    "Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors",
)
DEFAULT_VAE = os.environ.get("AVATAR_VAE", "Wan2_1_VAE_bf16.safetensors")
DEFAULT_TEXT_ENCODER = os.environ.get(
    "AVATAR_TEXT_ENCODER", "umt5-xxl-enc-bf16.safetensors"
)
DEFAULT_WAV2VEC = os.environ.get(
    "AVATAR_WAV2VEC", "TencentGameMate/chinese-wav2vec2-base"
)


def sage_attention_ready() -> tuple[bool, str]:
    """Whether WanVideoWrapper can use sageattn (import + optional quick CUDA probe)."""
    try:
        from sageattention import sageattn
    except Exception as exc:
        msg = str(exc)
        if "triton" in msg.lower():
            msg += " — run tools\\setup_sage_windows.ps1 on Windows"
        return False, msg
    try:
        import torch

        if torch.cuda.is_available():
            q = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
            sageattn(q, k, v, tensor_layout="HND")
    except Exception as exc:
        return False, f"CUDA probe failed: {exc}"
    ver = "unknown"
    try:
        import sageattention

        ver = getattr(sageattention, "__version__", ver)
    except Exception:
        pass
    return True, f"sageattention {ver}"


def resolve_attention_mode() -> tuple[str, str]:
    """Return (mode, detail) for workflow attention_mode and startup logging."""
    raw = os.environ.get("AVATAR_ATTENTION_MODE", "").strip()
    if raw:
        return raw, f"AVATAR_ATTENTION_MODE={raw}"
    ok, detail = sage_attention_ready()
    if ok:
        return "sageattn", detail
    return "sdpa", f"sage unavailable ({detail}); using sdpa"


def default_attention_mode() -> str:
    """sdpa works everywhere; sageattn when sageattention+triton work on this GPU."""
    return resolve_attention_mode()[0]

def _gpu_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return None


def resolve_blocks_to_swap(override: int | None = None) -> int:
    """Block swap trades speed for VRAM.

    SkyReels FP8 DiT materializes ~18GB on GPU. On 16GB cards (4080, etc.)
    blocks_to_swap=0 overflows VRAM and sampling crawls. ComfyUI graph uses 30.
    """
    if override is not None:
        return max(0, override)
    raw = os.environ.get("AVATAR_BLOCKS_TO_SWAP", "").strip()
    if raw:
        return max(0, int(raw))
    gb = _gpu_vram_gb()
    if gb is None:
        return 30
    if gb >= 24:
        return 0
    if gb >= 20:
        return 12
    return 30


def resolve_wan_load_device(blocks_to_swap: int) -> str:
    raw = os.environ.get("AVATAR_LOAD_DEVICE", "").strip()
    if raw in ("main_device", "offload_device"):
        return raw
    # ComfyUI SkyReels default — weights stay on CPU until sampler load_weights.
    # main_device preloads ~10GB FP8 DiT and makes WanVAE encode crawl on 16GB cards.
    return "offload_device"


def resolve_force_offload(blocks_to_swap: int) -> bool:
    """Match ComfyUI WanVideoSamplerv2 / SkyReels defaults (force_offload=True)."""
    raw = os.environ.get("AVATAR_FORCE_OFFLOAD", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # ComfyUI default; frees GPU for WanVAE encode (transformer reloads before sampling).
    return True


def resolve_defer_transformer_load(blocks_to_swap: int, force_offload: bool) -> bool:
    """Defer DiT ``load_weights`` until after WanVAE encode when VRAM is tight.

    Loading ~10GB FP8 DiT before WanVAE encode on a 16GB card makes VAE crawl
    (minutes). Only skip defer on GPUs with enough headroom for DiT + VAE together.
    """
    if not force_offload:
        return False
    raw = os.environ.get("AVATAR_DEFER_DIT_LOAD", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    gb = _gpu_vram_gb()
    if gb is not None and gb >= 24:
        return False
    return True


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_tiled_vae(override: bool | None = None) -> bool:
    """Full-GPU VAE encode is much faster on 16GB; tiling is for OOM only."""
    if override is not None:
        return override
    raw = os.environ.get("AVATAR_TILED_VAE", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False


def configure_avatar_tiled_vae_for_job(args: dict) -> bool | None:
    """Apply ``$avatar(..., tiled_vae=1)``; ignore WAN_I2V_TILED_VAE from MEGA $image2video."""
    override: bool | None = None
    for key in ("tiled_vae", "vae_tiles", "tiled"):
        if key not in args:
            continue
        raw = str(args[key]).strip().lower()
        if raw in ("0", "false", "no", "off"):
            override = False
            os.environ["AVATAR_TILED_VAE"] = "0"
        elif _truthy(raw):
            override = True
            os.environ["AVATAR_TILED_VAE"] = "1"
        break
    use_tiled = resolve_tiled_vae(override)
    if use_tiled:
        os.environ["WAN_I2V_TILED_VAE"] = "1"
    else:
        os.environ["WAN_I2V_TILED_VAE"] = "0"
        os.environ.pop("AH_COMFY_TILED_VAE", None)
    return override


def _default_max_area() -> int | None:
    raw = os.environ.get("AVATAR_MAX_AREA", "").strip()
    if raw in ("0", "false", "no", "off", "none"):
        return None
    if raw:
        val = int(raw)
        return val if val > 0 else None
    return None


def fit_avatar_resolution(width: int, height: int) -> tuple[int, int]:
    """Downscale large portraits when user did not set width/height explicitly."""
    max_area = _default_max_area()
    if max_area is None or width * height <= max_area:
        return width, height
    from externals.image2image.comfy_workflow import snap_latent_size

    scale = (max_area / (width * height)) ** 0.5
    return snap_latent_size(round(width * scale), round(height * scale), multiple=16)


def audio_frame_budget(
    audio_path: Path,
    fps: float,
    *,
    cap: int | None = None,
) -> int:
    """Return frame count from audio duration at ``fps`` (small pad for rounding)."""
    duration: float | None = None
    try:
        import soundfile as sf

        duration = float(sf.info(str(audio_path)).duration)
    except Exception:
        try:
            import wave

            with wave.open(str(audio_path), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass
    if duration is None or duration <= 0:
        return cap or 400
    frames = int(duration * fps) + 4
    if cap is not None:
        frames = min(frames, cap)
    return max(frames, 1)


DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
