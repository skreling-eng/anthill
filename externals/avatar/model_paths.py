"""Model defaults for $avatar (SkyReels V3 A2V via ComfyUI-WanVideoWrapper)."""

from __future__ import annotations

import os

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


def default_attention_mode() -> str:
    """sdpa works everywhere; sageattn needs sageattention+triton (often missing on Windows)."""
    raw = os.environ.get("AVATAR_ATTENTION_MODE", "").strip()
    if raw:
        return raw
    try:
        import sageattention  # noqa: F401
    except Exception:
        return "sdpa"
    return "sageattn"

DEFAULT_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
