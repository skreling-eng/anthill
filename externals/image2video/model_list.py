"""Named image-to-video profiles for $image2video(model='...')."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from externals.image.model_paths import resolve_model_path

DEFAULT_CHECKPOINT = "wan/wan2.2-rapid-mega-aio-v12.safetensors"
DEFAULT_BASE_REPO = os.environ.get(
    "WAN_I2V_BASE_REPO", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
)
DEFAULT_BASE_DIR = os.environ.get("WAN_I2V_BASE_DIR", "wan/i2v-base")
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

_LOCAL_BASE_CANDIDATES = (
    DEFAULT_BASE_DIR,
    "wan/Wan2.1-I2V-14B-480P-Diffusers",
    "wan/i2v-base",
)


@dataclass(frozen=True)
class VideoModel:
    name: str
    checkpoint: str
    base_repo: str = DEFAULT_BASE_REPO
    base_dir: str = DEFAULT_BASE_DIR
    num_frames: int = 81
    fps: int = 16
    # Phr00t MEGA rapid AIO: 4 steps, CFG 1 (matches Comfy README for v12).
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    max_area: int = 480 * 832

    def checkpoint_path(self) -> Path:
        return Path(resolve_model_path(self.checkpoint))

    def auxiliary_base(self) -> str:
        """Local diffusers tree (preferred) or Hugging Face repo id for VAE/encoders only."""
        if self.base_dir:
            path = Path(resolve_model_path(self.base_dir))
            if path.is_dir() and (path / "vae").is_dir():
                return str(path.resolve())
        for candidate in _LOCAL_BASE_CANDIDATES:
            if candidate == self.base_dir:
                continue
            path = Path(resolve_model_path(candidate))
            if path.is_dir() and (path / "vae").is_dir():
                return str(path.resolve())
        return self.base_repo


_video_models_list = [
    VideoModel(
        name="wan",
        checkpoint=DEFAULT_CHECKPOINT,
    ),
    VideoModel(
        name="default",
        checkpoint=DEFAULT_CHECKPOINT,
    ),
]

_video_models: dict[str, VideoModel] = {m.name: m for m in _video_models_list}


def get_video_model(name: str) -> VideoModel:
    if name in _video_models:
        return _video_models[name]
    if name == "default" or not name:
        return _video_models["wan"]
    available = ", ".join(sorted(_video_models))
    raise KeyError(f"Unknown image2video model {name!r}. Available: {available}")
