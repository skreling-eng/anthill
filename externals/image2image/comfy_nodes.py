"""Comfy TextEncodeQwenImageEditPlus + beta scheduler (Phr00t / ComfyUI parity)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import QwenImageEditPlusPipeline

try:
    import scipy.stats
except ImportError:  # pragma: no cover - scipy is a diffusers beta-scheduler dep
    scipy = None


def time_snr_shift(alpha: float, t: float) -> float:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

# Phr00t fixed-textencode-node / Comfy TextEncodeQwenImageEditPlus defaults.
VL_AREA = 384 * 384
DEFAULT_TARGET_SIZE = 896
from externals.image2image.comfy_qwen_prompt import (
    PICTURE_PROMPT,
    build_image_prompt_prefix,
    debug_log_prompt_encode,
    format_qwen_edit_llama_text,
    template_end_from_input_ids,
)


def _upscale_area(img: Image.Image, width: int, height: int) -> Image.Image:
    """Comfy common_upscale(..., 'area', 'disabled')."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid upscale size {width}x{height}")
    if width < img.width or height < img.height:
        resample = Image.Resampling.BOX
    else:
        resample = Image.Resampling.BICUBIC
    return img.resize((width, height), resample=resample)


def _upscale_lanczos_center(img: Image.Image, width: int, height: int) -> Image.Image:
    """Comfy common_upscale(..., 'lanczos', 'center')."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid upscale size {width}x{height}")
    scale = max(width / img.width, height / img.height)
    resized_w = max(1, round(img.width * scale))
    resized_h = max(1, round(img.height * scale))
    resized = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def vl_dimensions(width: int, height: int) -> tuple[int, int]:
    scale_by = math.sqrt(VL_AREA / (width * height))
    return round(width * scale_by), round(height * scale_by)


def vae_reference_dimensions(width: int, height: int, *, target_size: int = DEFAULT_TARGET_SIZE) -> tuple[int, int]:
    total = target_size * target_size
    scale_by = math.sqrt(total / (width * height))
    out_h = int(height * scale_by / 32) * 32
    out_w = int(width * scale_by / 32) * 32
    return max(32, out_w), max(32, out_h)


def prepare_comfy_edit_images(
    images: list[Image.Image],
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> tuple[list[Image.Image], list[Image.Image], list[tuple[int, int]]]:
    """Return VL images, VAE tensors-as-PIL path inputs, and VAE (w, h) sizes."""
    vl_images: list[Image.Image] = []
    vae_sizes: list[tuple[int, int]] = []
    vae_images: list[Image.Image] = []
    for img in images:
        w, h = img.size
        vl_w, vl_h = vl_dimensions(w, h)
        vl_images.append(_upscale_area(img, vl_w, vl_h))
        vae_w, vae_h = vae_reference_dimensions(w, h, target_size=target_size)
        vae_sizes.append((vae_w, vae_h))
        vae_images.append(_upscale_lanczos_center(img, vae_w, vae_h))
    return vl_images, vae_images, vae_sizes


def build_comfy_image_prompt(image_count: int) -> str:
    return "".join(PICTURE_PROMPT.format(i + 1) for i in range(image_count))


class ModelSamplingDiscreteFlow:
    """Comfy ModelSamplingDiscreteFlow (1000-step flow sigma table)."""

    def __init__(self, *, shift: float = 1.0, timesteps: int = 1000, multiplier: int = 1000) -> None:
        self.shift = float(shift)
        self.multiplier = multiplier
        self.noise_scale = 1.0
        sigmas = [
            float(time_snr_shift(self.shift, ((i + 1) / timesteps)))
            for i in range(timesteps)
        ]
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(time_snr_shift(self.shift, 1.0 - percent))


def comfy_beta_sigmas(
    steps: int,
    *,
    shift: float = 1.0,
    alpha: float = 0.6,
    beta: float = 0.6,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Comfy comfy/samplers.py beta_scheduler (not diffusers use_beta_sigmas)."""
    if scipy is None:
        raise RuntimeError("scipy is required for Comfy beta scheduler")
    model_sampling = ModelSamplingDiscreteFlow(shift=shift)
    total_timesteps = len(model_sampling.sigmas) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    sigs: list[float] = []
    last_t = -1
    for t in ts:
        ti = int(t)
        if ti != last_t:
            sigs.append(float(model_sampling.sigmas[ti]))
        last_t = ti
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32, device=device)


def encode_comfy_prompt(
    pipe: QwenImageEditPlusPipeline,
    *,
    prompt: str,
    vl_images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image_prompt = build_comfy_image_prompt(len(vl_images))
    txt = [format_qwen_edit_llama_text(image_prompt, prompt)]

    model_inputs = pipe.processor(
        text=txt,
        images=vl_images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    template_end = template_end_from_input_ids(model_inputs.input_ids)

    outputs = pipe.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = pipe._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[template_end:] for e in split_hidden_states]
    if split_hidden_states:
        debug_log_prompt_encode(
            user_prompt=prompt,
            llama_text=txt[0],
            template_end=template_end,
            seq_len=int(split_hidden_states[0].size(0)),
        )
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    if encoder_attention_mask.all():
        encoder_attention_mask = None
    return prompt_embeds, encoder_attention_mask
