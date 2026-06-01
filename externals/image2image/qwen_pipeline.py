"""Qwen-Image-Edit-2509 base assets and QwenImageEditPlusPipeline cache."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from transformers import AutoConfig, Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

from externals.anthill_models import ensure_anthill_tree, upstream_fallback_enabled
from externals.image.model_paths import models_roots
from externals.image2image.aio_loader import apply_aio_checkpoint, finalize_module
from externals.image2image.comfy_nodes import (
    DEFAULT_TARGET_SIZE,
    ModelSamplingDiscreteFlow,
    comfy_beta_sigmas,
    encode_comfy_prompt,
    prepare_comfy_edit_images,
)
from externals.image2image.comfy_sampler import QwenFlowDenoiser, sample_sa_solver

HF_BASE_REPO = "Qwen/Qwen-Image-Edit-2509"
BASE_SUBDIR = Path("qwen-rapid") / "Qwen-Image-Edit-2509"
COMFY_VAE_REFERENCE_AREA = DEFAULT_TARGET_SIZE * DEFAULT_TARGET_SIZE

_PIPELINE_CACHE: dict[tuple[str, bool], QwenImageEditPlusPipeline] = {}


def base_model_dir() -> Path:
    for root in models_roots():
        candidate = root / BASE_SUBDIR
        if (candidate / "model_index.json").is_file():
            return candidate
    return (models_roots()[0] / BASE_SUBDIR).resolve()


def base_assets_ready() -> bool:
    path = base_model_dir()
    return (path / "model_index.json").is_file() and (path / "processor").is_dir()


def ensure_base_assets(*, force: bool = False) -> Path:
    """Resolve tokenizer/processor configs (anthill bundle; optional upstream)."""
    path = base_model_dir()
    if base_assets_ready() and not force:
        return path

    ensure_anthill_tree(
        BASE_SUBDIR.as_posix(),
        ready=base_assets_ready,
        label="$image2image",
        force=force,
    )
    if base_assets_ready():
        return base_model_dir()

    if upstream_fallback_enabled():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "$image2image needs huggingface-hub: uv sync --extra media"
            ) from exc

        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            HF_BASE_REPO,
            local_dir=str(path),
            ignore_patterns=[
                "*.safetensors",
                "*.bin",
                "*.gguf",
                "*.pt",
                "*.pth",
            ],
        )

    if not base_assets_ready():
        raise FileNotFoundError(
            f"Base assets not ready under {path}. "
            f"Run: uv run python tools/download_models.py"
        )
    return path


def _resolve_dtype(*, use_gpu: bool) -> torch.dtype:
    if use_gpu and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_device(*, use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cuda_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _min_vram_gb_for_full_gpu() -> float:
    return float(os.environ.get("AH_IMAGE2IMAGE_MIN_VRAM_GB", "20"))


def should_full_gpu(*, use_gpu: bool) -> bool:
    if not use_gpu or not torch.cuda.is_available():
        return False
    return _cuda_total_vram_gb() >= _min_vram_gb_for_full_gpu()


def _comfy_scheduler_config(repo: str) -> dict:
    config = dict(FlowMatchEulerDiscreteScheduler.load_config(repo, subfolder="scheduler"))
    # Sigmas come from Comfy beta_scheduler; leave diffusers beta disabled.
    config["use_beta_sigmas"] = False
    config["use_karras_sigmas"] = False
    config["use_exponential_sigmas"] = False
    return config


def _apply_cuda_perf() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _tune_transformer(pipe: QwenImageEditPlusPipeline) -> None:
    transformer = pipe.transformer
    if hasattr(transformer, "set_attn_implementation"):
        try:
            transformer.set_attn_implementation("sdpa")
        except Exception:
            pass


def _pipeline_on_cuda(pipe: QwenImageEditPlusPipeline) -> bool:
    try:
        return next(pipe.transformer.parameters()).device.type == "cuda"
    except StopIteration:
        return False


def _place_pipeline(pipe: QwenImageEditPlusPipeline, *, use_gpu: bool, dtype: torch.dtype) -> None:
    device = _resolve_device(use_gpu=use_gpu)
    if use_gpu and device.type == "cuda" and should_full_gpu(use_gpu=use_gpu):
        if not _pipeline_on_cuda(pipe):
            print(
                f"$image2image: moving pipeline to GPU ({_cuda_total_vram_gb():.1f} GB VRAM)",
                flush=True,
            )
            pipe.to(device, dtype=dtype)
        else:
            print(
                f"$image2image: full GPU load ({_cuda_total_vram_gb():.1f} GB VRAM)",
                flush=True,
            )
        _apply_cuda_perf()
        _tune_transformer(pipe)
        return
    if use_gpu and device.type == "cuda":
        print("$image2image: CPU offload (insufficient VRAM for full GPU)", flush=True)
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to("cpu")
        return
    pipe.to("cpu")


def load_pipeline(checkpoint: Path, *, use_gpu: bool) -> QwenImageEditPlusPipeline:
    key = (str(checkpoint.resolve()), use_gpu)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    base = ensure_base_assets()
    dtype = _resolve_dtype(use_gpu=use_gpu)
    device = _resolve_device(use_gpu=use_gpu)
    repo = str(base)

    full_gpu = should_full_gpu(use_gpu=use_gpu)
    load_device = "cuda" if full_gpu else "cpu"
    print(
        f"$image2image: building pipeline from {checkpoint.name} ({device}, {dtype})",
        flush=True,
    )

    with init_empty_weights():
        transformer = QwenImageTransformer2DModel.from_config(
            QwenImageTransformer2DModel.load_config(repo, subfolder="transformer")
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(
            AutoConfig.from_pretrained(repo, subfolder="text_encoder")
        )
        vae = AutoencoderKLQwenImage.from_config(
            AutoencoderKLQwenImage.load_config(repo, subfolder="vae")
        )

    apply_aio_checkpoint(
        aio_path=checkpoint,
        transformer=transformer,
        text_encoder=text_encoder,
        vae=vae,
        load_device=load_device,
    )

    place_device = device if full_gpu else torch.device("cpu")
    transformer = finalize_module(
        transformer, dtype=dtype, device=place_device, label="transformer"
    )
    text_encoder = finalize_module(
        text_encoder, dtype=dtype, device=place_device, label="text_encoder"
    )
    vae = finalize_module(vae, dtype=dtype, device=place_device, label="vae")

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(_comfy_scheduler_config(repo))
    tokenizer = Qwen2Tokenizer.from_pretrained(repo, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(repo, subfolder="processor")

    pipe = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    _place_pipeline(pipe, use_gpu=use_gpu, dtype=dtype)

    _PIPELINE_CACHE[key] = pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe


def _decode_latents(pipe: QwenImageEditPlusPipeline, latents, *, height: int, width: int, output_type: str = "pil"):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type=output_type)


def _run_edit_inference(
    pipe: QwenImageEditPlusPipeline,
    *,
    images,
    prompt: str,
    steps: int,
    generator: torch.Generator | None,
    width: int,
    height: int,
):
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as pipe_mod

    calculate_shift = pipe_mod.calculate_shift

    # Comfy EmptyLatentImage: fixed output size (default 720x1280), not input aspect.
    multiple_of = pipe.vae_scale_factor * 2
    if width <= 0 or height <= 0:
        image_size = images[-1].size
        width, height = pipe_mod.calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    pipe.check_inputs(prompt, height, width, max_sequence_length=512)

    pipe._guidance_scale = 1.0
    pipe._attention_kwargs = {}
    pipe._current_timestep = None
    pipe._interrupt = False

    if not isinstance(images, list):
        images = [images]

    vl_images, vae_pil_images, vae_image_sizes = prepare_comfy_edit_images(
        images,
        target_size=DEFAULT_TARGET_SIZE,
    )
    vae_tensors = [
        pipe.image_processor.preprocess(img, vae_h, vae_w).unsqueeze(2)
        for img, (vae_w, vae_h) in zip(vae_pil_images, vae_image_sizes)
    ]

    device = pipe._execution_device
    dtype = pipe.text_encoder.dtype
    prompt_embeds, prompt_embeds_mask = encode_comfy_prompt(
        pipe,
        prompt=prompt,
        vl_images=vl_images,
        device=device,
        dtype=dtype,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        vae_tensors,
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    img_shapes = [
        [
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
            *[
                (1, vae_height // pipe.vae_scale_factor // 2, vae_width // pipe.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ],
        ]
    ]

    image_seq_len = latents.shape[1]
    flow_shift = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    sa_sigmas = comfy_beta_sigmas(steps, shift=flow_shift, device=device)
    model_sampling = ModelSamplingDiscreteFlow(shift=flow_shift)

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], 1.0, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    denoiser = QwenFlowDenoiser(
        pipe,
        image_latents=image_latents,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        guidance=guidance,
        latent_len=latents.size(1),
    )
    seed = generator.initial_seed() if generator is not None else None

    latents = sample_sa_solver(
        denoiser,
        latents,
        sa_sigmas,
        extra_args={"seed": seed},
        model_sampling=model_sampling,
    )
    return _decode_latents(pipe, latents, height=height, width=width)[0]


def run_edit(
    pipe: QwenImageEditPlusPipeline,
    *,
    image_paths: list[Path],
    prompt: str,
    steps: int,
    seed: int | None,
    width: int,
    height: int,
):
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    generator = None
    if seed is not None:
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    kwargs: dict = {}
    if width > 0 and height > 0:
        kwargs["width"] = width
        kwargs["height"] = height

    return _run_edit_inference(
        pipe,
        images=images,
        prompt=prompt,
        steps=steps,
        generator=generator,
        width=kwargs.get("width", 0),
        height=kwargs.get("height", 0),
    )
