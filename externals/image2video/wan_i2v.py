"""Wan image-to-video: Comfy-style AIO split load (DiT + VAE + UMT5 from one safetensors)."""

from __future__ import annotations

import gc
import os
from pathlib import Path

from externals.image.model_paths import resolve_model_path
from externals.image2video.model_list import DEFAULT_NEGATIVE_PROMPT, VideoModel

_PIPE_CACHE: dict[str, object] = {}
_AIO_CKPT_CACHE: dict[str, dict] = {}
# Bump when load/attention policy changes so cached pipes are rebuilt.
_PIPE_CACHE_VERSION = 5

_UMT5_PREFIX = "text_encoders.umt5xxl.transformer."


def _patch_wan_i2v_prompt_clean() -> None:
    """diffusers only imports ftfy when present at first import; basic_clean still calls it."""
    import html

    import regex as re
    from diffusers.pipelines.wan import pipeline_wan_i2v as mod

    try:
        import ftfy
    except ImportError:
        ftfy = None

    def basic_clean(text):
        if ftfy is not None:
            text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(text):
        return re.sub(r"\s+", " ", text).strip()

    mod.ftfy = ftfy
    mod.basic_clean = basic_clean
    mod.whitespace_clean = whitespace_clean
    mod.prompt_clean = lambda text: whitespace_clean(basic_clean(text))


def _dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _enable_cuda_perf() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # benchmark autotune on 80k-token attention can add many minutes to step 1
    if os.environ.get("WAN_I2V_CUDNN_BENCHMARK", "").lower() in ("1", "true", "yes"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def _sage_status() -> tuple[bool, str]:
    """Whether diffusers can use the sage attention backend."""
    try:
        from diffusers.utils import is_sageattention_available, is_sageattention_version
    except Exception as exc:
        return False, f"diffusers check failed: {exc}"

    if not is_sageattention_available():
        return False, "sageattention not installed (uv sync --extra media)"

    ver = "unknown"
    try:
        import sageattention

        ver = getattr(sageattention, "__version__", ver)
    except ImportError as exc:
        msg = str(exc)
        if "triton" in msg.lower():
            msg += " — run tools\\setup_sage_windows.ps1 on Windows"
        return False, msg

    if is_sageattention_version(">=", "2.1.1"):
        try:
            import diffusers.models.attention_dispatch as ad

            if ad._CAN_USE_SAGE_ATTN and ad.sageattn is not None:
                return True, f"sageattention {ver}"
            return False, f"sageattention {ver} installed but diffusers sage backend disabled"
        except Exception as exc:
            return False, str(exc)

    try:
        from sageattention import sageattn, sageattn_varlen  # noqa: F401
    except ImportError as exc:
        msg = str(exc)
        if "triton" in msg.lower():
            msg += " — run tools\\setup_sage_windows.ps1 on Windows"
        return False, msg

    if _patch_diffusers_sage_legacy():
        return True, f"sageattention {ver} (legacy patch)"
    return False, f"sageattention {ver} too old; need >=2.1.1 (see tools/setup_sage_windows.ps1)"


def _patch_diffusers_sage_legacy() -> bool:
    """diffusers wants sageattention>=2.1.1; Windows wheels are often 1.0.6 with sageattn only."""
    try:
        from diffusers.utils import is_sageattention_available, is_sageattention_version

        if not is_sageattention_available():
            return False
        if is_sageattention_version(">=", "2.1.1"):
            return False
    except Exception:
        return False

    import diffusers.models.attention_dispatch as ad

    if ad._CAN_USE_SAGE_ATTN and ad.sageattn is not None:
        return True
    try:
        from sageattention import sageattn, sageattn_varlen
    except ImportError:
        return False

    ad._CAN_USE_SAGE_ATTN = True
    ad.sageattn = sageattn
    ad.sageattn_varlen = sageattn_varlen
    for name in (
        "sageattn_qk_int8_pv_fp8_cuda",
        "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "sageattn_qk_int8_pv_fp16_cuda",
        "sageattn_qk_int8_pv_fp16_triton",
    ):
        setattr(ad, name, None)
    return True


def _silence_attention_backend_logs():
    import contextlib
    import logging

    @contextlib.contextmanager
    def _cm():
        log = logging.getLogger("diffusers.models.modeling_utils")
        prev = log.level
        log.setLevel(logging.ERROR)
        try:
            yield
        finally:
            log.setLevel(prev)

    return _cm()


def _set_attention_backend(transformer, backend: str) -> bool:
    try:
        with _silence_attention_backend_logs():
            transformer.set_attention_backend(backend)
        return True
    except Exception as exc:
        if os.environ.get("WAN_I2V_DEBUG", "").strip():
            print(f"$image2video: backend {backend!r} failed: {exc}", flush=True)
        return False


def _reset_attention_native(transformer) -> None:
    """PyTorch SDPA with full kernel fallback (avoids flash-only 'No available kernel' errors)."""
    if not hasattr(transformer, "reset_attention_backend"):
        return
    transformer.reset_attention_backend()
    _set_attention_backend(transformer, "native")


def _try_attention_backend(transformer, *, default: str = "sage") -> str | None:
    """Default attention is sage (Comfy-like). Falls back to cudnn/native if sage unavailable."""
    if not hasattr(transformer, "set_attention_backend"):
        return None

    preferred = os.environ.get("WAN_I2V_ATTN", "").strip() or (default or "sage").strip()
    if preferred.lower() in ("0", "off", "reset"):
        with _silence_attention_backend_logs():
            transformer.reset_attention_backend()
        return None

    sage_legacy = _patch_diffusers_sage_legacy()
    if preferred == "sage":
        sage_ready, sage_detail = _sage_status()
        if not sage_ready:
            print(f"$image2video: sage unavailable ({sage_detail})", flush=True)

    backends: list[str] = [preferred]
    if preferred == "sage":
        backends.extend(("_native_cudnn", "native"))
    elif preferred in ("native", "_native_cudnn", "_native_efficient", "_native_math"):
        pass
    else:
        backends.extend(("_native_cudnn", "native"))

    fast = os.environ.get("WAN_I2V_FAST_ATTN", "").lower() in ("1", "true", "yes")
    import sys

    if fast and preferred == "sage":
        if sys.platform == "win32":
            backends[1:1] = ["_native_flash", "flash", "xformers"]
        else:
            backends[1:1] = ["flash", "xformers", "_native_flash"]

    seen: set[str] = set()
    ordered: list[str] = []
    for backend in backends:
        if backend and backend not in seen:
            seen.add(backend)
            ordered.append(backend)

    for backend in ordered:
        if _set_attention_backend(transformer, backend):
            if backend == "sage" and sage_legacy:
                print(
                    "$image2video: sageattention 1.x enabled (diffusers expects >=2.1.1)",
                    flush=True,
                )
            if preferred == "sage" and backend != "sage":
                print(
                    f"$image2video: using {backend} fallback (slower than Comfy sage/flash)",
                    flush=True,
                )
            return backend
    _reset_attention_native(transformer)
    if preferred == "sage":
        print("$image2video: sage failed; using native SDPA", flush=True)
    return "native"


def _configure_attention(pipe, attn: str = "sage") -> str | None:
    """Apply attention backend on each run (cached pipe may be reused)."""
    if getattr(pipe, "transformer", None) is None:
        return None
    backend = _try_attention_backend(pipe.transformer, default=attn)
    if backend:
        print(f"$image2video: attention backend={backend}", flush=True)
    return backend


def _is_attention_kernel_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "no available kernel" in msg or "no kernel image" in msg


def _finalize_pipe(pipe):
    pipe.to(_device())
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    _enable_cuda_perf()
    return pipe


def _is_local_base(base: str) -> bool:
    return Path(base).is_dir()


def _load_kwargs(base: str) -> dict:
    if _is_local_base(base):
        return {"local_files_only": True}
    return {}


def _load_aio_checkpoint(ckpt: Path) -> dict:
    """Load full Comfy AIO once per path (cached)."""
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    key = str(ckpt.resolve())
    if key not in _AIO_CKPT_CACHE:
        print(f"$image2video: loading AIO checkpoint {ckpt.name}")
        _AIO_CKPT_CACHE[key] = load_single_file_checkpoint(key)
    return _AIO_CKPT_CACHE[key]


# DiT-only keys for diffusers WanTransformer3DModel (no VACE — different pipeline class).
_AIO_NON_DIT_PREFIXES = (
    "vae.",
    "text_encoders.",
    "vace_",
    "motion_encoder.",
    "face_adapter.",
)
_AIO_NON_DIT_SUBSTRINGS = (
    "vace_patch_embedding",
    "vace_blocks",
    "motion_encoder",
    "face_adapter",
)


def _is_non_dit_checkpoint_key(key: str) -> bool:
    if key == "spiece_model":
        return True
    if key.startswith(_AIO_NON_DIT_PREFIXES):
        return True
    return any(part in key for part in _AIO_NON_DIT_SUBSTRINGS)


def _filter_dit_checkpoint(checkpoint: dict) -> tuple[dict, int]:
    filtered: dict = {}
    dropped = 0
    for key, value in checkpoint.items():
        if _is_non_dit_checkpoint_key(key):
            dropped += 1
            continue
        filtered[key] = value
    return filtered, dropped


def _checkpoint_patch_in_channels(checkpoint: dict) -> int | None:
    for key in ("model.diffusion_model.patch_embedding.weight", "patch_embedding.weight"):
        weight = checkpoint.get(key)
        if weight is not None:
            return int(weight.shape[1])
    return None


def _load_vae_from_aio(
    checkpoint: dict,
    base: str,
    *,
    load_kwargs: dict,
):
    from diffusers import AutoencoderKLWan

    vae_sd = {key[4:]: value for key, value in checkpoint.items() if key.startswith("vae.")}
    if not vae_sd:
        return None
    try:
        vae = AutoencoderKLWan.from_single_file(
            vae_sd,
            config=base,
            subfolder="vae",
            **load_kwargs,
        )
        print(f"$image2video: VAE from AIO ({len(vae_sd)} tensors)")
        return vae
    except Exception as exc:
        print(f"$image2video: VAE from AIO failed ({exc}); using hub/aux")
        return None


def _load_text_encoder_from_aio(
    checkpoint: dict,
    base: str,
    *,
    load_kwargs: dict,
):
    from transformers import UMT5EncoderModel

    te_sd = {
        key[len(_UMT5_PREFIX) :]: value
        for key, value in checkpoint.items()
        if key.startswith(_UMT5_PREFIX)
    }
    if len(te_sd) < 50:
        return None
    try:
        from transformers import AutoConfig

        te_config = AutoConfig.from_pretrained(
            base, subfolder="text_encoder", **load_kwargs
        )
        text_encoder = UMT5EncoderModel(te_config)
        missing, unexpected = text_encoder.load_state_dict(te_sd, strict=False)
        te_dtype = load_kwargs.get("torch_dtype")
        if te_dtype is not None:
            text_encoder.to(dtype=te_dtype)
        # AIO usually matches hub UMT5 exactly; one optional key may be hub-only.
        if len(missing) > 8:
            print(
                f"$image2video: UMT5 from AIO incomplete ({len(missing)} missing); "
                "using hub/aux text_encoder"
            )
            return None
        if missing:
            print(f"$image2video: UMT5 from AIO ({len(te_sd)} tensors, hub-only keys: {len(missing)})")
        else:
            print(f"$image2video: UMT5 from AIO ({len(te_sd)} tensors)")
        if unexpected:
            print(f"$image2video: UMT5 from AIO ignored {len(unexpected)} unexpected keys")
        return text_encoder
    except Exception as exc:
        print(f"$image2video: UMT5 from AIO failed ({exc}); using hub/aux")
        return None


def _load_transformer_comfy_aio(
    checkpoint: dict,
    dtype,
    *,
    local_files_only: bool = False,
):
    """MEGA/Comfy AIO: 16-ch patch; start frame via expand_timesteps (not 36-ch diffusers I2V)."""
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    cfg_repo = os.environ.get(
        "WAN_T2V_CONFIG_REPO", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    ).strip()
    if "/" in cfg_repo or "\\" in cfg_repo:
        cfg_repo = resolve_model_path(cfg_repo)
    cfg_kw = {"local_files_only": True} if local_files_only else {}
    if Path(cfg_repo).is_dir():
        cfg_kw.setdefault("local_files_only", True)
    config = dict(
        WanTransformer3DModel.load_config(cfg_repo, subfolder="transformer", **cfg_kw)
    )
    config["in_channels"] = 16

    converted = convert_wan_transformer_to_diffusers(checkpoint=dict(checkpoint))
    transformer = WanTransformer3DModel.from_config(config)
    missing, unexpected = transformer.load_state_dict(converted, strict=False)
    if missing:
        print(f"$image2video: DiT missing keys: {len(missing)}")
    if unexpected:
        print(f"$image2video: DiT unexpected keys: {len(unexpected)}")
    transformer.to(dtype=dtype)
    return transformer


def _load_transformer(
    aio: dict,
    model: VideoModel,
    base: str,
    dtype,
    *,
    load_kwargs: dict,
):
    from diffusers import WanTransformer3DModel

    dit_ckpt, dropped = _filter_dit_checkpoint(aio)
    if dropped:
        print(
            f"$image2video: DiT uses {len(dit_ckpt)} tensors "
            f"(skipped {dropped} VAE/UMT5/VACE from same file)"
        )

    in_channels = _checkpoint_patch_in_channels(dit_ckpt)
    local = load_kwargs.get("local_files_only", False)

    if in_channels == 16:
        print(
            "$image2video: 16-ch patch (Comfy/MEGA); "
            "start frame via expand_timesteps"
        )
        return _load_transformer_comfy_aio(dit_ckpt, dtype, local_files_only=local)

    try:
        return WanTransformer3DModel.from_single_file(
            dit_ckpt, torch_dtype=dtype, **load_kwargs
        )
    except Exception as exc:
        print(
            f"$image2video: DiT auto-detect failed ({exc}); "
            f"config from {base!r}/transformer"
        )
        return WanTransformer3DModel.from_single_file(
            dit_ckpt,
            config=base,
            subfolder="transformer",
            torch_dtype=dtype,
            **load_kwargs,
        )


def _load_pipeline_components(
    aio: dict,
    base: str,
    dtype,
    *,
    load_kwargs: dict,
):
    """Tokenizer/scheduler/CLIP from aux; VAE + UMT5 prefer AIO (Comfy layout)."""
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

    vae_kw = {**load_kwargs, "torch_dtype": dtype}
    vae = _load_vae_from_aio(aio, base, load_kwargs=vae_kw)
    if vae is None:
        vae = AutoencoderKLWan.from_pretrained(
            base, subfolder="vae", **vae_kw
        )
        print("$image2video: VAE from hub/aux")

    te_kw = {**load_kwargs, "torch_dtype": dtype}
    text_encoder = _load_text_encoder_from_aio(aio, base, load_kwargs=te_kw)
    if text_encoder is None:
        text_encoder = UMT5EncoderModel.from_pretrained(
            base, subfolder="text_encoder", **te_kw
        )
        print("$image2video: text_encoder from hub/aux")

    tokenizer = AutoTokenizer.from_pretrained(
        base, subfolder="tokenizer", **load_kwargs
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base, subfolder="scheduler", **load_kwargs
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        base, subfolder="image_encoder", **te_kw
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        base, subfolder="image_processor", **load_kwargs
    )
    return tokenizer, text_encoder, vae, scheduler, image_encoder, image_processor


def _load_pipeline(model: VideoModel):
    from diffusers import WanImageToVideoPipeline

    _patch_wan_i2v_prompt_clean()
    ckpt = model.checkpoint_path()
    dtype = _dtype()
    base = model.auxiliary_base()
    load_kwargs = _load_kwargs(base)

    if ckpt.is_dir():
        pipe = WanImageToVideoPipeline.from_pretrained(
            str(ckpt),
            **_load_kwargs(str(ckpt)),
        )
        return _finalize_pipe(pipe)

    if _is_local_base(base):
        print(f"$image2video: configs/tokenizer/CLIP from local {base}")
    else:
        print(f"$image2video: configs/tokenizer/CLIP from {base}")
        print(
            "  Tip: snapshot Wan2.1-I2V into models/wan/i2v-base/ "
            "(exclude transformer/*) for offline configs"
        )

    if not ckpt.is_file():
        raise FileNotFoundError(
            f"$image2video: checkpoint not found: {ckpt}\n"
            f"  Place {model.checkpoint} under models/ or set MODELS_PATH."
        )

    aio = _load_aio_checkpoint(ckpt)
    pipe_config = WanImageToVideoPipeline.load_config(base, **_load_kwargs(base))
    tokenizer, text_encoder, vae, scheduler, image_encoder, image_processor = (
        _load_pipeline_components(aio, base, dtype, load_kwargs=load_kwargs)
    )
    transformer = _load_transformer(
        aio, model, base, dtype, load_kwargs={**load_kwargs, "torch_dtype": dtype}
    )

    expand_timesteps = pipe_config.get("expand_timesteps") or False
    if getattr(transformer.config, "in_channels", None) == 16:
        expand_timesteps = True

    pipe = WanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        transformer_2=None,
        boundary_ratio=pipe_config.get("boundary_ratio"),
        expand_timesteps=expand_timesteps,
    )
    return _finalize_pipe(pipe)


def get_pipeline(model: VideoModel):
    key = f"{model.name}:v{_PIPE_CACHE_VERSION}"
    if key not in _PIPE_CACHE:
        _PIPE_CACHE[key] = _load_pipeline(model)
    return _PIPE_CACHE[key]


def _align_dim(value: int, mod: int) -> int:
    value = max(value, mod)
    return round(value) // mod * mod


def _area_cap_pixels() -> int | None:
    """Only scale down when WAN_I2V_MAX_AREA is set (Comfy can run full 768×1280 in ~8 min)."""
    raw = os.environ.get("WAN_I2V_MAX_AREA", "").strip()
    if not raw:
        return None
    val = int(raw)
    return val if val > 0 else None


def _fit_max_area(out_w: int, out_h: int, max_area: int, mod: int) -> tuple[int, int]:
    if out_w * out_h <= max_area:
        return out_w, out_h
    scale = (max_area / (out_w * out_h)) ** 0.5
    return _align_dim(round(out_w * scale), mod), _align_dim(round(out_h * scale), mod)


def _estimate_dit_tokens(pipe, *, width: int, height: int, num_frames: int) -> int:
    patch = pipe.transformer.config.patch_size
    vae_sf = pipe.vae_scale_factor_spatial
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latent_h = height // vae_sf
    latent_w = width // vae_sf
    return num_latent_frames * (latent_h // patch[1]) * (latent_w // patch[2])


def _prepare_start_image(
    image,
    pipe,
    *,
    width: int | None = None,
    height: int | None = None,
):
    """Output size: width=/height= args, else input size (optionally capped via WAN_I2V_MAX_AREA)."""
    patch = pipe.transformer.config.patch_size
    mod = pipe.vae_scale_factor_spatial * patch[1]
    aspect = image.height / image.width

    if width is not None and height is not None:
        out_w = _align_dim(width, mod)
        out_h = _align_dim(height, mod)
    elif width is not None:
        out_w = _align_dim(width, mod)
        out_h = _align_dim(round(out_w * aspect), mod)
    elif height is not None:
        out_h = _align_dim(height, mod)
        out_w = _align_dim(round(out_h / aspect), mod)
    else:
        out_w = _align_dim(image.width, mod)
        out_h = _align_dim(image.height, mod)

    cap = _area_cap_pixels()
    if cap is not None:
        new_w, new_h = _fit_max_area(out_w, out_h, cap, mod)
        if (new_w, new_h) != (out_w, out_h):
            print(
                f"$image2video: scaling {out_w}x{out_h} -> {new_w}x{new_h} "
                f"(WAN_I2V_MAX_AREA={cap})",
                flush=True,
            )
            out_w, out_h = new_w, new_h

    return image.resize((out_w, out_h)), out_h, out_w


def _make_step_callback(total_steps: int):
    import time

    t0 = time.time()
    t_prev = [t0]

    def on_step_end(pipeline, step_index, timestep, callback_kwargs):
        elapsed = time.time() - t_prev[0]
        t_prev[0] = time.time()
        if step_index == 0:
            print(
                f"$image2video: step 1/{total_steps} finished ({elapsed:.0f}s since denoise start)",
                flush=True,
            )
        else:
            print(
                f"$image2video: step {step_index + 1}/{total_steps} finished ({elapsed:.0f}s)",
                flush=True,
            )
        return callback_kwargs

    return on_step_end


def generate(
    model: VideoModel,
    *,
    image_path: Path,
    prompt: str,
    output_path: Path,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 0,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    num_frames: int | None = None,
    attn: str = "sage",
) -> Path:
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    pipe = get_pipeline(model)
    attn_backend = _configure_attention(pipe, attn)
    image = Image.open(image_path).convert("RGB")
    image, height, width = _prepare_start_image(
        image,
        pipe,
        width=width,
        height=height,
    )

    generator = None
    if seed:
        generator = torch.Generator(device=_device()).manual_seed(seed)

    steps = (
        num_inference_steps
        if num_inference_steps is not None
        else model.num_inference_steps
    )
    guidance = (
        guidance_scale if guidance_scale is not None else model.guidance_scale
    )
    frames = num_frames if num_frames is not None else model.num_frames
    raw_frames = os.environ.get("WAN_I2V_FRAMES", "").strip()
    if raw_frames:
        frames = int(raw_frames)

    tokens = _estimate_dit_tokens(pipe, width=width, height=height, num_frames=frames)
    print(
        f"$image2video: {width}x{height}, {frames} frames, "
        f"steps={steps}, guidance={guidance}, ~{tokens:,} DiT tokens/step",
        flush=True,
    )
    print("$image2video: encoding prompt/image…", flush=True)

    pipe_kwargs = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frames,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
        callback_on_step_end=_make_step_callback(steps),
    )

    print("$image2video: denoising step 1 starting…", flush=True)
    if attn_backend == "sage":
        print(
            "$image2video: first sage run compiles sm_89 CUDA kernels (ptxas lines); "
            "progress may stay at 0/4 for 1-3 min, then steps 2-4 are faster",
            flush=True,
        )

    try:
        result = pipe(**pipe_kwargs)
    except RuntimeError as exc:
        if not _is_attention_kernel_error(exc) or getattr(pipe, "transformer", None) is None:
            raise
        print(
            "$image2video: attention kernel failed, retrying with native SDPA",
            flush=True,
        )
        _reset_attention_native(pipe.transformer)
        result = pipe(**pipe_kwargs)
    frames = result.frames[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=model.fps)
    return output_path


def release_pipeline(model_name: str | None = None) -> None:
    import torch

    if model_name is None:
        keys = list(_PIPE_CACHE)
    else:
        keys = [k for k in _PIPE_CACHE if k.startswith(f"{model_name}:")]
    for key in keys:
        _PIPE_CACHE.pop(key, None)
    _AIO_CKPT_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Enable sageattention 1.x before diffusers checks (Windows wheels are often 1.0.6).
_patch_diffusers_sage_legacy()
