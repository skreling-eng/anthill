"""BLIP-2 inference for $image2text."""

from __future__ import annotations

from pathlib import Path

from externals.image2text.model_list import Image2TextModel
from externals.image2text.qwen_vl import (
    _cuda_load_kwargs,
    _from_pretrained_with_dtype,
    _is_cuda_oom,
    _normalize_hf_cache_env,
    _resolve_device,
    _verify_full_gpu,
)

_MODEL_CACHE: dict[tuple[str, str, bool, bool], tuple[object, object]] = {}

_DEFAULT_PROMPT = "Describe this image in detail."


def _load_blip2(model_dir: Path, *, use_gpu: bool, force_gpu: bool) -> tuple[object, object]:
    _normalize_hf_cache_env()
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = _resolve_device(use_gpu)
    if device == "cuda":
        load_kwargs = _cuda_load_kwargs(force_gpu=force_gpu)
    else:
        load_kwargs = {"dtype": torch.float32, "device_map": "cpu"}

    model = _from_pretrained_with_dtype(
        Blip2ForConditionalGeneration, model_dir, load_kwargs
    )
    processor = Blip2Processor.from_pretrained(str(model_dir))
    return model, processor


def load_model(
    profile: Image2TextModel,
    model_dir: Path,
    *,
    use_gpu: bool,
    force_gpu: bool = False,
) -> tuple[object, object]:
    if force_gpu and not use_gpu:
        raise RuntimeError(
            "$image2text: force_gpu=1 requires GPU (set gpu=1 or unset AH_IMAGE2TEXT_GPU=0)."
        )
    if force_gpu and _resolve_device(True) != "cuda":
        raise RuntimeError(
            "$image2text: force_gpu=1 but CUDA is not available (install CUDA torch or use gpu=0)."
        )

    key = (profile.family, str(model_dir.resolve()), use_gpu, force_gpu)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = _resolve_device(use_gpu)
    mode = "cuda (full GPU)" if device == "cuda" and force_gpu else device
    print(
        f"$image2text: loading {profile.dir_name()} ({profile.family}, {mode})",
        flush=True,
    )
    try:
        model, processor = _load_blip2(model_dir, use_gpu=use_gpu, force_gpu=force_gpu)
    except Exception as exc:
        if force_gpu and _is_cuda_oom(exc):
            raise RuntimeError(
                f"$image2text: out of GPU memory loading {profile.name!r} "
                f"({profile.dir_name()}). Try model='qwen2', close other GPU apps, "
                "or run without force_gpu=1 to allow CPU offload."
            ) from exc
        raise

    if force_gpu:
        _verify_full_gpu(model, profile_name=profile.name)

    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _prompt_text(prompt: str) -> str | None:
    text = prompt.strip()
    if not text or text == _DEFAULT_PROMPT:
        return None
    if text.lower().startswith("question:"):
        return text if text.lower().endswith("answer:") else f"{text} Answer:"
    return f"Question: {text} Answer:"


def describe_image(
    profile: Image2TextModel,
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text = _prompt_text(prompt)
    if text:
        inputs = processor(images=image, text=text, return_tensors="pt")
    else:
        inputs = processor(images=image, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return (output[0].strip() + "\n") if output[0].strip() else "\n"
