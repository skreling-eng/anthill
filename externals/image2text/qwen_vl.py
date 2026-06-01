"""Qwen2-VL and Qwen3-VL inference for $image2text."""

from __future__ import annotations

import os
from pathlib import Path

from externals.image2text.model_list import Image2TextModel

_MODEL_CACHE: dict[tuple[str, str, bool], tuple[object, object]] = {}


def _normalize_hf_cache_env() -> None:
    """Migrate deprecated TRANSFORMERS_CACHE env usage to HF_HOME/HUB cache."""
    raw = os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if not raw:
        return
    cache = Path(raw).expanduser()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if not hf_home:
        # Common legacy value is .../huggingface/hub.
        os.environ["HF_HOME"] = str(cache.parent if cache.name == "hub" else cache)
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if not hub_cache:
        hf = Path(os.environ["HF_HOME"])
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(
            cache if cache.name == "hub" else (hf / "hub")
        )
    # Unset deprecated var to silence transformers v5 warning path.
    os.environ.pop("TRANSFORMERS_CACHE", None)


def _from_pretrained_with_dtype(model_cls, model_dir: Path, load_kwargs: dict):
    """Use `dtype` (new API), fallback to legacy `torch_dtype` if required."""
    try:
        return model_cls.from_pretrained(str(model_dir), **load_kwargs)
    except TypeError:
        legacy = dict(load_kwargs)
        if "dtype" in legacy:
            legacy["torch_dtype"] = legacy.pop("dtype")
        return model_cls.from_pretrained(str(model_dir), **legacy)


def _resolve_device(use_gpu: bool) -> str:
    import torch

    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_qwen2(model_dir: Path, *, use_gpu: bool) -> tuple[object, object]:
    _normalize_hf_cache_env()
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = _resolve_device(use_gpu)
    load_kwargs: dict = {}
    if device == "cuda":
        load_kwargs["dtype"] = "auto"
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    model = _from_pretrained_with_dtype(
        Qwen2VLForConditionalGeneration, model_dir, load_kwargs
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def _load_qwen3(model_dir: Path, *, use_gpu: bool) -> tuple[object, object]:
    _normalize_hf_cache_env()
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "$image2text model=qwen3 needs transformers>=4.57 with Qwen3-VL support. "
            "uv sync --extra image2text"
        ) from exc

    device = _resolve_device(use_gpu)
    load_kwargs: dict = {}
    if device == "cuda":
        load_kwargs["dtype"] = "auto"
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    model = _from_pretrained_with_dtype(
        Qwen3VLForConditionalGeneration, model_dir, load_kwargs
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def load_model(
    profile: Image2TextModel,
    model_dir: Path,
    *,
    use_gpu: bool,
) -> tuple[object, object]:
    key = (profile.family, str(model_dir.resolve()), use_gpu)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = _resolve_device(use_gpu)
    print(
        f"$image2text: loading {profile.dir_name()} ({profile.family}, {device})",
        flush=True,
    )
    if profile.family == "qwen3":
        model, processor = _load_qwen3(model_dir, use_gpu=use_gpu)
    else:
        model, processor = _load_qwen2(model_dir, use_gpu=use_gpu)
    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _conversation(image_path: Path, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _describe_qwen2(
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    conversation = _conversation(image_path, prompt)
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return (output[0].strip() + "\n") if output[0].strip() else "\n"


def _describe_qwen3(
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_ids = inputs["input_ids"]
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return (output[0].strip() + "\n") if output[0].strip() else "\n"


def describe_image(
    profile: Image2TextModel,
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    if profile.family == "qwen3":
        return _describe_qwen3(
            model,
            processor,
            image_path,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    return _describe_qwen2(
        model,
        processor,
        image_path,
        prompt,
        max_new_tokens=max_new_tokens,
    )
