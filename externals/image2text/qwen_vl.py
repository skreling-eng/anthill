"""Qwen2-VL inference for $image2text."""

from __future__ import annotations

from pathlib import Path

_MODEL_CACHE: dict[tuple[str, bool], tuple[object, object]] = {}


def _resolve_device(use_gpu: bool) -> str:
    import torch

    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_dir: Path, *, use_gpu: bool) -> tuple[object, object]:
    key = (str(model_dir.resolve()), use_gpu)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = _resolve_device(use_gpu)
    load_kwargs: dict = {}
    if device == "cuda":
        load_kwargs["torch_dtype"] = "auto"
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    print(f"$image2text: loading Qwen2-VL from {model_dir} ({device})", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def describe_image(
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
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
    result = output[0].strip()
    return (result + "\n") if result else "\n"
