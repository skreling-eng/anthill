"""Qwen2-Audio 4-bit inference for $audio_instruct."""

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
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    device = _resolve_device(use_gpu)
    if device != "cuda":
        raise RuntimeError(
            "$audio_instruct requires a CUDA GPU (4-bit Qwen2-Audio via bitsandbytes)."
        )

    print(f"$audio_instruct: loading Qwen2-Audio from {model_dir} ({device})", flush=True)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(model_dir),
        device_map="auto",
        quantization_config=bnb_config,
    )
    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def _load_audio_array(path: Path, *, sample_rate: int):
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio


def _audio_source(ele: dict) -> Path | None:
    """Local path from Qwen2-Audio chat content (audio_url is used upstream too)."""
    raw = ele.get("audio_url") or ele.get("path") or ""
    if not raw:
        return None
    text = str(raw).strip()
    if text.startswith("file://"):
        text = text[7:]
    path = Path(text)
    if path.is_file():
        return path.resolve()
    return None


def _collect_audios(conversation: list[dict], *, sample_rate: int) -> list:
    audios: list = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for ele in content:
            if ele.get("type") != "audio":
                continue
            path = _audio_source(ele)
            if path is None:
                continue
            audios.append(_load_audio_array(path, sample_rate=sample_rate))
    return audios


def _conversation(
    audio_path: Path,
    prompt: str,
    *,
    system: str,
) -> list[dict]:
    messages: list[dict] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    # Chat template expects audio_url (local path is fine — see Qwen2-Audio demo).
    audio_ref = str(audio_path.resolve())
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_ref},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def answer_audio(
    model,
    processor,
    audio_path: Path,
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
) -> str:
    import torch

    if not audio_path.is_file():
        raise FileNotFoundError(f"$audio_instruct: audio not found: {audio_path}")

    conversation = _conversation(audio_path, prompt, system=system)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    sample_rate = processor.feature_extractor.sampling_rate
    audios = _collect_audios(conversation, sample_rate=sample_rate)

    audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
    if not audios:
        raise RuntimeError(
            f"$audio_instruct: failed to load audio from {audio_path}"
        )
    if audio_token and audio_token not in text:
        raise RuntimeError(
            f"$audio_instruct: chat template did not include {audio_token!r}; "
            "audio would be ignored by the model"
        )

    print(
        f"$audio_instruct: {audio_path.name} "
        f"({len(audios[0]) / sample_rate:.1f}s @ {sample_rate} Hz)",
        flush=True,
    )

    # Processor API is `audio=` (not `audios=` — that kwarg is silently ignored).
    audio_arg = audios[0] if len(audios) == 1 else audios
    inputs = processor(
        text=text,
        audio=audio_arg,
        return_tensors="pt",
        padding=True,
        sampling_rate=sample_rate,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]

    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    text_out = response.strip()
    return (text_out + "\n") if text_out else "\n"
