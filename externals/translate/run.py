"""$translate — multilingual text translation via M2M100."""

from __future__ import annotations

import os

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_arg_list,
    read_bundle_texts,
    read_prompt_texts,
)
from ahlib.ah_runtime import ArrayBundle

_MODEL_CACHE: dict[tuple[str, str, bool], tuple[object, object]] = {}


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_TRANSLATE", "").lower() in ("1", "true", "yes")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _resolve_use_gpu(inp: ExternalInput) -> bool:
    raw = inp.args.get("gpu", "").strip()
    if raw:
        return _truthy(raw)
    env = os.environ.get("AH_TRANSLATE_GPU", "").strip()
    if env:
        return _truthy(env)
    return _cuda_available()


def _lang_arg(args: dict[str, str], *keys: str) -> str:
    for key in keys:
        val = args.get(key, "").strip().lower()
        if val:
            return val
    return ""


def _source_texts(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    texts = read_bundle_texts(ctx, inp)
    if texts:
        return texts
    prompts = read_prompt_texts(ctx, inp)
    if prompts:
        return prompts
    if inp.prompt_text.strip():
        return [inp.prompt_text.strip()]
    return []


def _help() -> str:
    return (
        "$translate requires torch, transformers, and sentencepiece (media venv).\n"
        "  tools\\setup_external_venvs.ps1   (or uv sync --extra media)\n"
        "  uv run python tools/download_models.py --upstream-fallback\n"
        "Test without models: AH_EMULATE_TRANSLATE=1"
    )


def _resolve_device(use_gpu: bool) -> str:
    import torch

    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_model(*, model_dir: str, use_gpu: bool) -> tuple[object, object]:
    key = (model_dir, _resolve_device(use_gpu), use_gpu)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    device = _resolve_device(use_gpu)
    load_kwargs: dict = {}
    if device == "cuda":
        load_kwargs["dtype"] = "auto"
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    print(f"$translate: loading M2M100 from {model_dir} device={device}", flush=True)
    try:
        model = M2M100ForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
    except TypeError:
        legacy = dict(load_kwargs)
        if "dtype" in legacy:
            legacy["torch_dtype"] = legacy.pop("dtype")
        model = M2M100ForConditionalGeneration.from_pretrained(model_dir, **legacy)
    tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _translate_text(
    model,
    tokenizer,
    *,
    text: str,
    src_lang: str,
    dst_lang: str,
    max_length: int,
) -> str:
    import torch

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(dst_lang),
            max_length=max_length,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()

    src_lang = _lang_arg(inp.args, "scr", "src")
    dst_lang = _lang_arg(inp.args, "dst")
    if not src_lang or not dst_lang:
        raise ValueError("$translate: scr= (or src=) and dst= are required")

    texts = _source_texts(ctx, inp)
    if not texts:
        link = ctx.new_link("texts", ".txt", "[ $translate: no texts[] or prompts input ]\n")
        out.texts.append(link)
        return out

    max_length = int(inp.args.get("max_length", "512"))
    use_gpu = _resolve_use_gpu(inp)
    model_name = read_arg_list(inp, "model", "default")[0]

    if _emulate_enabled():
        for text in texts:
            snippet = text.replace("\n", " ")[:80]
            content = (
                f"[emulated $translate scr={src_lang} dst={dst_lang}]\n"
                f"{snippet}\n"
            )
            out.texts.append(ctx.new_link("texts", ".txt", content))
        return out

    try:
        from externals.translate.model_list import get_translate_model
        from externals.translate.model_paths import ensure_model

        profile = get_translate_model(model_name)
        weights = ensure_model(profile)
        model, tokenizer = _get_model(model_dir=str(weights), use_gpu=use_gpu)
    except ImportError as exc:
        raise RuntimeError(_help()) from exc
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc

    for text in texts:
        try:
            translated = _translate_text(
                model,
                tokenizer,
                text=text,
                src_lang=src_lang,
                dst_lang=dst_lang,
                max_length=max_length,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"$translate: invalid language scr={src_lang!r} or dst={dst_lang!r} "
                f"({exc})"
            ) from exc
        if not translated.endswith("\n"):
            translated += "\n"
        out.texts.append(ctx.new_link("texts", ".txt", translated))

    return out
