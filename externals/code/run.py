"""$code — coding LLM via local Qwen2.5-Coder-14B-Instruct GGUF."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.llm.context_limit import prompt_char_budget, trim_code_request
from ahlib.ah_runtime import ArrayBundle


def _variant_count(inp: ExternalInput) -> int:
    if inp.repeat > 1:
        return inp.repeat
    return max(1, int(inp.args.get("count", "1")))


def _gpu_layers_from_args(inp: ExternalInput) -> int | None:
    raw = inp.args.get("gpu_layers", inp.args.get("n_gpu_layers"))
    if raw is None or raw == "":
        return None
    return int(raw)


def _read_file_entries(ctx: ExternalContext, inp: ExternalInput) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for link in inp.bundle.files:
        path = Path(link)
        content = ctx.read_link_text(link)
        entries.append({"path": link, "name": path.name, "content": content})
    return entries


def build_request(ctx: ExternalContext, inp: ExternalInput) -> dict:
    """Build the JSON request from prompts[], texts[] (code context), and files[]."""
    prompts: list[str] = []
    if inp.prompt_text.strip():
        prompts.append(inp.prompt_text.strip())
    for link in inp.bundle.prompts:
        text = ctx.read_link_text(link)
        if text:
            prompts.append(text)
    context_parts: list[str] = []
    for link in inp.bundle.texts:
        text = ctx.read_link_text(link)
        if text:
            context_parts.append(text)

    return {
        "prompts": prompts,
        "code_context": "\n\n".join(context_parts),
        "files": _read_file_entries(ctx, inp),
    }


def _prepare_request(
    ctx: ExternalContext,
    inp: ExternalInput,
    *,
    n_ctx: int,
    max_tokens: int,
    raw_request: dict | None = None,
) -> tuple[str, list[str]]:
    request, notes = trim_code_request(
        raw_request if raw_request is not None else build_request(ctx, inp),
        budget_chars=prompt_char_budget(n_ctx, max_tokens),
    )
    return json.dumps(request, ensure_ascii=False, indent=2), notes


def build_request_json(ctx: ExternalContext, inp: ExternalInput) -> str:
    return json.dumps(build_request(ctx, inp), ensure_ascii=False, indent=2)


def _emulate(
    ctx: ExternalContext, inp: ExternalInput, out: ArrayBundle, model: str
) -> ArrayBundle:
    request_json = build_request_json(ctx, inp)
    system = inp.args.get("system", "")
    for i in range(_variant_count(inp)):
        content = (
            f"[emulated $code model={model} variant={i}]\n"
            f"system: {system}\n"
            f"request:\n{request_json}\n"
        )
        link = ctx.new_link("texts", ".txt", content)
        out.texts.append(link)
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()
    model_name = inp.args.get("model", "default")
    max_tokens = int(inp.args.get("max_tokens", "2048"))
    temperature = float(inp.args.get("temperature", "0.2"))
    seed_base = int(inp.args.get("seed", "0"))
    system = inp.args.get(
        "system",
        "You are an expert coding assistant. Follow the JSON request.",
    )

    if os.environ.get("AH_EMULATE_CODE", "").lower() in ("1", "true", "yes"):
        request_json = build_request_json(ctx, inp)
        (ctx.op_dir / "request.json").write_text(request_json + "\n", encoding="utf-8")
        return _emulate(ctx, inp, out, model_name)

    try:
        from externals.code.model_list import get_code_llm, resolve_code_n_ctx
        from externals.llm.context_limit import estimate_tokens

        raw_request = build_request(ctx, inp)
        sizing_json = json.dumps(raw_request, ensure_ascii=False)
        n_ctx_raw = inp.args.get("n_ctx", "").strip()
        explicit_ctx = int(n_ctx_raw) if n_ctx_raw else None
        n_ctx = resolve_code_n_ctx(
            sizing_json, max_tokens, explicit=explicit_ctx
        )
        request_json, trim_notes = _prepare_request(
            ctx, inp, n_ctx=n_ctx, max_tokens=max_tokens, raw_request=raw_request
        )
        (ctx.op_dir / "request.json").write_text(request_json + "\n", encoding="utf-8")
        est_prompt = estimate_tokens(request_json)
        print(
            f"$code: n_ctx={n_ctx} "
            f"(est. prompt ~{est_prompt} tok, max_tokens={max_tokens}"
            f"{', YaRN' if n_ctx > 32768 else ''})",
            file=sys.stderr,
            flush=True,
        )
        for note in trim_notes:
            print(f"$code: {note}", file=sys.stderr, flush=True)

        llm = get_code_llm(
            model_name,
            n_gpu_layers=_gpu_layers_from_args(inp),
            n_ctx=n_ctx,
        )

        count = _variant_count(inp)
        for i in range(count):
            seed = seed_base if seed_base else random.randint(0, 2**31 - 1)
            if count > 1 and seed_base:
                seed = seed_base + i
            text = llm.complete(
                request_json,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            link = ctx.new_link("texts", ".txt", text + "\n")
            out.texts.append(link)
    except ImportError as exc:
        print(
            f"$code fallback to emulate ({exc}). "
            "Install with: uv pip install llama-cpp-python"
        )
        return _emulate(ctx, inp, out, model_name)
    except (KeyError, FileNotFoundError, OSError) as exc:
        print(f"$code error: {exc}")
        raise

    return out
