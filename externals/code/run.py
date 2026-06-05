"""$code — coding LLM via local Qwen2.5-Coder-14B-Instruct GGUF."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.code.script_fixup import fixup_generated_ah
from externals.llm.context_limit import prompt_char_budget, trim_code_request
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_CODE_SYSTEM = (
    "You are an expert coding assistant. Follow the JSON request."
)

_AH_CODE_SYSTEM = (
    "You are an expert Anthill (.ah) coding assistant. Follow the JSON request. "
    "Output only valid .ah source. Every script MUST end with exactly one line "
    "run @instruction_name (e.g. run @answer) for the main entry instruction; "
    "without it the runtime does nothing."
)


def _truthy_arg(args: dict[str, str], key: str) -> bool:
    return args.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _ah_mode(inp: ExternalInput) -> bool:
    return _truthy_arg(inp.args, "ah")


def _code_system(inp: ExternalInput) -> str:
    custom = inp.args.get("system", "").strip()
    if custom:
        return custom
    if _ah_mode(inp):
        return _AH_CODE_SYSTEM
    return _DEFAULT_CODE_SYSTEM


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


def _model_record(
    model_name: str,
    *,
    emulate: bool = False,
    emulate_reason: str | None = None,
    ah: bool | None = None,
    llm: object | None = None,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    n_gpu_layers: int | None = None,
    extended_ctx: bool | None = None,
) -> dict:
    """Metadata written to model.json under the session op dir."""
    record: dict = {
        "external": "code",
        "model": model_name,
        "emulate": emulate,
    }
    if emulate_reason:
        record["emulate_reason"] = emulate_reason
    if ah is not None:
        record["ah"] = ah
    if max_tokens is not None:
        record["max_tokens"] = max_tokens
    if temperature is not None:
        record["temperature"] = temperature
    if n_ctx is not None:
        record["n_ctx"] = n_ctx
    if n_gpu_layers is not None:
        record["n_gpu_layers"] = n_gpu_layers
    if extended_ctx is not None:
        record["extended_ctx"] = extended_ctx
    if not emulate:
        try:
            from externals.code.model_paths import get_code_profile, resolve_profile_key

            key = resolve_profile_key(model_name)
            profile = get_code_profile(model_name)
            record["profile_key"] = key
            record["profile_subdir"] = profile.subdir
            record["gguf_name"] = profile.gguf_name
            record["chat_format"] = profile.chat_format
        except KeyError:
            pass
        if llm is not None:
            record["gguf_path"] = llm.gguf_path
            record["n_ctx"] = llm.n_ctx
            if llm.n_gpu_layers is not None:
                record["n_gpu_layers"] = llm.n_gpu_layers
            if llm.chat_format:
                record["chat_format"] = llm.chat_format
            if llm.yarn_orig_ctx:
                record["yarn_orig_ctx"] = llm.yarn_orig_ctx
                record["rope_scaling"] = "yarn"
    return record


def _write_model_json(ctx: ExternalContext, record: dict) -> None:
    (ctx.op_dir / "model.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _emulate(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    model: str,
    *,
    system: str,
) -> ArrayBundle:
    request_json = build_request_json(ctx, inp)
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
    system = _code_system(inp)
    ah = _ah_mode(inp)
    n_gpu_layers = _gpu_layers_from_args(inp)
    gen_meta = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n_gpu_layers": n_gpu_layers,
        "ah": ah,
    }

    if os.environ.get("AH_EMULATE_CODE", "").lower() in ("1", "true", "yes"):
        request_json = build_request_json(ctx, inp)
        (ctx.op_dir / "request.json").write_text(request_json + "\n", encoding="utf-8")
        _write_model_json(
            ctx,
            _model_record(
                model_name,
                emulate=True,
                emulate_reason="AH_EMULATE_CODE",
                **gen_meta,
            ),
        )
        return _emulate(ctx, inp, out, model_name, system=system)

    try:
        from externals.code.model_list import get_code_llm, resolve_code_n_ctx
        from externals.llm.context_limit import estimate_tokens

        raw_request = build_request(ctx, inp)
        n_ctx_raw = inp.args.get("n_ctx", "").strip()
        explicit_ctx = int(n_ctx_raw) if n_ctx_raw else None

        from externals.code.model_list import (
            YARN_ORIG_CTX,
            auto_max_code_n_ctx,
            extended_code_ctx_enabled,
        )

        trim_notes: list[str] = []
        sizing_request = raw_request
        if explicit_ctx is None:
            pre_cap = auto_max_code_n_ctx()
            if not extended_code_ctx_enabled():
                pre_cap = min(pre_cap, YARN_ORIG_CTX)
            sizing_request, trim_notes = trim_code_request(
                raw_request,
                budget_chars=prompt_char_budget(pre_cap, max_tokens),
            )

        sizing_json = json.dumps(sizing_request, ensure_ascii=False)
        n_ctx = resolve_code_n_ctx(
            sizing_json, max_tokens, explicit=explicit_ctx
        )
        request_json, more_trim_notes = _prepare_request(
            ctx,
            inp,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            raw_request=sizing_request if explicit_ctx is None else raw_request,
        )
        trim_notes = trim_notes + more_trim_notes
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
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        _write_model_json(
            ctx,
            _model_record(
                model_name,
                llm=llm,
                n_ctx=n_ctx,
                extended_ctx=extended_code_ctx_enabled(),
                **gen_meta,
            ),
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
            if ah:
                text, fix_notes = fixup_generated_ah(text)
                for note in fix_notes:
                    print(f"$code: {note}", file=sys.stderr, flush=True)
            link = ctx.new_link("texts", ".txt", text + "\n")
            out.texts.append(link)
    except ImportError as exc:
        print(
            f"$code fallback to emulate ({exc}). "
            "Install with: uv pip install llama-cpp-python"
        )
        _write_model_json(
            ctx,
            _model_record(
                model_name,
                emulate=True,
                emulate_reason="import_error",
                **gen_meta,
            ),
        )
        return _emulate(ctx, inp, out, model_name, system=system)
    except (KeyError, FileNotFoundError, OSError) as exc:
        print(f"$code error: {exc}")
        raise

    return out
