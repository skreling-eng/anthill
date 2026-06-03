"""$ah — execute Anthill (.ah) source from texts[] and return the result bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_bundle_texts
from ahlib.ah_parser import parse_ah_source
from ahlib.ah_parser import ARRAY_TYPES
from ahlib.ah_runtime import ArrayBundle, Runtime, RuntimeCancelled, Session

_FENCE_START = ("```ah", "```anthill", "```")


def _prefilter_ah_text(text: str) -> str:
    """Unescape LLM-style \\n and \\\" in texts[] before nested .ah parse ($ah only)."""
    if not text:
        return text
    if "\\n" not in text and '\\"' not in text and "\\t" not in text:
        return text
    protected = text.replace("\\\\", "\x00")
    protected = protected.replace("\\r\\n", "\n").replace("\\r", "\n")
    protected = protected.replace("\\n", "\n").replace("\\t", "\t")
    protected = protected.replace('\\"', '"').replace("\\'", "'")
    return protected.replace("\x00", "\\")


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_AH", "").lower() in ("1", "true", "yes")


def _unwrap_source(text: str) -> str:
    """Strip optional markdown fence from LLM-generated scripts."""
    text = _prefilter_ah_text(text.strip())
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    first = lines[0].strip().lower()
    if any(first.startswith(prefix) for prefix in _FENCE_START):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _read_source(ctx: ExternalContext, inp: ExternalInput) -> str:
    parts = [_unwrap_source(text) for text in read_bundle_texts(ctx, inp) if text.strip()]
    if not parts:
        raise ValueError("$ah requires AH source in texts[]")
    return "\n\n".join(parts)


def _resolve_entry(inp: ExternalInput, program) -> str:
    raw = inp.args.get("entry", "").strip()
    if raw.startswith("@"):
        raw = raw[1:]
    if raw:
        return raw
    target = program.run_target
    if not target:
        raise ValueError(
            "$ah: no run target — add 'run @instruction' to the script or entry=@name"
        )
    return target


def _nested_session_root(ctx: ExternalContext) -> Path:
    root = ctx.op_dir / "nested"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _reparent_bundle(
    bundle: ArrayBundle, nested_base: Path, parent_base: Path
) -> ArrayBundle:
    """Express nested-session links relative to the parent session root."""
    nested = nested_base.resolve()
    parent = parent_base.resolve()
    out = bundle.copy()
    for key in ARRAY_TYPES:
        if key == "changes":
            continue
        reparented: list = []
        for link in getattr(out, key):
            path = Path(link)
            if path.is_absolute():
                reparented.append(str(path.resolve()).replace("\\", "/"))
            else:
                full = (nested / link).resolve()
                reparented.append(
                    str(full.relative_to(parent)).replace("\\", "/")
                )
        setattr(out, key, reparented)
    return out


def _emulate(ctx: ExternalContext, source: str, entry: str) -> ArrayBundle:
    link = ctx.new_link(
        "texts",
        ".txt",
        f"[emulated $ah entry=@{entry}]\n{source}\n",
    )
    return ArrayBundle(texts=[link])


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    source = _read_source(ctx, inp)
    program = parse_ah_source(source)
    entry = _resolve_entry(inp, program)

    (ctx.op_dir / "script.ah").write_text(source, encoding="utf-8")

    if _emulate_enabled():
        return _emulate(ctx, source, entry)

    nested_root = _nested_session_root(ctx)
    nested_session = Session(nested_root)
    runtime = Runtime(
        program,
        nested_session,
        callback=ctx.callback,
        cancel_event=ctx.cancel_event,
    )
    try:
        result = runtime.run(entry)
    except RuntimeCancelled:
        raise RuntimeCancelled("$ah cancelled") from None

    result = _reparent_bundle(result, nested_root, ctx.session.base_dir)

    meta = {
        "entry": entry,
        "run_target": program.run_target,
        "output": result.as_dict(),
    }
    if runtime.last_output_json_path is not None:
        meta["output_json"] = str(runtime.last_output_json_path.resolve())
    (ctx.op_dir / "nested_run.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return result
