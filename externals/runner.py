"""Run one $ external in an isolated Python (own venv when AH_EXTERNAL_PYTHON is set).

Usage (called by Anthill runtime, not usually by hand):
  python -m externals.runner image2video /path/to/op_dir

Reads:
  {op_dir}/input.json   — bundle (arrays + changes)
  {op_dir}/invoke.json  — args, prompt_text, repeat, arg_lists

Writes:
  {op_dir}/output.json  — result bundle

Requires AH_SESSION_BASE_DIR (set by parent) pointing at the session root.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session
from externals.api import ExternalContext, ExternalInput
from externals.bootstrap import bootstrap_external_env
from externals.invoke import load_invoke


def _load_handler(name: str):
    import importlib

    mod = importlib.import_module(f"externals.{name}.run")
    return mod.run


def run_in_process(name: str, op_dir: Path) -> ArrayBundle:
    base = os.environ.get("AH_SESSION_BASE_DIR", "").strip()
    if not base:
        raise RuntimeError(
            "AH_SESSION_BASE_DIR is not set (subprocess runner needs session root)"
        )
    session = Session(Path(base).resolve())
    ctx = ExternalContext(session=session, op_dir=op_dir.resolve())

    input_path = op_dir / "input.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}")
    bundle = ArrayBundle.from_dict(
        json.loads(input_path.read_text(encoding="utf-8"))
    )
    invoke = load_invoke(op_dir)
    inp = ExternalInput(
        bundle=bundle,
        args=invoke.get("args") or {},
        prompt_text=invoke.get("prompt_text") or "",
        repeat=int(invoke.get("repeat") or 1),
        arg_lists=invoke.get("arg_lists") or {},
    )
    handler = _load_handler(name)
    return handler(ctx, inp)


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if len(argv) != 2:
        print(
            "usage: python -m externals.runner <external_name> <op_dir>",
            file=sys.stderr,
        )
        return 2
    name, op_dir_str = argv
    op_dir = Path(op_dir_str).resolve()
    op_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_external_env()
    try:
        out = run_in_process(name, op_dir)
        (op_dir / "output.json").write_text(
            json.dumps(out.as_dict(), indent=2), encoding="utf-8"
        )
        return 0
    except Exception:
        err_path = op_dir / "error.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
