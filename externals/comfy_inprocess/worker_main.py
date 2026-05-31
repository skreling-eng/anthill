"""Generic warm-worker entrypoint for GPU-heavy $ externals."""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _redirect_worker_logs_to_stderr() -> None:
    import builtins

    real_print = builtins.print

    def _print(*args, **kwargs):
        kwargs.setdefault("file", sys.stderr)
        kwargs.setdefault("flush", True)
        real_print(*args, **kwargs)

    builtins.print = _print  # type: ignore[misc, assignment]


def _release_worker_gpu() -> None:
    """Best-effort VRAM cleanup after a failed or heavy GPU job."""
    import gc

    gc.collect()
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_job(op_dir: Path, run_fn: Callable) -> None:
    from ahlib.ah_runtime import ArrayBundle, Session
    from externals.api import ExternalContext, ExternalInput
    from externals.invoke import load_invoke

    input_path = op_dir / "input.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}")

    base = os.environ.get("AH_SESSION_BASE_DIR", "").strip()
    if not base:
        raise RuntimeError("AH_SESSION_BASE_DIR is not set")

    session = Session(Path(base).resolve())
    ctx = ExternalContext(session=session, op_dir=op_dir.resolve())
    bundle = ArrayBundle.from_dict(json.loads(input_path.read_text(encoding="utf-8")))
    invoke = load_invoke(op_dir)
    inp = ExternalInput(
        bundle=bundle,
        args=invoke.get("args") or {},
        prompt_text=invoke.get("prompt_text") or "",
        repeat=int(invoke.get("repeat") or 1),
        arg_lists=invoke.get("arg_lists") or {},
    )
    out = run_fn(ctx, inp)
    (op_dir / "output.json").write_text(
        json.dumps(out.as_dict(), indent=2),
        encoding="utf-8",
    )


def worker_main(run_fn: Callable) -> int:
    from externals.bootstrap import bootstrap_external_env

    _redirect_worker_logs_to_stderr()
    bootstrap_external_env()
    _emit({"status": "ready"})
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("cmd") == "shutdown":
            break

        op_dir = Path(msg["op_dir"]).resolve()
        session_base = msg.get("session_base_dir", "").strip()
        if session_base:
            os.environ["AH_SESSION_BASE_DIR"] = str(Path(session_base).resolve())

        try:
            run_job(op_dir, run_fn)
        except Exception as exc:
            err_text = traceback.format_exc()
            (op_dir / "error.txt").write_text(err_text, encoding="utf-8")
            sys.stderr.write(err_text)
            if not err_text.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.flush()
            if "OutOfMemoryError" in err_text or "out of memory" in err_text.lower():
                _release_worker_gpu()
            _emit(
                {
                    "status": "error",
                    "op_dir": str(op_dir),
                    "error": str(exc),
                }
            )
            continue

        _emit({"status": "ok", "op_dir": str(op_dir)})
    return 0
