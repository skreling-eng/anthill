"""Subprocess entry for custom &action codegen and execution (custom-actions venv)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from ahlib.ah_runtime import ArrayBundle, Session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_complete(args: argparse.Namespace) -> int:
    from ahlib.custom_action_codegen import _code_complete_inprocess
    from ahlib.custom_action_env import ensure_custom_actions_env

    repo = Path(args.repo_root).resolve()
    ensure_custom_actions_env(repo)
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    text = _code_complete_inprocess(
        payload["prompt"],
        system=payload["system"],
        max_tokens=int(payload.get("max_tokens", 4096)),
        repo_root=repo,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ahlib.custom_action_codegen import load_run_function
    from ahlib.custom_action_env import ensure_venv, sync_imports_for_code

    repo = Path(args.repo_root).resolve()
    run_py = Path(args.run_py).resolve()
    op_dir = Path(args.op_dir).resolve()
    meta_path = run_py.parent / "meta.json"

    ensure_venv(repo)
    code = run_py.read_text(encoding="utf-8")
    if meta_path.is_file():
        sync_imports_for_code(repo, code, meta_path)

    base = os.environ.get("AH_SESSION_BASE_DIR", "").strip()
    if not base:
        print("AH_SESSION_BASE_DIR is required", file=sys.stderr)
        return 2
    session = Session(Path(base).resolve())

    input_path = op_dir / "input.json"
    bundle = ArrayBundle.from_dict(
        json.loads(input_path.read_text(encoding="utf-8"))
    )
    run_fn = load_run_function(run_py)
    base_dir = str(session.base_dir.resolve())
    op_dir_str = str(op_dir.resolve())
    import inspect

    params = list(inspect.signature(run_fn).parameters)
    if len(params) >= 3:
        raw = run_fn(bundle.as_dict(), base_dir, op_dir_str)
    else:
        raw = run_fn(bundle.as_dict(), base_dir)
    if not isinstance(raw, dict):
        raise TypeError(f"run() must return dict, got {type(raw).__name__}")
    (op_dir / "output.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_complete = sub.add_parser("complete", help="LLM completion for codegen/validate")
    p_complete.add_argument("--repo-root", required=True)
    p_complete.add_argument("--payload", required=True, help="JSON file with prompt/system")
    p_complete.add_argument("--out", required=True, help="Write completion text here")

    p_run = sub.add_parser("run", help="Execute custom_actions/<name>/run.py")
    p_run.add_argument("--repo-root", required=True)
    p_run.add_argument("--run-py", required=True)
    p_run.add_argument("--op-dir", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "complete":
            return cmd_complete(args)
        if args.cmd == "run":
            return cmd_run(args)
    except Exception:
        if args.cmd == "run":
            err = Path(args.op_dir) / "error.txt"
            err.write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
