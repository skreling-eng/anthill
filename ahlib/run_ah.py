#!/usr/bin/env python3
"""Run an .ah program: parse, emulate externals, execute with session storage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ahlib.ah_parser import parse_ah_file
from ahlib.ah_runtime import run_program


def anthill_home() -> Path:
    """Anthill install root (models, .venvs, .env)."""
    if env := os.environ.get("AH_HOME"):
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def launch_dir() -> Path:
    """Directory where the user invoked ah / run_ah (sessions + $folder/$file cwd)."""
    if env := os.environ.get("AH_LAUNCH_DIR"):
        return Path(env).resolve()
    return Path.cwd().resolve()


def _load_dotenv() -> None:
    """Load project .env into os.environ (only unset keys)."""
    env_path = anthill_home() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def _bootstrap_env() -> None:
    from externals.music.models_env import configure_models_environment

    configure_models_environment()
    os.environ.setdefault("AH_HOME", str(anthill_home()))
    _load_dotenv()


def main() -> int:
    _bootstrap_env()
    parser = argparse.ArgumentParser(description="Interpret .ah agentic system files")
    parser.add_argument("file", type=Path, help="Path to .ah file")
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="Sessions root directory (default: <launch-dir>/sessions)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Anthill install root (default: AH_HOME or this package's repo)",
    )
    parser.add_argument(
        "--dump-parse",
        action="store_true",
        help="Print parsed instruction dictionary and exit",
    )
    args = parser.parse_args()

    work_dir = launch_dir()
    raw = args.file
    script_path = raw.resolve() if raw.is_absolute() else (work_dir / raw).resolve()
    if not script_path.is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    sessions_root = (args.sessions or work_dir / "sessions").resolve()
    repo_root = (args.repo_root or anthill_home()).resolve()

    os.chdir(work_dir)

    source = script_path.read_text(encoding="utf-8")

    if args.dump_parse:
        program_dict = parse_ah_file(str(script_path))
        print(json.dumps(program_dict, indent=2))
        return 0

    meta, session_dir = run_program(
        source,
        sessions_root,
        repo_root=repo_root,
    )
    print(json.dumps(meta, indent=2))
    print(f"\nSession: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
