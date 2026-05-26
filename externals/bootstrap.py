"""Env setup for external subprocess runners (matches run_ah.py parent bootstrap)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
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


def bootstrap_external_env() -> None:
    load_dotenv()
    from externals.music.models_env import configure_models_environment

    configure_models_environment()
