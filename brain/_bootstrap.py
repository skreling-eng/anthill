"""Ensure the `brain` package is importable regardless of cwd."""

from __future__ import annotations

import sys
from pathlib import Path

_BOOTSTRAPPED = False


def ensure_brain_importable() -> Path:
    """Insert repo root on sys.path when running scripts from brain/."""
    global _BOOTSTRAPPED
    brain_dir = Path(__file__).resolve().parent
    repo_root = brain_dir.parent
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _BOOTSTRAPPED = True
    return brain_dir
