#!/usr/bin/env python3
"""Launch the Brain AH codebase analyzer.

Prefer the isolated venv launchers (no anthill deps):
  powershell -File brain/run.ps1
  ./brain/run.sh
"""

from __future__ import annotations

import sys
from pathlib import Path


def _warn_if_not_brain_venv() -> None:
    exe = Path(sys.executable).resolve()
    brain_venv = Path(__file__).resolve().parent / ".venv"
    try:
        exe.relative_to(brain_venv.resolve())
    except ValueError:
        print(
            "brain: not running from brain/.venv — use brain/run.ps1 or brain/run.sh "
            "for an isolated environment.",
            file=sys.stderr,
        )


def main() -> None:
    _warn_if_not_brain_venv()
    from brain._bootstrap import ensure_brain_importable

    ensure_brain_importable()
    from brain.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
