"""Run brain desktop app: python -m brain (from repo root) or uv run brain (from brain/)."""

from brain._bootstrap import ensure_brain_importable

ensure_brain_importable()

from brain.app import main

if __name__ == "__main__":
    main()
