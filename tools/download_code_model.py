#!/usr/bin/env python3
"""Download Qwen2.5-Coder GGUF for $code (14B or 1.5B Q4_K_M)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(CODE_MODELS.keys()),
        default="1.5b",
        help="GGUF profile to fetch (default: 1.5b)",
    )
    parser.add_argument(
        "--hf-instruct",
        action="store_true",
        help="Download full HF instruct weights (1.5b only, for QLoRA / transformers)",
    )
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="With --hf-instruct: skip GGUF download",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("AH_MODEL_UPSTREAM_FALLBACK", "1")

    from externals.code.model_paths import CODE_MODELS, ensure_instruct_hf, ensure_model

    if args.hf_instruct and args.hf_only:
        profile = CODE_MODELS[args.model]
        if not profile.hf_instruct_repo:
            print(f"--hf-instruct not configured for {args.model}", file=sys.stderr)
            sys.exit(1)
        hf_dir = ensure_instruct_hf(key=args.model, force=args.force)
        print(f"HF instruct ready: {hf_dir}")
        return

    path = ensure_model(key=args.model, force=args.force)
    print(f"GGUF ready: {path} ({path.stat().st_size / 1e9:.2f} GB)")

    if args.hf_instruct:
        profile = CODE_MODELS[args.model]
        if not profile.hf_instruct_repo:
            print(f"--hf-instruct not configured for {args.model}", file=sys.stderr)
            sys.exit(1)
        hf_dir = ensure_instruct_hf(key=args.model, force=args.force)
        print(f"HF instruct ready: {hf_dir}")


if __name__ == "__main__":
    main()
