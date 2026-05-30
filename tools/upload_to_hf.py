#!/usr/bin/env python3
"""Upload models/ and test_data/ to skreling-eng/anthill (incremental).

Run from repo root:
  uv run python tools/upload_to_hf.py --token hf_...
  uv run python tools/upload_to_hf.py --bundle models --dry-run
  uv run python tools/upload_to_hf.py --bundle test-data
  uv run python tools/upload_to_hf.py --bundle all

Requires a Write HF token (not Read-only). Token precedence: --token, then HF_TOKEN env,
then cached ``hf auth login``. If .env sets a read-only HF_TOKEN, pass --token or update .env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_tools = Path(__file__).resolve().parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

from upload_models_to_hf import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    DEFAULT_REPO_ID,
    DEFAULT_TEST_DATA_DIR,
    check_write_access,
    fetch_remote_paths,
    resolve_token,
    upload_bundle,
)

BUNDLES = {
    "models": (DEFAULT_MODELS_DIR, "", "models/"),
    "test-data": (DEFAULT_TEST_DATA_DIR, "test_data", "test_data/"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        choices=("all", "models", "test-data"),
        default="all",
        help="What to upload (default: all = models then test_data)",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("ANTHILL_HF_REPO_ID", DEFAULT_REPO_ID),
        help=f"Target model repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without pushing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even when the path already exists on the Hub",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face Write token (overrides HF_TOKEN / hf auth login)",
    )
    args = parser.parse_args(argv)

    token = resolve_token(args.token)
    check_write_access(args.repo_id, token)
    print(f"Remote: https://huggingface.co/{args.repo_id}")

    remote_paths = fetch_remote_paths(args.repo_id, token)
    print(f"Hub:    {len(remote_paths)} file(s) already in repo")

    names = ("models", "test-data") if args.bundle == "all" else (args.bundle,)
    total_uploaded = 0
    for name in names:
        local_dir, prefix, label = BUNDLES[name]
        uploaded, _skipped = upload_bundle(
            local_dir=local_dir,
            path_prefix=prefix,
            repo_id=args.repo_id,
            token=token,
            dry_run=args.dry_run,
            force=args.force,
            remote_paths=remote_paths,
            label=label,
        )
        total_uploaded += uploaded
        if not args.dry_run and uploaded:
            remote_paths = fetch_remote_paths(args.repo_id, token)

    if total_uploaded and not args.dry_run:
        print(f"\nDone: {total_uploaded} new file(s) on {args.repo_id}")
    elif args.dry_run:
        print("\n(dry-run — no upload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
