#!/usr/bin/env python3
"""Upload local models/ weights to a Hugging Face model repo (incremental).

Skips cache and junk directories (.cache, __pycache__, huggingface, openvino-cache, …).
Only uploads files whose path is not already present on the Hub repo.

Run from repo root:
  uv run python tools/upload_models_to_hf.py --dry-run
  uv run python tools/upload_models_to_hf.py

Requires a **Write** token (not Read-only):
  https://huggingface.co/settings/tokens  →  New token  →  type: Write
  hf auth login --token hf_...
  # or:  $env:HF_TOKEN = "hf_..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REPO_ID = "skreling-eng/anthill"

# Directory names anywhere under models/ — skipped entirely.
SKIP_DIR_NAMES = frozenset(
    {
        ".cache",
        "__pycache__",
        ".hub_tmp",
        "huggingface",  # models/huggingface/ (HF_HOME) and */.cache/huggingface/
        "openvino-cache",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

# Individual file names to skip.
SKIP_FILE_NAMES = frozenset(
    {
        ".gitignore",
        "CACHEDIR.TAG",
    }
)

# Small files are batched; large weights get one file per commit (LFS-friendly).
COMMIT_BATCH_SIZE = 25
LARGE_FILE_BYTES = 50 * 1024 * 1024  # 50 MiB


def _in_skip_dir(rel: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in rel.parts)


def iter_upload_candidates(models_dir: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, path_in_repo) for files to consider uploading."""
    if not models_dir.is_dir():
        raise SystemExit(f"models directory not found: {models_dir}")

    out: list[tuple[Path, str]] = []
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(models_dir)
        if _in_skip_dir(rel):
            continue
        if rel.name in SKIP_FILE_NAMES:
            continue
        if rel.suffix == ".metadata":
            continue
        out.append((path, rel.as_posix()))
    return out


def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_write_access(repo_id: str, token: str | None) -> None:
    """Fail fast if the token cannot upload to the repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "Not logged in to Hugging Face.\n"
            "  1. Create a Write token: https://huggingface.co/settings/tokens\n"
            "  2. Run: hf auth login --token hf_...\n"
            f"     ({exc})"
        ) from exc

    user = who.get("name") or who.get("fullname") or "?"
    print(f"Auth:   {user} ({who.get('type', 'user')})")

    access = (who.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")
    if role == "read":
        raise SystemExit(_write_token_help())
    if role == "fineGrained":
        repos = (access.get("repositories") or []) + (access.get("orgs") or [])
        repo_names = {r.get("name") for r in repos if isinstance(r, dict)}
        if repo_names and repo_id not in repo_names and repo_id.split("/")[0] not in repo_names:
            raise SystemExit(
                f"Fine-grained token has no write access to {repo_id!r}.\n"
                f"  Edit the token at https://huggingface.co/settings/tokens\n"
                f"  and allow write on repository {repo_id} (or org skreling-eng)."
            )


def _write_token_help() -> str:
    return (
        "Your Hugging Face token is Read-only; uploads need Write access.\n"
        "  1. https://huggingface.co/settings/tokens  →  Create new token  →  Write\n"
        "  2. hf auth login --token hf_...\n"
        "  3. If .env sets HF_TOKEN, replace it with the write token (it overrides hf auth)"
    )


def fetch_remote_paths(repo_id: str, token: str | None) -> set[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        raise SystemExit(
            f"Could not list remote files for {repo_id!r}: {exc}\n"
            "Run: hf auth login   (or set HF_TOKEN)"
        ) from exc
    return set(files)


def upload_missing(
    *,
    candidates: list[tuple[Path, str]],
    remote_paths: set[str],
    repo_id: str,
    token: str | None,
    dry_run: bool,
    force: bool,
) -> tuple[int, int]:
    missing = [(p, r) for p, r in candidates if force or r not in remote_paths]
    skipped = len(candidates) - len(missing)
    if not missing:
        print(f"Nothing to upload ({skipped} local file(s) already on Hub).")
        return 0, skipped

    total_bytes = sum(p.stat().st_size for p, _ in missing)
    print(f"Upload: {len(missing)} file(s), skip {skipped} already on Hub")
    print(f"        ~{total_bytes / (1024**3):.2f} GiB")

    for path, repo_path in missing:
        size_mb = path.stat().st_size / (1024**2)
        print(f"  + {repo_path}  ({size_mb:.1f} MiB)")

    if dry_run:
        print("(dry-run — no upload)")
        return len(missing), skipped

    from huggingface_hub import CommitOperationAdd, HfApi

    def _batches() -> list[list[tuple[Path, str]]]:
        large = [(p, r) for p, r in missing if p.stat().st_size >= LARGE_FILE_BYTES]
        small = [(p, r) for p, r in missing if p.stat().st_size < LARGE_FILE_BYTES]
        out: list[list[tuple[Path, str]]] = [[item] for item in large]
        for start in range(0, len(small), COMMIT_BATCH_SIZE):
            out.append(small[start : start + COMMIT_BATCH_SIZE])
        return out

    api = HfApi(token=token)
    uploaded = 0
    batches = _batches()
    for i, batch in enumerate(batches, start=1):
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(path))
            for path, repo_path in batch
        ]
        if len(batch) == 1:
            msg = f"anthill: add {batch[0][1]}"
        else:
            msg = f"anthill: add {len(batch)} small file(s)"
        print(f"\nCommit [{i}/{len(batches)}]: {msg}")
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=msg,
            )
        except Exception as exc:
            err = str(exc).lower()
            if "403" in err or "write token" in err or "forbidden" in err:
                raise SystemExit(_write_token_help()) from exc
            raise
        uploaded += len(batch)
        print(f"  done ({uploaded}/{len(missing)} files)")

    return uploaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("ANTHILL_HF_REPO_ID", DEFAULT_REPO_ID),
        help=f"Target model repo (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Local models root (default: models/)",
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
        help="HF token (default: HF_TOKEN env, else cached hf auth login)",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    token = resolve_token(args.token)

    check_write_access(args.repo_id, token)

    candidates = iter_upload_candidates(models_dir)
    print(f"Local:  {models_dir}  ({len(candidates)} file(s) after filters)")
    print(f"Remote: https://huggingface.co/{args.repo_id}")

    remote_paths = fetch_remote_paths(args.repo_id, token)
    print(f"Hub:    {len(remote_paths)} file(s) already in repo")

    uploaded, skipped = upload_missing(
        candidates=candidates,
        remote_paths=remote_paths,
        repo_id=args.repo_id,
        token=token,
        dry_run=args.dry_run,
        force=args.force,
    )
    if uploaded and not args.dry_run:
        print(f"\nUploaded {uploaded} file(s) to {args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
