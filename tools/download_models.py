#!/usr/bin/env python3
"""Download Anthill bundle from skreling-eng/anthill (models/ + test_data/).

Re-runs are incremental: only missing profile groups and test_data files are fetched.

The Hub repo mirrors local trees:
  models/     → pulled into ./models/
  test_data/  → pulled into ./test_data/  (unless --skip-test-data)

  uv run python tools/download_models.py
  uv run python tools/download_models.py --status
  uv run python tools/download_models.py --upstream-fallback

Publish test_data (Write token):  init.bat -UploadTestData

Requires: hf auth login  (read token is fine for download)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODELS_DIR = REPO_ROOT / "models"
TEST_DATA_DIR = REPO_ROOT / "test_data"
from externals.anthill_models import (  # noqa: E402
    ANTHILL_REPO,
    CHECKS,
    PROFILE_GROUPS,
    ensure_anthill_file,
    group_ready,
    group_tree_prefix,
    missing_group_names,
    resolve_models_file,
    sync_anthill_tree,
)

# Spot-check that example media was pulled from the Hub.
TEST_DATA_CHECKS: list[str] = [
    "test_data/clip/demo-poster.png",
    "test_data/music/Cats.wav",
    "test_data/app/app_screenshot_1.png",
]


def _models_path(rel: str) -> Path:
    return MODELS_DIR / rel.replace("/", os.sep)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _models_dir_empty() -> bool:
    if not MODELS_DIR.is_dir():
        return True
    try:
        return not any(MODELS_DIR.iterdir())
    except OSError:
        return True


def _download_full_snapshot(*, dry_run: bool) -> None:
    _log(f"\n=== {ANTHILL_REPO} -> models/ (full bundle sync) ===")
    if dry_run:
        _log("  (dry-run — would hf snapshot_download entire bundle)")
        return
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    snapshot_download(
        ANTHILL_REPO,
        repo_type="model",
        local_dir=str(MODELS_DIR),
    )
    _log("  ok")


def _download_missing_groups(*, profile: str, dry_run: bool) -> None:
    missing = missing_group_names(profile)
    if not missing:
        _log("\n=== models/ — all groups ready for profile ===")
        return

    _log(f"\n=== Missing model groups ({len(missing)}) ===")
    for name in missing:
        _log(f"  - {name}")

    if dry_run:
        for name in missing:
            for rel in CHECKS[name]:
                if resolve_models_file(rel) is None:
                    _log(f"  would fetch models/{rel}")
            prefix = group_tree_prefix(name)
            if prefix and not group_ready(name):
                _log(f"  would sync models/{prefix}/**")
        return

    for name in missing:
        for rel in CHECKS[name]:
            if resolve_models_file(rel) is None:
                _log(f"  fetching models/{rel}")
                ensure_anthill_file(rel, label=name)

    for name in missing_group_names(profile):
        prefix = group_tree_prefix(name)
        if prefix:
            _log(f"\n=== sync models/{prefix}/** ===")
            sync_anthill_tree(prefix)

    still = missing_group_names(profile)
    if still:
        _log(
            f"\n  note: {len(still)} group(s) still incomplete after anthill sync: "
            + ", ".join(still)
        )
    else:
        _log("\n  all profile groups ready")


def download_anthill(*, profile: str, dry_run: bool) -> None:
    """Fetch missing models for profile (full sync on cold start, else incremental)."""
    missing = missing_group_names(profile)
    if not missing:
        _log("\n=== models/ — all groups ready for profile ===")
        return

    if _models_dir_empty() or len(missing) >= len(PROFILE_GROUPS[profile]):
        _download_full_snapshot(dry_run=dry_run)
        if dry_run:
            return
        still = missing_group_names(profile)
        if still:
            _log("\n=== finishing incomplete groups after full sync ===")
            _download_missing_groups(profile=profile, dry_run=False)
        return

    _download_missing_groups(profile=profile, dry_run=dry_run)


def download_test_data(*, dry_run: bool) -> None:
    if test_data_ready():
        _log("\n=== test_data/ — already present ===")
        return

    _log(f"\n=== {ANTHILL_REPO} -> test_data/ (missing files) ===")
    if dry_run:
        _log("  (dry-run — would hf download test_data/**)")
        return
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        ANTHILL_REPO,
        repo_type="model",
        local_dir=str(REPO_ROOT),
        allow_patterns=["test_data/**"],
    )
    _log("  ok")


def test_data_ready() -> bool:
    return all((REPO_ROOT / rel).is_file() for rel in TEST_DATA_CHECKS)


def print_test_data_status() -> None:
    _log("\ntest_data (from anthill test_data/**):")
    for rel in TEST_DATA_CHECKS:
        ok = (REPO_ROOT / rel).is_file()
        _log(f"  [{'ok' if ok else 'MISSING':7}] {rel}")


def print_status(profile: str) -> None:
    groups = PROFILE_GROUPS[profile]
    _log(f"\nModel status ({profile}):")
    for name in sorted(groups):
        ok = group_ready(name)
        _log(f"  [{'ok' if ok else 'MISSING':7}] {name}")


def print_notes() -> None:
    rvc = MODELS_DIR / "rvc"
    if not rvc.is_dir() or not any(rvc.iterdir()):
        _log(
            "\nOptional: RVC voices under models/rvc/<name>/ — only for $change_voice examples."
        )


def _run_upstream_fallback(*, dry_run: bool, profile: str) -> None:
    """Legacy per-repo downloads — only when anthill bundle is incomplete."""
    _log("\n=== upstream fallback (missing groups only) ===")
    if dry_run:
        _log("  (dry-run)")
        return

    groups = PROFILE_GROUPS[profile]
    missing = [n for n in sorted(groups) if not group_ready(n)]
    if not missing:
        return

    # Delegate to existing one-off scripts / externals where practical.
    if "wan_i2v_aux" in missing or "wan_t2v_config" in missing:
        import subprocess

        ps1 = REPO_ROOT / "tools" / "download_wan_models.ps1"
        if ps1.is_file():
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
                check=True,
                cwd=REPO_ROOT,
            )

    sys.path.insert(0, str(REPO_ROOT))
    if "demucs_openvino" in missing:
        from externals.music_separation.model_paths import ensure_model

        ensure_model()
    if "resemble_enhance" in missing:
        from externals.voice_enhance.model_paths import ensure_models

        ensure_models()
    if "kokoro" in missing:
        from externals.text2speech.assets import ensure_model_assets

        ensure_model_assets()
    if "code_qwen" in missing:
        from externals.code.model_paths import ensure_model

        ensure_model()
    if any(g.startswith("ocr_") for g in missing):
        from externals.ocr.model_paths import ensure_all_core_packs

        ensure_all_core_packs()
    if "qwen2_vl" in missing:
        from externals.image2text.model_paths import ensure_model

        ensure_model("qwen2")
    if "qwen3_vl" in missing:
        from externals.image2text.model_paths import ensure_model

        ensure_model("qwen3")
    if "qwen_rapid_base" in missing:
        from externals.image2image.qwen_pipeline import ensure_base_assets

        ensure_base_assets()
    if "m2m100" in missing:
        from externals.translate.model_paths import ensure_model

        ensure_model()
    if "qwen2_audio_4bit" in missing:
        from externals.audio_instruct.model_paths import ensure_model

        ensure_model()

    _log(
        "\nOther missing groups may need manual HF pulls — see models/*/README.md\n"
        f"Still missing after fallback: "
        + ", ".join(n for n in missing if not group_ready(n))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_GROUPS.keys()),
        default="standard",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print readiness table and exit",
    )
    parser.add_argument(
        "--upstream-fallback",
        action="store_true",
        help="After anthill pull, fetch remaining files from upstream HF repos",
    )
    parser.add_argument(
        "--skip-test-data",
        action="store_true",
        help="Do not download test_data/** from anthill (default: download)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Models root: {MODELS_DIR}")
    _log(f"Source:      https://huggingface.co/{ANTHILL_REPO}")

    if args.status:
        print_status(args.profile)
        print_test_data_status()
        print_notes()
        return 0

    os.environ.setdefault("AH_ANTHILL_AUTO_DOWNLOAD", "1")

    missing_before = missing_group_names(args.profile)
    if missing_before:
        _log(f"\nProfile {args.profile}: {len(missing_before)} group(s) to fetch")
    else:
        _log(f"\nProfile {args.profile}: models already complete")

    download_anthill(profile=args.profile, dry_run=args.dry_run)

    if not args.skip_test_data:
        download_test_data(dry_run=args.dry_run)
        if not args.dry_run:
            print_test_data_status()

    if args.upstream_fallback and not args.dry_run:
        _run_upstream_fallback(dry_run=False, profile=args.profile)

    print_status(args.profile)
    print_notes()

    missing = [n for n in sorted(PROFILE_GROUPS[args.profile]) if not group_ready(n)]
    if missing and not args.dry_run:
        _log(f"\nStill missing ({len(missing)}): {', '.join(missing)}")
        if not args.upstream_fallback:
            _log("Re-run with --upstream-fallback, or wait for anthill upload to finish.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
