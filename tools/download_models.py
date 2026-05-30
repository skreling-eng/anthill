#!/usr/bin/env python3
"""Download Anthill model weights from skreling-eng/anthill into models/.

The anthill Hugging Face repo mirrors the local models/ tree. One snapshot
pull is enough when the bundle is complete.

  uv run python tools/download_models.py
  uv run python tools/download_models.py --status
  uv run python tools/download_models.py --upstream-fallback   # only if anthill is incomplete

Requires: hf auth login  (read token is fine)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
ANTHILL_REPO = os.environ.get("ANTHILL_HF_REPO_ID", "skreling-eng/anthill")

# Key files used to report readiness after download.
CHECKS: dict[str, list[str]] = {
    "kokoro": ["kokoro/kokoro-v1_0.pth", "kokoro/config.json"],
    "resemble_enhance": [
        "resemble-enhance/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt"
    ],
    "demucs_openvino": ["demucs-openvino/htdemucs_v4/htdemucs_v4.xml"],
    "ace_step_gguf": [
        "ace-step-1.5/acestep-v15-xl-turbo-BF16.gguf",
        "ace-step-1.5/vae-BF16.gguf",
        "ace-step-1.5/Qwen3-Embedding-0.6B-BF16.gguf",
    ],
    "ace_step_st": [
        "ace-step-1.5_st/acestep-v15-turbo/model.safetensors",
        "ace-step-1.5_st/vae/diffusion_pytorch_model.safetensors",
    ],
    "flux_dev": ["FLUX.1-dev/flux1-dev.safetensors"],
    "flux_nf4": ["flux.1-dev-nf4-pkg/transformer/diffusion_pytorch_model.safetensors"],
    "wan_i2v_aux": ["wan/i2v-base/vae/diffusion_pytorch_model.safetensors"],
    "wan_t2v_config": ["wan/Wan2.2-T2V-A14B-Diffusers/transformer/config.json"],
    "wan_aio": ["wan/wan2.2-rapid-mega-aio-v12.safetensors"],
    "roformer_sw": ["roformer/BS-RoFormer-SW.ckpt"],
    "roformer_viperx": ["roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt"],
    "llm_gemma": ["llm/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ4_XS/model.gguf"],
    "code_qwen": ["code/Qwen2.5-Coder-14B-Instruct/model.gguf"],
    "ocr_en": [
        "ocr/PP-OCRv4/en/det/inference.pdmodel",
        "ocr/PP-OCRv4/en/rec/inference.pdmodel",
        "ocr/PP-OCRv4/en/cls/inference.pdmodel",
    ],
    "ocr_latin": [
        "ocr/PP-OCRv4/latin/det/inference.pdmodel",
        "ocr/PP-OCRv4/latin/rec/inference.pdmodel",
    ],
    "ocr_ch": [
        "ocr/PP-OCRv4/ch/det/inference.pdmodel",
        "ocr/PP-OCRv4/ch/rec/inference.pdmodel",
    ],
    "ocr_arabic": [
        "ocr/PP-OCRv4/arabic/rec/inference.pdmodel",
    ],
    "ocr_cyrillic": [
        "ocr/PP-OCRv4/cyrillic/rec/inference.pdmodel",
    ],
    "qwen2_vl": [
        "qwen-vl/Qwen2-VL-2B-Instruct/config.json",
        "qwen-vl/Qwen2-VL-2B-Instruct/model-00001-of-00002.safetensors",
        "qwen-vl/Qwen2-VL-2B-Instruct/model-00002-of-00002.safetensors",
    ],
    "qwen_rapid_base": [
        "qwen-rapid/Qwen-Image-Edit-2509/model_index.json",
    ],
    "qwen_rapid_ckpt": [
        "qwen-rapid/Qwen-Rapid-AIO-SFW-v23.safetensors",
        "qwen-rapid/Qwen-Rapid-AIO-NSFW-v23.safetensors",
    ],
}

PROFILE_GROUPS: dict[str, frozenset[str]] = {
    "minimal": frozenset(
        {
            "kokoro",
            "demucs_openvino",
            "ace_step_st",
            "flux_nf4",
            "roformer_viperx",
            "llm_gemma",
        }
    ),
    "standard": frozenset(CHECKS.keys()) - {"wan_i2v_aux", "wan_t2v_config", "wan_aio"},
    "full": frozenset(CHECKS.keys()),
}


def _models_path(rel: str) -> Path:
    return MODELS_DIR / rel.replace("/", os.sep)


def group_ready(name: str) -> bool:
    return all(_models_path(rel).is_file() for rel in CHECKS[name])


def _log(msg: str) -> None:
    print(msg, flush=True)


def download_anthill(*, dry_run: bool) -> None:
    _log(f"\n=== {ANTHILL_REPO} -> models/ ===")
    if dry_run:
        _log("  (dry-run — would hf download)")
        return
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    snapshot_download(
        ANTHILL_REPO,
        repo_type="model",
        local_dir=str(MODELS_DIR),
    )
    _log("  ok")


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

        ensure_model()
    if "qwen_rapid_base" in missing:
        from externals.image2image.qwen_pipeline import ensure_base_assets

        ensure_base_assets()

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
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Models root: {MODELS_DIR}")
    _log(f"Source:      https://huggingface.co/{ANTHILL_REPO}")

    if args.status:
        print_status(args.profile)
        print_notes()
        return 0

    download_anthill(dry_run=args.dry_run)

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
