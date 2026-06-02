"""Unified local model resolution + on-demand fetch from skreling-eng/anthill."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from externals.image.model_paths import models_roots

ANTHILL_REPO = os.environ.get("ANTHILL_HF_REPO_ID", "skreling-eng/anthill")

# Spot-check files (posix paths relative to models/). Shared with tools/download_models.py.
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
    "wan_aio": [
        "wan/wan2.2-i2v-rapid-aio-v10.safetensors",
        "wan/wan2.2-rapid-mega-aio-v12.safetensors",
    ],
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
    "qwen3_vl": [
        "qwen-vl/Qwen3-VL-8B-Instruct/config.json",
    ],
    "qwen_rapid_base": [
        "qwen-rapid/Qwen-Image-Edit-2509/model_index.json",
    ],
    "qwen_rapid_ckpt": [
        "qwen-rapid/Qwen-Rapid-AIO-SFW-v23.safetensors",
        "qwen-rapid/Qwen-Rapid-AIO-NSFW-v23.safetensors",
    ],
    "m2m100": [
        "m2m100_1.2B/config.json",
        "m2m100_1.2B/pytorch_model.bin",
        "m2m100_1.2B/sentencepiece.bpe.model",
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

_download_lock = threading.Lock()
_downloaded: set[str] = set()


def auto_download_enabled() -> bool:
    raw = os.environ.get("AH_ANTHILL_AUTO_DOWNLOAD", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def upstream_fallback_enabled() -> bool:
    raw = os.environ.get("AH_MODEL_UPSTREAM_FALLBACK", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def models_rel(*parts: str) -> str:
    return "/".join(parts).replace("\\", "/")


def primary_models_dir() -> Path:
    roots = models_roots()
    return roots[0] if roots else Path("models")


def resolve_models_file(rel: str) -> Path | None:
    """Find a file under models_roots() by repo-relative posix path."""
    rel_posix = rel.replace("\\", "/").lstrip("/")
    name = Path(rel_posix).name
    for root in models_roots():
        direct = root / rel_posix
        if direct.is_file():
            return direct.resolve()
        try:
            for hit in root.rglob(name):
                if hit.is_file() and hit.as_posix().endswith(rel_posix):
                    return hit.resolve()
        except OSError:
            continue
        if direct.is_file():
            return direct.resolve()
    return None


def resolve_models_dir(rel: str) -> Path | None:
    rel_posix = rel.replace("\\", "/").rstrip("/")
    for root in models_roots():
        direct = root / rel_posix
        if direct.is_dir():
            return direct.resolve()
    return None


def files_ready(relative_paths: list[str]) -> bool:
    return all(resolve_models_file(rel) is not None for rel in relative_paths)


def _hf_hub_download_file(rel: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download models from "
            f"https://huggingface.co/{ANTHILL_REPO}\n"
            "  uv sync --extra media\n"
            "  or: uv run python tools/download_models.py"
        ) from exc

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    rel_posix = rel.replace("\\", "/").lstrip("/")
    dest_root = primary_models_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=ANTHILL_REPO,
        filename=rel_posix,
        repo_type="model",
        local_dir=str(dest_root),
    )
    path = Path(downloaded)
    if path.is_file():
        return path.resolve()
    fallback = dest_root / rel_posix
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Download from {ANTHILL_REPO} finished but file is missing: {rel_posix}"
    )


def _hf_snapshot_pattern(pattern: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download models from "
            f"https://huggingface.co/{ANTHILL_REPO}"
        ) from exc

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    pattern = pattern.replace("\\", "/").strip("/")
    dest_root = primary_models_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ANTHILL_REPO,
        repo_type="model",
        local_dir=str(dest_root),
        allow_patterns=[f"{pattern}/**"],
    )


def ensure_anthill_file(rel: str, *, label: str = "", force: bool = False) -> Path:
    """Resolve a models/ file locally or download it from the anthill bundle."""
    rel_posix = rel.replace("\\", "/").lstrip("/")
    if not force:
        found = resolve_models_file(rel_posix)
        if found is not None:
            return found

    if not auto_download_enabled() and not force:
        raise FileNotFoundError(
            f"Model file not found: models/{rel_posix}. "
            f"Set AH_ANTHILL_AUTO_DOWNLOAD=1 or run: "
            f"uv run python tools/download_models.py"
        )

    with _download_lock:
        if not force and rel_posix in _downloaded:
            found = resolve_models_file(rel_posix)
            if found is not None:
                return found
        prefix = f"{label}: " if label else ""
        print(
            f"{prefix}downloading models/{rel_posix} from {ANTHILL_REPO}",
            flush=True,
        )
        path = _hf_hub_download_file(rel_posix)
        _downloaded.add(rel_posix)
        return path


def ensure_anthill_files(
    relative_paths: list[str],
    *,
    label: str = "",
    force: bool = False,
) -> list[Path]:
    return [ensure_anthill_file(rel, label=label, force=force) for rel in relative_paths]


def ensure_anthill_tree(
    rel_dir: str,
    *,
    ready: Callable[[], bool],
    label: str = "",
    force: bool = False,
) -> Path:
    """Ensure a directory tree exists under models/ (snapshot allow_patterns)."""
    rel_posix = rel_dir.replace("\\", "/").strip("/")
    if not force and ready():
        found = resolve_models_dir(rel_posix)
        if found is not None:
            return found

    if not auto_download_enabled() and not force:
        raise FileNotFoundError(
            f"Model tree not ready under models/{rel_posix}. "
            f"Run: uv run python tools/download_models.py"
        )

    with _download_lock:
        key = f"tree:{rel_posix}"
        if not force and key in _downloaded and ready():
            return resolve_models_dir(rel_posix) or (primary_models_dir() / rel_posix)
        prefix = f"{label}: " if label else ""
        print(
            f"{prefix}downloading models/{rel_posix}/** from {ANTHILL_REPO}",
            flush=True,
        )
        _hf_snapshot_pattern(rel_posix)
        _downloaded.add(key)
        if not ready():
            raise FileNotFoundError(
                f"Model tree still incomplete under models/{rel_posix} after "
                f"download from {ANTHILL_REPO}"
            )
        return resolve_models_dir(rel_posix) or (primary_models_dir() / rel_posix)


def require_models_file(rel: str, *, label: str = "") -> Path:
    """Resolve locally, else download from anthill when enabled."""
    found = resolve_models_file(rel)
    if found is not None:
        return found
    return ensure_anthill_file(rel, label=label)


def group_ready(name: str) -> bool:
    return files_ready(CHECKS[name])


def missing_group_names(profile: str) -> list[str]:
    """Group keys from profile that fail CHECKS spot-checks."""
    groups = PROFILE_GROUPS.get(profile, frozenset())
    return [name for name in sorted(groups) if not group_ready(name)]


def group_tree_prefix(name: str) -> str | None:
    """Longest common models/ directory prefix for a CHECKS group."""
    paths = CHECKS.get(name) or []
    if not paths:
        return None
    dir_parts = [
        p.replace("\\", "/").split("/")[:-1] for p in paths if "/" in p.replace("\\", "/")
    ]
    if not dir_parts:
        return None
    common: list[str] = []
    for i in range(min(len(parts) for parts in dir_parts)):
        segment = dir_parts[0][i]
        if all(parts[i] == segment for parts in dir_parts):
            common.append(segment)
        else:
            break
    return "/".join(common) if common else None


def sync_anthill_tree(rel_dir: str) -> None:
    """Download or refresh a models/ subtree from the anthill bundle."""
    rel_posix = rel_dir.replace("\\", "/").strip("/")
    with _download_lock:
        _hf_snapshot_pattern(rel_posix)
        _downloaded.add(f"tree:{rel_posix}")
