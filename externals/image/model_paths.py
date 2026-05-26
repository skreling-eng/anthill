"""Resolve checkpoint / LoRA paths under project models/ (and optional MODELS_PATH)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Repo root: externals/image/model_paths.py -> anthill/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLUX_CKPT_ID = "FLUX.1-dev"
FLUX_CKPT_4BIT_ID = "flux.1-dev-nf4-pkg"
SD3_TURBO_CKPT_ID = "stable-diffusion-3.5-medium-turbo"


@lru_cache(maxsize=1)
def models_roots() -> tuple[Path, ...]:
    """Directories searched for checkpoints, LoRAs, and subfolders."""
    roots: list[Path] = []
    extra = os.environ.get("MODELS_PATH", "").strip()
    if extra:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if part:
                roots.append(Path(part))
    roots.append(_PROJECT_ROOT / "models")
    # de-dupe, preserve order
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        key = r.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return tuple(unique)


def _exists(path: Path) -> bool:
    return path.is_file() or path.is_dir()


def _match_name(root: Path, name: str) -> Path | None:
    """Case-insensitive name match for a single path segment under root."""
    if not root.is_dir():
        return None
    target = name.lower()
    for child in root.iterdir():
        if child.name.lower() == target:
            return child
    return None


def resolve_model_path(ref: str) -> str:
    """
    Find a local file or directory for ref by searching models_roots().

    - Absolute / cwd-relative existing paths are returned as-is.
    - Otherwise try root/ref, then case-insensitive segment match, then rglob by basename.
    - If nothing is found, return ref unchanged (Hugging Face hub id).
    """
    if not ref:
        return ref

    path = Path(ref)
    if _exists(path):
        return str(path.resolve())

    ref_posix = ref.replace("\\", "/")
    basename = Path(ref_posix).name

    for root in models_roots():
        if not root.is_dir():
            continue

        direct = root / ref_posix
        if _exists(direct):
            return str(direct.resolve())

        # e.g. ref "FLUX.1-dev" vs folder "FLUX.1-dev" with different casing
        matched = _match_name(root, basename if "/" not in ref_posix else ref_posix.split("/")[0])
        if matched and _exists(matched):
            if "/" in ref_posix:
                nested = matched
                for part in ref_posix.split("/")[1:]:
                    nested = nested / part
                    hit = _match_name(nested.parent, part) if nested.parent.is_dir() else None
                    nested = hit or nested
                if _exists(nested):
                    return str(nested.resolve())
            return str(matched.resolve())

    # deep search: lora/foo.safetensors or any checkpoint file
    for root in models_roots():
        if not root.is_dir():
            continue
        try:
            for hit in root.rglob(basename):
                if _exists(hit):
                    return str(hit.resolve())
        except OSError:
            continue

    return ref


def resolve_pretrained_dir(ref: str) -> str:
    """Resolve a diffusers-style repo directory (from_pretrained root)."""
    return resolve_model_path(ref)


def resolve_lora(ref: str) -> tuple[str, str]:
    """
    Return (load_path, weight_name) for pipeline.load_lora_weights().

    load_path is a directory or repo id; weight_name is the .safetensors filename when needed.
    """
    if not ref:
        return ref, ref

    resolved = resolve_model_path(ref)
    p = Path(resolved)
    if p.is_file() and p.suffix.lower() == ".safetensors":
        return str(p.parent), p.name
    if p.is_dir():
        weights = list(p.glob("*.safetensors"))
        if len(weights) == 1:
            return str(p), weights[0].name
        return resolved, Path(ref).name
    return resolved, Path(ref).name


def subfolder_path(repo_ref: str, subfolder: str) -> str:
    """Resolve repo/subfolder for from_pretrained(..., subfolder=...)."""
    repo = resolve_pretrained_dir(repo_ref)
    sub = Path(repo) / subfolder
    if _exists(sub):
        return str(sub.resolve())
    return f"{repo}/{subfolder}"


def load_pretrained_sub(model_cls, repo_ref: str, subfolder: str, **kwargs):
    """Load a subfolder from a resolved local repo, or fall back to hub + subfolder."""
    local = subfolder_path(repo_ref, subfolder)
    if Path(local).is_dir():
        return model_cls.from_pretrained(local, **kwargs)
    repo = resolve_pretrained_dir(repo_ref)
    return model_cls.from_pretrained(repo, subfolder=subfolder, **kwargs)
