"""Resolve paths for model_ah externals."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.code.model_paths import ensure_instruct_hf


def resolve_link_path(ctx: ExternalContext, link: str) -> Path:
    path = Path(link)
    if not path.is_absolute():
        path = ctx.base_dir / link
    return path.resolve()


def read_ah_files(ctx: ExternalContext, inp: ExternalInput) -> list[tuple[str, str]]:
    """All .ah texts from input files[] links."""
    items: list[tuple[str, str]] = []
    for link in inp.bundle.files:
        text = ctx.read_link_text(link)
        if text:
            items.append((link, text))
    return items


def read_jsonl_path(ctx: ExternalContext, inp: ExternalInput) -> Path:
    """First JSONL file from files[] (by path suffix or JSONL content)."""
    for link in inp.bundle.files:
        path = resolve_link_path(ctx, link)
        if path.suffix.lower() == ".jsonl" and path.is_file():
            return path
    for link in inp.bundle.files:
        path = resolve_link_path(ctx, link)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text and text.splitlines()[0].lstrip().startswith("{"):
            try:
                json.loads(text.splitlines()[0])
            except json.JSONDecodeError:
                continue
            return path
    raise RuntimeError(
        "$model_ah_train_lora needs a JSONL file in files[] "
        "(from $model_ah_create_jsonl)."
    )


def resolve_adapter_dir(ctx: ExternalContext, inp: ExternalInput, *, work_dir: Path) -> Path:
    """LoRA adapter directory from files[] (adapter dir, config file, or .zip)."""
    if not inp.bundle.files:
        raw = inp.args.get("adapter", "").strip()
        if raw:
            path = resolve_link_path(ctx, raw)
            if path.is_dir():
                return path
            if path.name == "adapter_config.json":
                return path.parent
        raise RuntimeError(
            "$model_ah_merge_lora needs a LoRA adapter in files[] "
            "(adapter_config.json link or .zip from $model_ah_train_lora)."
        )

    link = inp.bundle.files[0]
    path = resolve_link_path(ctx, link)

    if path.suffix.lower() == ".zip" and path.is_file():
        dest = work_dir / "adapter"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        if (dest / "adapter_config.json").is_file():
            return dest
        for sub in dest.iterdir():
            if sub.is_dir() and (sub / "adapter_config.json").is_file():
                return sub
        raise RuntimeError(f"No adapter_config.json in zip: {link}")

    if path.name == "adapter_config.json" and path.is_file():
        return path.parent
    if path.is_dir() and (path / "adapter_config.json").is_file():
        return path
    if path.is_file() and path.parent.is_dir():
        parent = path.parent
        if (parent / "adapter_config.json").is_file():
            return parent

    raise RuntimeError(f"Not a LoRA adapter path: {link}")


def default_hf_model_dir(model_key: str) -> Path:
    from externals.code.model_paths import get_code_profile, resolve_profile_key

    profile = get_code_profile(resolve_profile_key(model_key or "1.5b"))
    if profile.hf_instruct_dir is not None:
        return ensure_instruct_hf(key=profile.key)
    hf = profile.model_dir.parent / f"{profile.subdir}-HF"
    if hf.is_dir():
        return hf
    raise RuntimeError(
        f"HF instruct weights not found for model={model_key!r}. "
        f"Set hf_instruct_repo on profile {profile.key!r} in externals/code/model_paths.py"
    )
