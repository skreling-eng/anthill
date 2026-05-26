"""ACE-Step model paths (GGUF for ace-synth; safetensors for native PyTorch)."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from externals.image.model_paths import models_roots, resolve_model_path

_DEFAULT_DIR = "ace-step-1.5"
_DEFAULT_DIT = "acestep-v15-xl-base-BF16.gguf"
_DEFAULT_VAE = "vae-BF16.gguf"
_DEFAULT_EMBEDDING = "Qwen3-Embedding-0.6B-BF16.gguf"
_HF_GGUF_REPO = "Serveurperso/ACE-Step-1.5-GGUF"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ComfyUI split-file names (audio_ace_step_1_5_split_002.json)
_COMFY_DIT = "diffusion_models/acestep_v1.5_turbo.safetensors"
_COMFY_VAE = "vae/ace_1.5_vae.safetensors"
_COMFY_TEXT = "text_encoders/qwen_0.6b_ace15.safetensors"


def ace_step_dir(subdir: str = _DEFAULT_DIR) -> Path:
    for root in models_roots():
        candidate = root / subdir
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / subdir


def safetensors_models_dir(subdir: str) -> Path:
    return ace_step_dir(subdir)


def comfy_safetensors_layout(folder: Path) -> dict[str, Path]:
    """Detect Comfy-style split safetensors under folder."""
    found: dict[str, Path] = {}
    for rel, key in (
        (_COMFY_DIT, "dit"),
        (_COMFY_VAE, "vae"),
        (_COMFY_TEXT, "text"),
    ):
        path = folder / rel
        if path.is_file():
            found[key] = path
    if not found:
        for path in folder.rglob("*.safetensors"):
            low = path.as_posix().lower()
            if "vae" in low:
                found.setdefault("vae", path)
            elif "qwen" in low or "embedding" in low or "text_encoder" in low:
                found.setdefault("text", path)
            elif "turbo" in low or "dit" in low or "diffusion" in low:
                found.setdefault("dit", path)
    return found


def safetensors_stack_ready(folder: Path) -> bool:
    """True if folder looks like it has DiT+VAE+text safetensors (Comfy or HF layout)."""
    if comfy_safetensors_layout(folder):
        hits = comfy_safetensors_layout(folder)
        if len(hits) >= 2:
            return True
    for name in ("acestep-v15-turbo", "acestep-v15-xl-turbo", "acestep-v15-turbo"):
        if (folder / name).is_dir():
            return True
    return any(folder.rglob("*.safetensors"))


def _scan_ace_step_dir(folder: Path | None = None) -> dict[str, Path]:
    folder = folder or ace_step_dir()
    found: dict[str, Path] = {}
    if not folder.is_dir():
        return found
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".gguf":
            found[path.name.lower()] = path
    return found


def _find_gguf(filename: str, folder: Path | None = None) -> Path | None:
    env_map = {
        _DEFAULT_DIT: "ACESTEP_DIT_GGUF",
        _DEFAULT_VAE: "ACESTEP_VAE_GGUF",
        _DEFAULT_EMBEDDING: "ACESTEP_EMBEDDING_GGUF",
    }
    env_key = env_map.get(filename)
    if env_key and folder is None:
        raw = os.environ.get(env_key, "").strip()
        if raw:
            path = Path(resolve_model_path(raw))
            if path.is_file():
                return path

    scan_root = folder or ace_step_dir()
    hit = _scan_ace_step_dir(scan_root).get(filename.lower())
    if hit:
        return hit

    for root in models_roots():
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return path
        except OSError:
            continue
    return None


def resolve_gguf_stack_in(folder: Path) -> tuple[Path, Path, Path]:
    dit_name = Path(_DEFAULT_DIT).name
    dit = _find_gguf(dit_name, folder)
    if not dit:
        for path in folder.glob("*.gguf"):
            low = path.name.lower()
            if "vae" not in low and "embedding" not in low and "qwen" not in low:
                dit = path
                break
    if not dit:
        raise FileNotFoundError(f"No DiT GGUF in {folder}")
    vae = _find_gguf(_DEFAULT_VAE, folder)
    if not vae:
        raise FileNotFoundError(_missing_file_message(_DEFAULT_VAE, "ACESTEP_VAE_GGUF", folder))
    emb = _find_gguf(_DEFAULT_EMBEDDING, folder)
    if not emb:
        raise FileNotFoundError(
            _missing_file_message(_DEFAULT_EMBEDDING, "ACESTEP_EMBEDDING_GGUF", folder)
        )
    return dit, emb, vae


def resolve_dit_gguf(ref: str | None = None) -> Path:
    if ref:
        path = Path(resolve_model_path(ref))
        if path.is_file():
            return path
        raise FileNotFoundError(f"ACE-Step DiT GGUF not found: {path}")
    found = _find_gguf(_DEFAULT_DIT)
    if found:
        return found
    raise FileNotFoundError(f"ACE-Step DiT GGUF not found in {ace_step_dir()}")


def resolve_vae_gguf() -> Path:
    found = _find_gguf(_DEFAULT_VAE)
    if found:
        return found
    raise FileNotFoundError(_missing_file_message(_DEFAULT_VAE, "ACESTEP_VAE_GGUF"))


def resolve_embedding_gguf() -> Path:
    found = _find_gguf(_DEFAULT_EMBEDDING)
    if found:
        return found
    raise FileNotFoundError(
        _missing_file_message(_DEFAULT_EMBEDDING, "ACESTEP_EMBEDDING_GGUF")
    )


def resolve_gguf_stack() -> tuple[Path, Path, Path]:
    return resolve_dit_gguf(), resolve_vae_gguf(), resolve_embedding_gguf()


def _missing_file_message(
    filename: str, env_var: str, folder: Path | None = None
) -> str:
    folder = folder or ace_step_dir()
    present = sorted(p.name for p in folder.glob("**/*")) if folder.is_dir() else []
    listing = ", ".join(present[:20]) if present else "(none)"
    return (
        f"{filename} not found in {folder}\n"
        f"  Present files: {listing}\n"
        f"  Place {filename} in that folder, or set {env_var}."
    )


def _download_hf_file(filename: str, dest: Path) -> None:
    url = f"https://huggingface.co/{_HF_GGUF_REPO}/resolve/main/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"$music downloading {filename} from Hugging Face…")
    with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    print(f"$music saved {dest}")


def ensure_companion_gguf() -> None:
    from externals.music.models_env import configure_models_environment

    configure_models_environment()
    if os.environ.get("ACESTEP_DOWNLOAD_MISSING", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    folder = ace_step_dir()
    for name in (_DEFAULT_VAE, _DEFAULT_EMBEDDING):
        dest = folder / name
        if not dest.is_file():
            _download_hf_file(name, dest)


def gguf_stack_ready() -> bool:
    try:
        resolve_gguf_stack()
        return True
    except FileNotFoundError:
        return False


def synth_bin_candidates() -> list[Path]:
    from externals.music.ace_bin import synth_bin_candidates as _candidates

    return _candidates()
