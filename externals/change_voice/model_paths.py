"""RVC voice model paths under models/rvc/."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
RVC_MODELS_DIR = _REPO_ROOT / "models" / "rvc"
DEFAULT_MODEL = "MuscleMan"

_EMB_PHONE_KEY = "enc_p.emb_phone.weight"


def _version_from_config_json(model_dir: Path) -> str | None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_block = cfg.get("model") if isinstance(cfg.get("model"), dict) else cfg
    if not isinstance(model_block, dict):
        return None
    ssl_dim = model_block.get("ssl_dim")
    if ssl_dim == 256:
        return "v1"
    if ssl_dim == 768:
        return "v2"
    return None


def detect_rvc_version(pth: Path) -> str:
    """Detect RVC v1 vs v2 from a .pth checkpoint (emb_phone input width 256 vs 768)."""
    import torch

    cpt = torch.load(str(pth), map_location="cpu", weights_only=False)
    if not isinstance(cpt, dict):
        return _version_from_config_json(pth.parent) or "v2"

    weight = cpt.get("weight")
    if isinstance(weight, dict):
        tensor = weight.get(_EMB_PHONE_KEY)
        if tensor is not None and getattr(tensor, "ndim", 0) == 2:
            in_dim = int(tensor.shape[1])
            if in_dim == 256:
                return "v1"
            if in_dim == 768:
                return "v2"

    version = cpt.get("version")
    if version in ("v1", "v2"):
        return str(version)

    nested = cpt.get("model")
    if isinstance(nested, dict):
        tensor = nested.get(_EMB_PHONE_KEY)
        if tensor is not None and getattr(tensor, "ndim", 0) == 2:
            in_dim = int(tensor.shape[1])
            if in_dim == 256:
                return "v1"
            if in_dim == 768:
                return "v2"

    from_config = _version_from_config_json(pth.parent)
    if from_config:
        return from_config
    return "v2"


def resolve_rvc_version(pth: Path, requested: str = "") -> str:
    """Use explicit version=v1|v2 when set; otherwise auto-detect from checkpoint."""
    key = requested.strip().lower()
    if key in ("v1", "v2"):
        return key
    return detect_rvc_version(pth)


def require_inference_checkpoint(pth: Path, *, model_dir: Path | None = None) -> None:
    """Raise if ``pth`` is not an RVC inference checkpoint (weight + config)."""
    from externals.change_voice.ckpt_convert import (
        is_inference_checkpoint,
        is_training_checkpoint,
    )

    if is_inference_checkpoint(pth):
        return
    if is_training_checkpoint(pth):
        hint = (
            f"\n  {pth.name} is a training checkpoint (G_/D_), not an inference model.\n"
            "  Export an inference .pth in RVC-WebUI (Process ckpt → Extract small model)\n"
            "  or set AH_CHANGE_VOICE_AUTO_CONVERT=1 for classic RVC G_ checkpoints."
        )
        if model_dir is not None:
            hint += f"\n  Folder: {model_dir}"
        raise RuntimeError(f"$change_voice: invalid model file: {pth}{hint}")
    raise RuntimeError(
        f"$change_voice: {pth} is not a valid RVC inference .pth "
        f"(expected keys: weight, config, f0, version)."
    )


def resolve_model(
    model: str,
    *,
    model_path: str = "",
    index_path: str = "",
) -> tuple[Path, Path | None, str]:
    """Return ``(.pth, .index|None, display_name)``."""
    if model_path.strip():
        pth = _resolve_file(model_path.strip())
        idx: Path | None = None
        if index_path.strip():
            idx = _resolve_file(index_path.strip())
            if not idx.is_file():
                raise FileNotFoundError(
                    f"$change_voice: index_path not found: {index_path!r}"
                )
        name = pth.stem
        require_inference_checkpoint(pth)
        return pth, idx, name

    key = (model or DEFAULT_MODEL).strip()
    if not key:
        key = DEFAULT_MODEL
    model_dir = RVC_MODELS_DIR / key
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"$change_voice: model directory not found: {model_dir}\n"
            f"Place {key}.pth (+ optional .index) under models/rvc/{key}/ "
            f"or pass model_path= / index_path="
        )
    pth_files = sorted(model_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"$change_voice: no .pth in {model_dir}"
        )
    infer_pths = [
        p
        for p in pth_files
        if not p.stem.startswith(("G_", "D_"))
    ]
    training_g = [p for p in pth_files if p.stem.startswith("G_")]
    auto_convert = os.environ.get("AH_CHANGE_VOICE_AUTO_CONVERT", "1").strip().lower()
    convert_on = auto_convert not in ("0", "false", "no", "off")

    if convert_on:
        from externals.change_voice.ckpt_convert import ensure_inference_pth

        ensured = ensure_inference_pth(model_dir)
        if ensured is not None:
            pth = ensured.resolve()
        elif infer_pths:
            pth = infer_pths[0].resolve()
        elif training_g:
            raise RuntimeError(
                f"$change_voice: {model_dir} has only training checkpoint(s) "
                f"({training_g[-1].name}) and auto-conversion failed or is unsupported.\n"
                "  Place an inference .pth in this folder (from RVC-WebUI: Extract small model), "
                "or use a classic RVC G_ checkpoint that can be converted."
            )
        else:
            pth = pth_files[0].resolve()
    elif infer_pths:
        pth = infer_pths[0].resolve()
    elif training_g:
        raise RuntimeError(
            f"$change_voice: {model_dir} has only training checkpoint(s) "
            f"({training_g[-1].name}), not an inference model.\n"
            "  Export an inference .pth from your trainer, or set "
            "AH_CHANGE_VOICE_AUTO_CONVERT=1 (classic RVC G_ only)."
        )
    else:
        pth = pth_files[0].resolve()

    require_inference_checkpoint(pth, model_dir=model_dir)
    if index_path.strip():
        idx = _resolve_file(index_path.strip())
    else:
        index_files = sorted(model_dir.glob("*.index"))
        idx = index_files[0].resolve() if index_files else None
    return pth, idx, key


def _resolve_file(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    for base in (_REPO_ROOT, Path.cwd()):
        candidate = (base / raw).resolve()
        if candidate.is_file():
            return candidate
    return path.resolve()
