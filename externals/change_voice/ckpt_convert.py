"""Convert RVC training checkpoints (G_*.pth) to rvc-python inference format."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import torch

from externals.change_voice.model_paths import detect_rvc_version

# Applio / so-vits style training (not compatible with rvc_python infer_pack).
_NONSTANDARD_MARKERS = (
    "enc_p.enc_.ffn_layers",
    "enc_p.enc_.norm_layers",
)


def is_inference_checkpoint(path: Path) -> bool:
    try:
        cpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(cpt, dict) and "weight" in cpt and isinstance(cpt["weight"], dict)


def is_training_checkpoint(path: Path) -> bool:
    if path.stem.startswith("D_"):
        return False
    try:
        cpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(cpt, dict) and "model" in cpt and isinstance(cpt["model"], dict)


def _is_nonstandard_training(state: dict) -> bool:
    keys = list(state.keys())
    return any(marker in k for k in keys for marker in _NONSTANDARD_MARKERS)


def _remap_training_keys(state: dict) -> dict:
    """Map enc_p.enc_.* → enc_p.encoder.* for older RVC training checkpoints."""
    out: dict = {}
    for key, tensor in state.items():
        new_key = key.replace("enc_p.enc_.", "enc_p.encoder.")
        out[new_key] = tensor
    return out


def _fuse_weight_norm(state: dict) -> dict:
    """Merge weight_g / weight_v parametrization into plain .weight tensors."""
    out: dict = {}
    weight_g: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            weight_g[base] = tensor
            continue
        if key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            g = weight_g.pop(base, None)
            v = tensor
            if g is not None:
                norm = torch.norm(v)
                if norm > 0:
                    fused = g * v / norm
                else:
                    fused = v.clone()
                out[f"{base}.weight"] = fused
            continue
        if key.endswith(".weight_g") or key.endswith(".weight_v"):
            continue
        out[key] = tensor

    return out


def _sr_label_from_hz(hz: int) -> str:
    if hz >= 46000:
        return "48k"
    if hz >= 38000:
        return "40k"
    return "32k"


def _config_list(sr: str, version: str, spk_embed_dim: int) -> list:
    """RVC WebUI extract_small_model config tables (process_ckpt.py)."""
    if sr == "40k":
        return [
            1025,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2],
            512,
            [16, 16, 4, 4],
            spk_embed_dim,
            256,
            40000,
        ]
    if sr == "48k":
        if version == "v1":
            return [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 6, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                spk_embed_dim,
                256,
                48000,
            ]
        return [
            1025,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [12, 10, 2, 2],
            512,
            [24, 20, 4, 4],
            spk_embed_dim,
            256,
            48000,
        ]
    if version == "v1":
        return [
            513,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 4, 2, 2, 2],
            512,
            [16, 16, 4, 4, 4],
            spk_embed_dim,
            256,
            32000,
        ]
    return [
        513,
        32,
        192,
        192,
        768,
        2,
        6,
        3,
        0,
        "1",
        [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [10, 8, 2, 2],
        512,
        [20, 16, 4, 4],
        spk_embed_dim,
        256,
        32000,
    ]


def _read_training_meta(model_dir: Path, training_pth: Path) -> tuple[str, str, int, int]:
    """Return (version, sr_label, spk_embed_dim, iteration)."""
    version = detect_rvc_version(training_pth)
    sr_hz = 48000
    spk = 109
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            model_block = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
            sr_hz = int(cfg.get("data", {}).get("sampling_rate", sr_hz))
            spk = int(model_block.get("n_speakers", spk))
            ssl = model_block.get("ssl_dim")
            if ssl == 256:
                version = "v1"
            elif ssl == 768:
                version = "v2"
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return version, _sr_label_from_hz(sr_hz), spk


def convert_training_checkpoint(
    training_pth: Path,
    output_pth: Path,
    *,
    model_dir: Path | None = None,
    if_f0: int = 1,
    info: str = "",
) -> Path:
    """
    Convert a classic RVC ``G_*.pth`` training checkpoint to inference ``.pth``.

    Raises ``RuntimeError`` for non-standard training stacks (e.g. Applio ffn_layers).
    """
    model_dir = model_dir or training_pth.parent
    raw = torch.load(str(training_pth), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "model" not in raw:
        raise RuntimeError(f"$change_voice: not a training checkpoint: {training_pth}")

    state = dict(raw["model"])
    if _is_nonstandard_training(state):
        raise RuntimeError(
            f"$change_voice: {training_pth.name} uses a non-standard RVC training "
            f"format (e.g. Applio) that cannot be auto-converted for rvc-python.\n"
            f"Export an inference .pth from the trainer (RVC-WebUI: Process ckpt → "
            f"Extract small model), then place it in {model_dir}/\n"
            f"Expected keys like weight/config/f0/version — not G_*.pth alone."
        )

    iteration = int(raw.get("iteration", 0) or 0)
    version, sr, spk = _read_training_meta(model_dir, training_pth)

    state = _remap_training_keys(state)
    if any(".weight_g" in k for k in state):
        state = _fuse_weight_norm(state)

    opt: OrderedDict = OrderedDict()
    opt["weight"] = OrderedDict()
    for key, tensor in state.items():
        if "enc_q" in key:
            continue
        opt["weight"][key] = tensor.half() if tensor.is_floating_point() else tensor

    opt["config"] = _config_list(sr, version, spk)
    opt["info"] = info or (f"{iteration}iter" if iteration else "Extracted model.")
    opt["version"] = version
    opt["sr"] = sr
    opt["f0"] = int(if_f0)

    output_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(opt, str(output_pth))
    return output_pth


def ensure_inference_pth(
    model_dir: Path,
    *,
    training_pth: Path | None = None,
    force: bool = False,
) -> Path | None:
    """
    If ``model_dir`` only has ``G_*.pth``, convert to ``{name}_inference.pth``.

    Returns the inference path, or None if no conversion was needed.
    """
    name = model_dir.name
    infer_candidates = [
        p
        for p in model_dir.glob("*.pth")
        if not p.stem.startswith(("G_", "D_"))
    ]
    if infer_candidates and not force:
        return max(infer_candidates, key=lambda p: p.stat().st_mtime)

    g_files = sorted(model_dir.glob("G_*.pth"), key=lambda p: p.stat().st_mtime)
    if not g_files:
        return None
    src = training_pth or g_files[-1]
    if not is_training_checkpoint(src):
        return None

    out = model_dir / f"{_safe_name(name)}_inference.pth"
    if out.is_file() and not force:
        if is_inference_checkpoint(out):
            return out
        out.unlink(missing_ok=True)

    print(
        f"$change_voice: converting training checkpoint {src.name} → {out.name}",
        flush=True,
    )
    convert_training_checkpoint(src, out, model_dir=model_dir)
    return out


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name) or "model"
