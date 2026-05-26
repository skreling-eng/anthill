"""HF Kokoro-82M clone backend (models.py + kokoro.generate), optional."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from externals.text2speech.espeak import configure_espeak
from externals.text2speech.model_paths import (
    SAMPLE_RATE,
    kokoro_root,
    lang_code_for_voice,
    resolve_model_checkpoint,
    resolve_voice_pack,
)

_MODEL: Any = None
_DEVICE: str = ""


def _load_legacy_modules(root: Path) -> tuple[Any, Any]:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "kokoro_legacy_generate",
        root / "kokoro.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {root / 'kokoro.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from models import build_model  # type: ignore[import-not-found]

    return build_model, mod.generate


def _get_model(device: str, checkpoint: Path):
    global _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE == device:
        return _MODEL
    root = kokoro_root()
    build_model, _generate = _load_legacy_modules(root)
    import torch

    dev = device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = build_model(str(checkpoint), dev)
    _DEVICE = dev
    return _MODEL


def synthesize(
    text: str,
    voice: str,
    *,
    device: str = "",
    espeak_lib: str = "",
    chunk_by: str = "sentence",
) -> np.ndarray:
    """Prototype-style: split on '.', generate per chunk, concatenate."""
    configure_espeak(espeak_lib or None)
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        EspeakWrapper = None  # type: ignore[misc, assignment]
    if EspeakWrapper is not None:
        lib = configure_espeak(espeak_lib or None)
        if lib:
            EspeakWrapper.set_library(lib)

    import torch

    root = kokoro_root()
    checkpoint = resolve_model_checkpoint(root)
    pack_path = resolve_voice_pack(voice, root)
    voice_name = pack_path.stem if isinstance(pack_path, Path) else voice
    lang = lang_code_for_voice(voice_name)

    model = _get_model(device, checkpoint)
    dev = _DEVICE
    voicepack = torch.load(str(pack_path), weights_only=True).to(dev)

    _, generate = _load_legacy_modules(root)
    body = re.sub(r"\s+", " ", text).strip()
    if not body:
        return np.zeros(0, dtype=np.float32)

    audio: list[float] = []
    if chunk_by.strip().lower() in ("sentence", ".", "period"):
        parts = body.split(".")
        for chunk in parts:
            if len(chunk.strip()) < 2:
                continue
            snippet, _ = generate(model, chunk.strip(), voicepack, lang=lang)
            audio.extend(snippet)
    else:
        snippet, _ = generate(model, body, voicepack, lang=lang)
        audio.extend(snippet)

    return np.asarray(audio, dtype=np.float32)
