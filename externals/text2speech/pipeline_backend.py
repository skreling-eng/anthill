"""Kokoro KPipeline synthesis (pip package kokoro>=0.9.4)."""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np

from externals.text2speech.espeak import configure_espeak
from externals.text2speech.assets import ensure_model_assets
from externals.text2speech.g2p_setup import ensure_english_g2p
from externals.text2speech.model_paths import (
    SAMPLE_RATE,
    lang_code_for_voice,
    resolve_voice_pack,
)

_PIPELINES: dict[tuple[str, str, str], object] = {}
_SPLIT_PATTERNS = {
    "sentence": r"(?<=[.!?…])\s+",
    "paragraph": r"\n\s*\n+",
    "newline": r"\n+",
    "none": "",
}


def _split_pattern(mode: str) -> str | None:
    key = (mode or "sentence").strip().lower()
    if key in ("off", "none", "false", "0"):
        return None
    pat = _SPLIT_PATTERNS.get(key, _SPLIT_PATTERNS["sentence"])
    return pat or None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _resolve_device(device: str) -> str | None:
    dev = device.strip().lower()
    if dev in ("cuda", "gpu"):
        return "cuda"
    if dev.startswith("cuda:"):
        return "cuda"
    if dev in ("cpu", "mps"):
        return dev
    if not dev or dev == "auto":
        return None
    return dev


def _get_pipeline(*, lang_code: str, device: str, repo_id: str):
    from kokoro import KPipeline
    from kokoro.model import KModel

    key = (lang_code, device, repo_id)
    if key not in _PIPELINES:
        if lang_code in "ab":
            ensure_english_g2p()
        dev = _resolve_device(device)
        config_path, model_path = ensure_model_assets(repo_id)
        kmodel = KModel(
            repo_id=repo_id,
            config=str(config_path),
            model=str(model_path),
        )
        if dev:
            kmodel = kmodel.to(dev).eval()
        _PIPELINES[key] = KPipeline(
            lang_code=lang_code,
            repo_id=repo_id,
            model=kmodel,
            device=dev,
        )
    return _PIPELINES[key]


def _audio_chunks(
    pipeline,
    text: str,
    voice: str | object,
    *,
    speed: float,
    split_mode: str,
) -> Iterator[np.ndarray]:
    pattern = _split_pattern(split_mode)
    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=pattern,
    )
    for result in generator:
        audio = result.audio
        if audio is None:
            continue
        arr = audio.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.size:
            yield arr.astype(np.float32)


def synthesize(
    text: str,
    voice: str,
    *,
    device: str = "",
    speed: float = 1.0,
    split: str = "sentence",
    repo_id: str = "hexgrad/Kokoro-82M",
    espeak_lib: str = "",
    join_chunks: bool = True,
) -> np.ndarray:
    configure_espeak(espeak_lib or None)
    body = _normalize_text(text)
    if not body:
        return np.zeros(0, dtype=np.float32)

    lang = lang_code_for_voice(voice)
    pipeline = _get_pipeline(lang_code=lang, device=device, repo_id=repo_id)
    pack = str(resolve_voice_pack(voice, repo_id=repo_id))
    chunks = list(
        _audio_chunks(
            pipeline,
            body,
            pack,
            speed=speed,
            split_mode=split,
        )
    )
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if join_chunks or len(chunks) == 1:
        return np.concatenate(chunks)
    return chunks[0]


def write_wav(path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sample_rate)
