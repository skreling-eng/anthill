"""Paths and voice resolution for $text2speech (Kokoro)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
KOKORO_DIR = _REPO_ROOT / "models" / "kokoro"
DEFAULT_VOICE = "af_bella"
DEFAULT_MODEL_FILE = "kokoro-v0_19.pth"
SAMPLE_RATE = 24_000

# Voices from Kokoro-82M / user prototype (lang prefix: a=US, b=UK).
KNOWN_VOICES = frozenset(
    {
        "af",
        "af_bella",
        "af_sarah",
        "af_nicole",
        "af_sky",
        "af_heart",
        "am_adam",
        "am_michael",
        "am_liam",
        "am_onyx",
        "am_puck",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
        "bm_daniel",
        "bm_fable",
    }
)


def kokoro_root() -> Path:
    raw = os.environ.get("AH_KOKORO_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    return KOKORO_DIR.resolve()


def legacy_available(root: Path | None = None) -> bool:
    """True when HF-style Kokoro-82M clone is present (models.py + kokoro.py)."""
    base = root or kokoro_root()
    return (base / "models.py").is_file() and (base / "kokoro.py").is_file()


def use_legacy_backend() -> bool:
    raw = os.environ.get("AH_TEXT2SPEECH_BACKEND", "").strip().lower()
    if raw in ("legacy", "v0", "hf"):
        return True
    if raw in ("pipeline", "kokoro", "modern"):
        return False
    return legacy_available()


def lang_code_for_voice(voice: str) -> str:
    """First letter of voice id: a=American English, b=British English."""
    name = Path(voice).stem if voice.endswith(".pt") else voice
    if not name:
        return "a"
    return name[0].lower()


def resolve_voice_pack(
    voice: str,
    root: Path | None = None,
    *,
    repo_id: str = "hexgrad/Kokoro-82M",
    download: bool = True,
) -> Path:
    """Resolve voices/<name>.pt under models/kokoro; download there if missing."""
    base = root or kokoro_root()
    v = voice.strip()
    if not v:
        v = DEFAULT_VOICE
    path = Path(v)
    if path.is_file() and path.suffix.lower() == ".pt":
        return path.resolve()
    if v.endswith(".pt"):
        for candidate in (base / v, base / "voices" / Path(v).name):
            if candidate.is_file():
                return candidate.resolve()
        if not download:
            raise FileNotFoundError(
                f"$text2speech: voice pack not found: {v} (looked under {base})"
            )
        from externals.text2speech.assets import ensure_voice_pack

        return ensure_voice_pack(Path(v).stem, repo_id=repo_id, root=base)
    local = base / "voices" / f"{v}.pt"
    if local.is_file():
        return local.resolve()
    if not download:
        raise FileNotFoundError(
            f"$text2speech: voice pack not found: {local} (set AH_KOKORO_DIR or download)"
        )
    from externals.text2speech.assets import ensure_voice_pack

    return ensure_voice_pack(v, repo_id=repo_id, root=base)


def resolve_model_checkpoint(root: Path | None = None) -> Path:
    base = root or kokoro_root()
    raw = os.environ.get("AH_KOKORO_MODEL", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"$text2speech: model not found: {path}")
        return path.resolve()
    for name in (DEFAULT_MODEL_FILE, "kokoro-v0_19.pth"):
        candidate = base / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"$text2speech: no Kokoro checkpoint in {base}. "
        f"Place {DEFAULT_MODEL_FILE} there or set AH_KOKORO_MODEL=."
    )
