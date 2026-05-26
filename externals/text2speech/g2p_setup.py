"""Ensure misaki/spacy G2P models for Kokoro English pipelines."""

from __future__ import annotations

_SPACY_EN_SM = "en_core_web_sm"
_SPACY_EN_TRF = "en_core_web_trf"


def ensure_spacy_model(name: str = _SPACY_EN_SM) -> None:
    """Load or download a spaCy model required by misaki.en.G2P."""
    import spacy

    try:
        spacy.load(name)
        return
    except OSError:
        pass
    print(f"$text2speech: downloading spaCy model {name!r}", flush=True)
    try:
        from spacy.cli import download

        download(name)
        spacy.load(name)
        return
    except Exception as exc:
        raise RuntimeError(
            f"$text2speech: spaCy model {name!r} is required for English G2P.\n"
            f"  .venvs/text2speech/Scripts/python.exe -m spacy download {name}\n"
            f"  or: powershell -File tools\\setup_external_venvs.ps1"
        ) from exc


def ensure_english_g2p(*, trf: bool = False) -> None:
    ensure_spacy_model(_SPACY_EN_TRF if trf else _SPACY_EN_SM)
