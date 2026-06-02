"""Named translation profiles for $translate(model='...')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

M2M100_HF_REPO = "facebook/m2m100_1.2B"
M2M100_SUBDIR = Path("m2m100_1.2B")


@dataclass(frozen=True)
class TranslateModel:
    name: str
    hf_repo: str
    subdir: Path


_profiles = [
    TranslateModel("default", M2M100_HF_REPO, M2M100_SUBDIR),
    TranslateModel("m2m100", M2M100_HF_REPO, M2M100_SUBDIR),
    TranslateModel("1.2b", M2M100_HF_REPO, M2M100_SUBDIR),
]

_by_name: dict[str, TranslateModel] = {p.name: p for p in _profiles}


def get_translate_model(name: str) -> TranslateModel:
    raw = (name or "default").strip().lower()
    if not raw or raw == "default":
        return _by_name["default"]
    if raw in _by_name:
        return _by_name[raw]
    available = ", ".join(sorted({p.name for p in _profiles if p.name != "default"}))
    raise KeyError(f"Unknown translate model {name!r}. Available: default, {available}")
