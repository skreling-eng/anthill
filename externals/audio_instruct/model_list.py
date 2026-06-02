"""Named audio-instruction profiles for $audio_instruct(model='...')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QWEN2_AUDIO_4BIT_HF_REPO = "alicekyting/Qwen2-Audio-7B-Instruct-4bit"
QWEN2_AUDIO_4BIT_SUBDIR = Path("qwen-audio") / "Qwen2-Audio-7B-Instruct-4bit"


@dataclass(frozen=True)
class AudioInstructModel:
    name: str
    hf_repo: str
    subdir: Path


_profiles = [
    AudioInstructModel("default", QWEN2_AUDIO_4BIT_HF_REPO, QWEN2_AUDIO_4BIT_SUBDIR),
    AudioInstructModel("4bit", QWEN2_AUDIO_4BIT_HF_REPO, QWEN2_AUDIO_4BIT_SUBDIR),
    AudioInstructModel("qwen2-audio", QWEN2_AUDIO_4BIT_HF_REPO, QWEN2_AUDIO_4BIT_SUBDIR),
]

_by_name: dict[str, AudioInstructModel] = {p.name: p for p in _profiles}


def get_audio_instruct_model(name: str) -> AudioInstructModel:
    raw = (name or "default").strip().lower()
    if not raw or raw == "default":
        return _by_name["default"]
    if raw in _by_name:
        return _by_name[raw]
    available = ", ".join(sorted({p.name for p in _profiles if p.name != "default"}))
    raise KeyError(
        f"Unknown audio_instruct model {name!r}. Available: default, {available}"
    )
