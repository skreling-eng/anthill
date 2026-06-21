"""Named vision-language profiles for $image2text(model='...')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QWEN2_HF_REPO = "Qwen/Qwen2-VL-2B-Instruct"
QWEN3_HF_REPO = "Qwen/Qwen3-VL-8B-Instruct"
BLIP2_HF_REPO = "Salesforce/blip2-opt-2.7b"
QWEN2_SUBDIR = Path("qwen-vl") / "Qwen2-VL-2B-Instruct"
QWEN3_SUBDIR = Path("qwen-vl") / "Qwen3-VL-8B-Instruct"
BLIP2_SUBDIR = Path("blip2") / "blip2-opt-2.7b"


@dataclass(frozen=True)
class Image2TextModel:
    name: str
    hf_repo: str
    subdir: Path
    family: str  # "qwen2" | "qwen3" | "blip2"

    def dir_name(self) -> str:
        return self.subdir.name


_profiles = [
    Image2TextModel("qwen2", QWEN2_HF_REPO, QWEN2_SUBDIR, "qwen2"),
    Image2TextModel("default", QWEN2_HF_REPO, QWEN2_SUBDIR, "qwen2"),
    Image2TextModel("2b", QWEN2_HF_REPO, QWEN2_SUBDIR, "qwen2"),
    Image2TextModel("qwen3", QWEN3_HF_REPO, QWEN3_SUBDIR, "qwen3"),
    Image2TextModel("8b", QWEN3_HF_REPO, QWEN3_SUBDIR, "qwen3"),
    Image2TextModel("qwen3-8b", QWEN3_HF_REPO, QWEN3_SUBDIR, "qwen3"),
    Image2TextModel("blip2", BLIP2_HF_REPO, BLIP2_SUBDIR, "blip2"),
    Image2TextModel("blip2-opt", BLIP2_HF_REPO, BLIP2_SUBDIR, "blip2"),
    Image2TextModel("blip2-opt-2.7b", BLIP2_HF_REPO, BLIP2_SUBDIR, "blip2"),
]

_by_name: dict[str, Image2TextModel] = {p.name: p for p in _profiles}


def get_image2text_model(name: str) -> Image2TextModel:
    raw = (name or "qwen2").strip().lower()
    if not raw or raw == "default":
        return _by_name["qwen2"]
    if raw in _by_name:
        return _by_name[raw]
    available = ", ".join(sorted({p.name for p in _profiles if p.name not in ("default", "2b", "8b")}))
    raise KeyError(f"Unknown image2text model {name!r}. Available: {available}")
