"""Named SigLIP 2 profiles for $image2embedding(model='...')."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SIGLIP2_BASE_384_HF = "google/siglip2-base-patch16-384"
SIGLIP2_BASE_224_HF = "google/siglip2-base-patch16-224"
SIGLIP2_BASE_384_SUBDIR = Path("siglip2") / "google-siglip2-base-patch16-384"
SIGLIP2_BASE_224_SUBDIR = Path("siglip2") / "google-siglip2-base-patch16-224"


@dataclass(frozen=True)
class Image2EmbeddingModel:
    name: str
    hf_repo: str
    subdir: Path
    source_dim: int = 768

    def dir_name(self) -> str:
        return self.subdir.name


_profiles = [
    Image2EmbeddingModel("default", SIGLIP2_BASE_384_HF, SIGLIP2_BASE_384_SUBDIR),
    Image2EmbeddingModel("base", SIGLIP2_BASE_384_HF, SIGLIP2_BASE_384_SUBDIR),
    Image2EmbeddingModel("384", SIGLIP2_BASE_384_HF, SIGLIP2_BASE_384_SUBDIR),
    Image2EmbeddingModel("224", SIGLIP2_BASE_224_HF, SIGLIP2_BASE_224_SUBDIR),
    Image2EmbeddingModel("base-224", SIGLIP2_BASE_224_HF, SIGLIP2_BASE_224_SUBDIR),
]

_by_name: dict[str, Image2EmbeddingModel] = {p.name: p for p in _profiles}


def get_image2embedding_model(name: str) -> Image2EmbeddingModel:
    raw = (name or "default").strip().lower()
    if not raw or raw == "default":
        return _by_name["default"]
    if raw in _by_name:
        return _by_name[raw]
    available = ", ".join(
        sorted({p.name for p in _profiles if p.name not in ("default", "384")})
    )
    raise KeyError(
        f"Unknown image2embedding model {name!r}. Available: default, {available}"
    )
