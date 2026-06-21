"""$music_separation model variants (HTDemucs OpenVINO + BS-RoFormer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROFORMER_MODELS_DIR = _REPO_ROOT / "models" / "roformer"

# Default 2-stem checkpoint (BS-RoFormer-Viperx-1297 / UVR name).
ROFORMER_2STEM_CKPT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_6STEM_CKPT = "BS-Roformer-SW.ckpt"

DEFAULT_MODEL = "bs_roformer_sw"
# 2-stem vocals + instrumental (BS-RoFormer-Viperx-1297); alias: 2stem
VOCAL_STEM_MODEL = "bs_roformer_viperx_1297"

STEM_ALIASES = {
    "vocals": "vocals",
    "vocal": "vocals",
    "instrumental": "instrumental",
    "inst": "instrumental",
    "karaoke": "instrumental",
    "no_vocals": "instrumental",
    "drums": "drums",
    "drum": "drums",
    "bass": "bass",
    "other": "other",
    "others": "other",
    "guitar": "guitar",
    "piano": "piano",
}


@dataclass(frozen=True)
class SeparationVariant:
    id: str
    backend: str  # openvino | roformer
    model_filename: str
    stems: tuple[str, ...]
    label: str
    checkpoint: Path | None = None


VARIANTS: dict[str, SeparationVariant] = {
    "htdemucs_v4": SeparationVariant(
        id="htdemucs_v4",
        backend="openvino",
        model_filename="",
        stems=("drums", "bass", "other", "vocals"),
        label="HTDemucs v4 (Intel OpenVINO, Audacity plugin; use RoFormer for better quality)",
    ),
    "bs_roformer_viperx_1297": SeparationVariant(
        id="bs_roformer_viperx_1297",
        backend="roformer",
        model_filename=ROFORMER_2STEM_CKPT,
        stems=("vocals", "instrumental"),
        label="BS-RoFormer-Viperx-1297 (2-stem)",
    ),
    "bs_roformer_sw": SeparationVariant(
        id="bs_roformer_sw",
        backend="roformer",
        model_filename=ROFORMER_6STEM_CKPT,
        stems=("vocals", "drums", "bass", "guitar", "piano", "other"),
        label="BS-RoFormer-SW by jarredou (6-stem)",
    ),
}

_ALIASES: dict[str, str] = {
    "htdemucs": "htdemucs_v4",
    "demucs": "htdemucs_v4",
    "4stem": "htdemucs_v4",
    "roformer_2stem": "bs_roformer_viperx_1297",
    "2stem": "bs_roformer_viperx_1297",
    "1297": "bs_roformer_viperx_1297",
    "bs-roformer-viperx-1297": "bs_roformer_viperx_1297",
    "roformer_6stem": "bs_roformer_sw",
    "6stem": "bs_roformer_sw",
    "multi": "bs_roformer_sw",
    "bs-roformer-sw": "bs_roformer_sw",
}


def normalize_stem_name(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return STEM_ALIASES.get(key, key)


def resolve_variant(model: str, *, model_path: str = "") -> SeparationVariant:
    """Resolve model= alias; optional model_path= overrides RoFormer checkpoint file."""
    key = (model or DEFAULT_MODEL).strip().lower()
    variant_id = _ALIASES.get(key, key)
    if variant_id not in VARIANTS:
        known = ", ".join(sorted({*VARIANTS, *_ALIASES}))
        raise ValueError(
            f"$music_separation: unknown model={model!r}. "
            f"Known: {known}"
        )
    variant = VARIANTS[variant_id]
    if model_path.strip():
        ckpt = Path(model_path.strip())
        if not ckpt.suffix:
            ckpt = ROFORMER_MODELS_DIR / model_path.strip()
        if not ckpt.is_file() and not ckpt.is_absolute():
            ckpt = _REPO_ROOT / model_path.strip()
        if not ckpt.is_file():
            raise FileNotFoundError(f"$music_separation: model_path not found: {model_path!r}")
        return SeparationVariant(
            id=variant.id,
            backend="roformer",
            model_filename=ckpt.name,
            stems=variant.stems,
            label=f"{variant.label} ({ckpt.name})",
            checkpoint=ckpt.resolve(),
        )
    # User dropped BS-RoFormer-Viperx-1297.ckpt into models/roformer/
    if variant.backend == "roformer" and variant.id == "bs_roformer_viperx_1297":
        for name in (
            "BS-RoFormer-Viperx-1297.ckpt",
            ROFORMER_2STEM_CKPT,
        ):
            if (ROFORMER_MODELS_DIR / name).is_file():
                return SeparationVariant(
                    id=variant.id,
                    backend="roformer",
                    model_filename=name,
                    stems=variant.stems,
                    label=variant.label,
                )
    return variant


def list_model_ids() -> list[str]:
    return sorted(VARIANTS)
