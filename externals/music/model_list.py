"""$music model registry (GGUF/ace-synth or safetensors/native)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from externals.image.model_paths import models_roots
from externals.music.model_paths import ace_step_dir, resolve_dit_gguf

WeightsFormat = Literal["gguf", "safetensors"]

DEFAULT_MUSIC_MODEL = "st"


@dataclass
class MusicModel:
    name: str
    weights: WeightsFormat = "gguf"
    models_root: str = "ace-step-1.5"
    dit_gguf: str = ""
    api_model: str | None = None
    config_path: str = "acestep-v15-xl-base"
    duration: float = 30.0
    steps: int = 50
    adapter: str | None = None
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""
    vocal_language: str = ""
    guidance_scale: float | None = None
    shift: float | None = None
    lm_temperature: float | None = None
    lm_cfg_scale: float | None = None
    lm_top_p: float | None = None
    use_cot_caption: bool = False
    extra_synth: dict[str, Any] = field(default_factory=dict)

    def models_dir(self) -> Path:
        for root in models_roots():
            candidate = root / self.models_root
            if candidate.is_dir():
                return candidate
        return models_roots()[0] / self.models_root

    def preferred_backend(self) -> str:
        return "native" if self.weights == "safetensors" else "synth"

    def synth_request_defaults(self) -> dict[str, Any]:
        req: dict[str, Any] = {"use_cot_caption": self.use_cot_caption}
        if self.bpm is not None and self.bpm > 0:
            req["bpm"] = self.bpm
        if self.keyscale:
            req["keyscale"] = self.keyscale
        if self.timesignature:
            req["timesignature"] = self.timesignature
        if self.vocal_language:
            req["vocal_language"] = self.vocal_language
        if self.guidance_scale is not None:
            req["guidance_scale"] = self.guidance_scale
        if self.shift is not None:
            req["shift"] = self.shift
        if self.lm_temperature is not None:
            req["lm_temperature"] = self.lm_temperature
        if self.lm_cfg_scale is not None:
            req["lm_cfg_scale"] = self.lm_cfg_scale
        if self.lm_top_p is not None:
            req["lm_top_p"] = self.lm_top_p
        if self.steps > 0:
            req["inference_steps"] = self.steps
        req.update(self.extra_synth)
        return req

    def native_generation_defaults(self) -> dict[str, Any]:
        """Fields for acestep.inference.GenerationParams."""
        out: dict[str, Any] = {
            "duration": self.duration,
            "thinking": False,
            "use_cot_metas": False,
            "use_cot_caption": False,
            "use_cot_language": False,
        }
        if self.bpm is not None and self.bpm > 0:
            out["bpm"] = self.bpm
        if self.keyscale:
            out["keyscale"] = self.keyscale
        if self.timesignature:
            out["timesignature"] = self.timesignature
        if self.vocal_language:
            out["vocal_language"] = self.vocal_language
        if self.guidance_scale is not None:
            out["guidance_scale"] = self.guidance_scale
        if self.shift is not None:
            out["shift"] = self.shift
        if self.steps > 0:
            out["inference_steps"] = self.steps
        return out


_MODELS: dict[str, MusicModel] = {
    "default": MusicModel(
        name="default",
        dit_gguf="ace-step-1.5/acestep-v15-xl-turbo-BF16.gguf",
        api_model="acestep-v15-xl-turbo",
        config_path="acestep-v15-xl-turbo",
        steps=20,
    ),
    "st": MusicModel(
        name="st",
        weights="safetensors",
        models_root="ace-step-1.5_st",
        config_path="acestep-v15-turbo",
        api_model="acestep-v15-turbo",
        duration=170.0,
        steps=20,
        bpm=190,
        timesignature="4",
        vocal_language="en",
        keyscale="D major",
        guidance_scale=1.0,
        shift=3.0,
        lm_cfg_scale=2.0,
        lm_temperature=0.90,
        lm_top_p=0.90,
    ),
    "st_temp_07": MusicModel(
        name="st",
        weights="safetensors",
        models_root="ace-step-1.5_st",
        config_path="acestep-v15-turbo",
        api_model="acestep-v15-turbo",
        duration=170.0,
        steps=20,
        bpm=190,
        timesignature="4",
        vocal_language="en",
        keyscale="D major",
        guidance_scale=1.0,
        shift=3.0,
        lm_cfg_scale=2.0,
        lm_temperature=0.70,
        lm_top_p=0.90,
    ),
    "st_top_95": MusicModel(
        name="st",
        weights="safetensors",
        models_root="ace-step-1.5_st",
        config_path="acestep-v15-turbo",
        api_model="acestep-v15-turbo",
        duration=170.0,
        steps=20,
        bpm=190,
        timesignature="4",
        vocal_language="en",
        keyscale="D major",
        guidance_scale=1.0,
        shift=3.0,
        lm_cfg_scale=2.0,
        lm_temperature=0.90,
        lm_top_p=0.95,
    ),
    "st_cfg_scale_3": MusicModel(
        name="st",
        weights="safetensors",
        models_root="ace-step-1.5_st",
        config_path="acestep-v15-turbo",
        api_model="acestep-v15-turbo",
        duration=170.0,
        steps=20,
        bpm=190,
        timesignature="4",
        vocal_language="en",
        keyscale="D major",
        guidance_scale=1.0,
        shift=3.0,
        lm_cfg_scale=2.0,
        lm_temperature=0.90,
        lm_top_p=0.90,
    ),
    "xl-base": MusicModel(
        name="xl-base",
        dit_gguf="ace-step-1.5/acestep-v15-xl-base-BF16.gguf",
        api_model="acestep-v15-xl-base",
        steps=50,
    ),
    "xl-turbo": MusicModel(
        name="xl-turbo",
        dit_gguf="ace-step-1.5/acestep-v15-xl-turbo-BF16.gguf",
        api_model="acestep-v15-xl-turbo",
        config_path="acestep-v15-xl-turbo",
        steps=8,
    ),
    "turbo": MusicModel(
        name="turbo",
        dit_gguf="ace-step-1.5/acestep-v15-turbo-BF16.gguf",
        api_model="acestep-v15-turbo",
        config_path="acestep-v15-turbo",
        steps=8,
    ),
}


def get_music_model(name: str) -> MusicModel:
    if name in _MODELS:
        return _MODELS[name]
    if not name:
        return _MODELS[DEFAULT_MUSIC_MODEL]
    if name == "default":
        return _MODELS["default"]
    available = ", ".join(sorted(_MODELS))
    raise KeyError(f"Unknown music model {name!r}. Available: {available}")


def dit_path_for_model(model: MusicModel) -> str:
    if model.weights != "gguf":
        raise ValueError(f"Model {model.name!r} uses safetensors, not GGUF")
    return str(resolve_dit_gguf(model.dit_gguf))


def resolve_adapter_stem(name: str | None, *, models_dir: Path | None = None) -> str | None:
    if not name or not str(name).strip():
        return None
    stem = Path(name).stem
    folder = (models_dir or ace_step_dir()) / "adapters"
    if not folder.is_dir():
        return None
    for path in folder.iterdir():
        if path.is_file() and path.stem == stem:
            return stem
    return None


def adapter_for_model(model: MusicModel, override: str | None = None) -> str | None:
    if model.weights != "gguf":
        return None
    return resolve_adapter_stem(
        override if override is not None else model.adapter,
        models_dir=model.models_dir(),
    )
