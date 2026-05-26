"""Build ace-synth request overrides from $music external args."""

from __future__ import annotations

from externals.api import ExternalInput


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def synth_request_extras(inp: ExternalInput) -> dict:
    """Build ace-synth JSON overrides from $music(...) keyword args."""
    extras: dict = {}

    for key, raw in inp.args.items():
        if key in ("model", "count", "format", "duration", "seed", "steps"):
            continue
        if key == "bpm":
            v = _parse_optional_int(raw)
            if v is not None:
                extras["bpm"] = v
        elif key == "shift":
            v = _parse_optional_float(raw)
            if v is not None:
                extras["shift"] = v
        elif key in ("guidance_scale", "cfg", "guidance"):
            v = _parse_optional_float(raw)
            if v is not None:
                extras["guidance_scale"] = v
        elif key in ("timesignature", "time_signature"):
            extras["timesignature"] = str(raw)
        elif key in ("vocal_language", "language", "lang"):
            extras["vocal_language"] = str(raw)
        elif key == "keyscale":
            extras["keyscale"] = str(raw)
        elif key == "adapter":
            extras["adapter"] = str(raw)
        elif key == "use_cot_caption":
            extras["use_cot_caption"] = str(raw).lower() in ("1", "true", "yes")
        elif key == "inference_steps":
            extras["inference_steps"] = int(raw)
        elif key in ("lm_temperature", "temperature"):
            extras["lm_temperature"] = float(raw)
        elif key in ("lm_cfg_scale", "cfg_scale"):
            extras["lm_cfg_scale"] = float(raw)
        elif key in ("lm_top_p", "top_p"):
            extras["lm_top_p"] = float(raw)

    return extras


def merge_synth_extras(inp: ExternalInput, model_name: str) -> dict:
    """Model registry defaults, then $music(...) keyword overrides."""
    from externals.music.model_list import get_music_model

    merged = get_music_model(model_name).synth_request_defaults()
    merged.update(synth_request_extras(inp))
    return merged


def default_steps(inp: ExternalInput, model_name: str) -> int:
    if "steps" in inp.args:
        return int(inp.args["steps"])
    if "inference_steps" in inp.args:
        return int(inp.args["inference_steps"])
    from externals.music.model_list import get_music_model

    return get_music_model(model_name).steps
