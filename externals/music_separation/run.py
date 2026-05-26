"""$music_separation — split music into stems (HTDemucs v4 or BS-RoFormer)."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.music_separation.audio_io import (
    load_stereo,
    resample_to_native,
    write_wav_bytes,
)
from externals.music_separation.model_paths import (
    CACHE_DIR,
    configure_models_environment,
    ensure_model,
)
from externals.music_separation.models import (
    DEFAULT_MODEL,
    ROFORMER_MODELS_DIR,
    SeparationVariant,
    resolve_variant,
)
from externals.music_separation.pipeline import DEFAULT_SHIFTS, get_compiled, require_openvino, separate
from externals.music_separation.roformer import separate_file as roformer_separate
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$music_separation needs sounds[] in the input bundle.

Example:
  @track: $file('song.wav')
  @stems: @track -> $music_separation -> $save('stems/')

Optional args:
  model=bs_roformer_sw (default) — 6-stem BS-RoFormer (vocals, drums, bass, guitar, piano, other)
  model=@models              — run each model from @models texts[] (via $list)
  model=htdemucs_v4          — 4-stem HTDemucs v4 OpenVINO (Audacity plugin; lower quality than RoFormer)
  model=bs_roformer_viperx_1297 — 2-stem BS-RoFormer (vocals + instrumental)
  model_path=...             — local .ckpt for RoFormer variants
  device=CPU, shifts=2       — OpenVINO HTDemucs only

AH_EMULATE_MUSIC_SEPARATION=1 for stub output without models.
"""


def _output_name(model_id: str, stem: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w.-]", "_", model_id)
    return f"{ts}_{safe_model}_{stem}_{index}.wav"


def _int_arg(inp: ExternalInput, key: str, default: int, *, min_value: int = 0) -> int:
    raw = inp.args.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"$music_separation: invalid {key}={raw!r}") from exc
    if value < min_value:
        raise ValueError(f"$music_separation: {key} must be >= {min_value}, got {value}")
    return value


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    sounds: list[str],
    models: list[str],
) -> None:
    for model_name in models:
        variant = resolve_variant(model_name)
        for sound in sounds:
            for stem in variant.stems:
                content = (
                    f"[emulated $music_separation model={model_name} stem={stem}]\n"
                    f"from: {sound}\n"
                )
                link = ctx.new_link("sounds", ".wav", content)
                out.sounds.append(link)


def _run_htdemucs(
    ctx: ExternalContext,
    sounds: list[str],
    variant: SeparationVariant,
    *,
    model_id: str,
    device: str,
    shifts: int,
) -> list[str]:
    configure_models_environment()
    require_openvino()
    model_xml = ensure_model()
    compiled = get_compiled(model_xml, device, cache_dir=CACHE_DIR)

    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    out_links: list[str] = []

    for si, sound_link in enumerate(sounds):
        path = (ctx.base_dir / sound_link).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"$music_separation: sound not found: {path}")

        audio, native_sr = load_stereo(path)
        stem_arrays = separate(audio, compiled, shifts=shifts)
        for stem in variant.stems:
            stem_audio = resample_to_native(stem_arrays[stem], native_sr)
            dest = sounds_dir / _output_name(model_id, stem, si)
            dest.write_bytes(write_wav_bytes(stem_audio, native_sr))
            link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
            out_links.append(link)
    return out_links


def _run_roformer(
    ctx: ExternalContext,
    sounds: list[str],
    variant: SeparationVariant,
    *,
    model_id: str,
) -> list[str]:
    configure_models_environment()
    ROFORMER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AUDIO_SEPARATOR_MODEL_DIR", str(ROFORMER_MODELS_DIR))

    sounds_dir = ctx.op_dir / "sounds"
    work_root = ctx.op_dir / "work"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    out_links: list[str] = []

    for si, sound_link in enumerate(sounds):
        path = (ctx.base_dir / sound_link).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"$music_separation: sound not found: {path}")

        stem_paths = roformer_separate(
            path,
            model_filename=variant.model_filename,
            expected_stems=variant.stems,
            work_dir=work_root / f"{variant.id}_track_{si}",
            checkpoint=variant.checkpoint,
        )
        for stem in variant.stems:
            src = stem_paths[stem]
            if not src.is_file():
                raise FileNotFoundError(
                    f"$music_separation: missing RoFormer output for stem {stem!r}: {src}"
                )
            dest = sounds_dir / _output_name(model_id, stem, si)
            shutil.copy2(src, dest)
            link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
            out_links.append(link)
    return out_links


def _run_variant(
    ctx: ExternalContext,
    sounds: list[str],
    variant: SeparationVariant,
    *,
    model_id: str,
    device: str,
    shifts: int,
) -> list[str]:
    if variant.backend == "openvino":
        return _run_htdemucs(
            ctx,
            sounds,
            variant,
            model_id=model_id,
            device=device,
            shifts=shifts,
        )
    if variant.backend == "roformer":
        return _run_roformer(ctx, sounds, variant, model_id=model_id)
    raise RuntimeError(f"$music_separation: unsupported backend {variant.backend!r}")


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    sounds = list(inp.bundle.sounds)
    models = read_arg_list(inp, "model", DEFAULT_MODEL)
    device = inp.args.get("device", "CPU").strip() or "CPU"
    model_path = inp.args.get("model_path", "").strip()
    shifts = _int_arg(inp, "shifts", DEFAULT_SHIFTS, min_value=1)

    if not sounds:
        raise RuntimeError(_HELP.strip())

    if os.environ.get("AH_EMULATE_MUSIC_SEPARATION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        out.sounds = []
        _emulate(ctx, out, sounds, models)
        return out

    out.sounds = []
    for model_name in models:
        variant = resolve_variant(model_name, model_path=model_path)
        out.sounds.extend(
            _run_variant(
                ctx,
                sounds,
                variant,
                model_id=model_name,
                device=device,
                shifts=shifts,
            )
        )

    return out
