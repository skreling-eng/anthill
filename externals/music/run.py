"""$music — ACE-Step music generation (caption=prompts, lyrics=texts)."""

from __future__ import annotations

import os
import random
import io
import wave
from datetime import datetime
from pathlib import Path

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_arg_list,
    read_bundle_texts,
)
from externals.music.model_paths import ensure_companion_gguf, gguf_stack_ready
from ahlib.ah_runtime import ArrayBundle

_MUSIC_HELP = """
$music could not run — ace-synth (GGUF) backend not ready.

1. GGUF weights in models/ace-step-1.5/:
     acestep-v15-xl-base-BF16.gguf, vae-BF16.gguf, Qwen3-Embedding-0.6B-BF16.gguf
   Optional: ACESTEP_DOWNLOAD_MISSING=1 for missing VAE/embedding

2. ace-synth binary:
     set ACESTEP_DOWNLOAD_BIN=1   (Win/Linux/macOS ARM64)
     or set ACESTEP_SYNTH_BIN to ace-synth / ace-synth.exe

Other backends (optional): ACESTEP_BACKEND=native|api

Test placeholder (audible beep): set AH_EMULATE_MUSIC=1
"""


def _repeat_count(inp: ExternalInput) -> int:
    if inp.repeat > 1:
        return inp.repeat
    return max(1, int(inp.args.get("count", "1")))


def _read_captions(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    """Style / caption from prompts[] (generation prompt), never from texts[]."""
    captions: list[str] = []
    for link in inp.bundle.prompts:
        text = ctx.read_link_text(link)
        if text:
            captions.append(text)
    return captions


def _read_lyrics(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    """Song lyrics from texts[] only."""
    return read_bundle_texts(ctx, inp)


def _caption_lyrics_pairs(
    captions: list[str], lyrics_list: list[str]
) -> list[tuple[str, str]]:
    if not captions:
        captions = [""]
    if not lyrics_list:
        lyrics_list = ["[Instrumental]"]
    if len(captions) == len(lyrics_list):
        return list(zip(captions, lyrics_list))
    if len(captions) == 1:
        return [(captions[0], lyr) for lyr in lyrics_list]
    if len(lyrics_list) == 1:
        return [(cap, lyrics_list[0]) for cap in captions]
    return [("\n\n".join(captions), "\n\n".join(lyrics_list))]


def _placeholder_wav_bytes(*, seconds: float = 2.0, sample_rate: int = 48000) -> bytes:
    import math
    import struct

    buf = io.BytesIO()
    n_frames = int(sample_rate * seconds)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            t = i / sample_rate
            sample = int(8000 * math.sin(2 * math.pi * 440.0 * t))
            frames.extend(struct.pack("<h", sample))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(frames)
    return buf.getvalue()


def _output_name(model: str, index: int, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in model)
    return f"{ts}_{safe}_music_{index}{ext}"


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    model: str,
    pairs: list[tuple[str, str]],
    count: int,
) -> ArrayBundle:
    print(
        "$music AH_EMULATE_MUSIC=1 — placeholder tone only. "
        "Unset for real ACE-Step output via ace-synth."
    )
    for _caption, _lyrics in pairs:
        for _vi in range(count):
            link = ctx.new_link("sounds", ".wav", _placeholder_wav_bytes(seconds=2.0))
            out.sounds.append(link)
    return out


def _generate_via_api(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    model_name: str,
    duration: float,
    seed: int,
) -> Path:
    from externals.music.ace_client import generate_via_api
    from externals.music.model_list import get_music_model

    music_model = get_music_model(model_name)
    return generate_via_api(
        caption=caption,
        lyrics=lyrics,
        output_path=output_path,
        model=music_model.api_model,
        duration=duration,
        seed=seed if seed >= 0 else None,
    )


def _generate_via_native(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    model_name: str,
    duration: float,
    seed: int,
    steps: int,
    audio_format: str,
    extras: dict | None = None,
) -> Path:
    from externals.music.ace_native import generate_via_native, native_available
    from externals.music.model_list import get_music_model
    from externals.music.model_paths import safetensors_stack_ready

    music_model = get_music_model(model_name)
    if not native_available():
        raise RuntimeError(
            f"$music model={model_name!r} uses *.safetensors (PyTorch). "
            "Install ace-step from https://github.com/ace-step/ACE-Step-1.5 "
            "and set ACESTEP_BACKEND=native (or use a GGUF model like default)."
        )
    folder = music_model.models_dir()
    if not safetensors_stack_ready(folder):
        raise FileNotFoundError(
            f"No *.safetensors found under {folder}\n"
            "  Expected Comfy layout, e.g.:\n"
            "    diffusion_models/acestep_v1.5_turbo.safetensors\n"
            "    vae/ace_1.5_vae.safetensors\n"
            "    text_encoders/qwen_0.6b_ace15.safetensors\n"
            "  Or HF layout: acestep-v15-turbo/ under that folder."
        )
    merged = dict(extras or {})
    synth_only = {k: v for k, v in merged.items() if k not in ("adapter",)}
    return generate_via_native(
        caption=caption,
        lyrics=lyrics,
        output_path=output_path,
        config_path=music_model.config_path,
        checkpoints_dir=folder,
        duration=duration,
        seed=seed,
        steps=steps,
        audio_format=audio_format,
        param_defaults=music_model.native_generation_defaults(),
        extras=synth_only,
    )


def _generate_via_synth_gguf(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    model_name: str,
    duration: float,
    seed: int,
    steps: int,
    extras: dict | None = None,
) -> Path:
    from externals.music.ace_subprocess import generate_via_synth
    from externals.music.model_list import dit_path_for_model, get_music_model

    from externals.music.model_list import adapter_for_model

    music_model = get_music_model(model_name)
    dit = Path(dit_path_for_model(music_model))
    models_dir = music_model.models_dir()
    merged = dict(extras or {})
    adapter_override = merged.pop("adapter", None)
    adapter = adapter_for_model(music_model, override=adapter_override)
    return generate_via_synth(
        caption=caption,
        lyrics=lyrics,
        output_path=output_path,
        dit_gguf=dit,
        duration=duration,
        seed=seed,
        steps=steps,
        extras=merged,
        adapter=adapter,
        models_dir=models_dir,
    )


def _backend() -> str:
    raw = os.environ.get("ACESTEP_BACKEND", "synth").strip().lower()
    if raw in ("native", "python", "pytorch"):
        return "native"
    if raw in ("gguf", "synth", "cpp"):
        return "synth"
    if raw in ("api", "http"):
        return "api"
    if raw == "auto":
        return "auto"
    return "synth"


def _prepare_synth() -> None:
    from externals.music.ace_bin import ensure_synth_bin, find_synth_bin

    ensure_companion_gguf()
    if find_synth_bin() is None:
        ensure_synth_bin()


def _generate_one(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    model_name: str,
    duration: float,
    seed: int,
    steps: int,
    audio_format: str,
    extras: dict | None = None,
) -> Path:
    from externals.music.ace_bin import find_synth_bin, synth_stack_ready
    from externals.music.ace_native import native_available

    from externals.music.model_list import get_music_model

    music_model = get_music_model(model_name)
    backend = _backend()
    if music_model.weights == "safetensors":
        backend = "native"
    elif backend == "auto":
        backend = music_model.preferred_backend()

    use_api = backend == "api" and os.environ.get("ACESTEP_API_URL", "").strip()

    if backend == "native":
        return _generate_via_native(
            caption=caption,
            lyrics=lyrics,
            output_path=output_path,
            model_name=model_name,
            duration=duration,
            seed=seed,
            steps=steps,
            audio_format=audio_format,
            extras=extras,
        )

    if backend in ("synth", "gguf", "cpp"):
        _prepare_synth()
        if not synth_stack_ready():
            raise RuntimeError(_MUSIC_HELP.strip())
        return _generate_via_synth_gguf(
            caption=caption,
            lyrics=lyrics,
            output_path=output_path,
            model_name=model_name,
            duration=duration,
            seed=seed,
            steps=steps,
            extras=extras,
        )

    if use_api:
        return _generate_via_api(
            caption=caption,
            lyrics=lyrics,
            output_path=output_path,
            model_name=model_name,
            duration=duration,
            seed=seed,
        )

    _prepare_synth()
    raise RuntimeError(_MUSIC_HELP.strip())


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    from externals.music.models_env import configure_models_environment
    from externals.music.model_list import get_music_model
    from externals.music.synth_options import default_steps, merge_synth_extras

    configure_models_environment()

    out = inp.bundle.copy()
    models = read_arg_list(inp, "model", "default")
    captions = _read_captions(ctx, inp)
    lyrics_list = _read_lyrics(ctx, inp)
    pairs = _caption_lyrics_pairs(captions, lyrics_list)
    count = _repeat_count(inp)
    seed_base = int(inp.args.get("seed", "-1"))

    def _duration_for(model_name: str) -> float:
        if "duration" in inp.args:
            return float(inp.args["duration"])
        return get_music_model(model_name).duration

    if os.environ.get("AH_EMULATE_MUSIC", "").lower() in ("1", "true", "yes"):
        for model_name in models:
            _emulate(ctx, out, model_name, pairs, count)
        out.prompts.clear()
        return out

    if _backend() in ("synth", "gguf", "cpp", "auto"):
        _prepare_synth()

    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    track_index = 0

    for model_name in models:
        for caption, lyrics in pairs:
            for vi in range(count):
                seed = seed_base
                if seed_base < 0:
                    seed = random.randint(0, 2**31 - 1)
                elif count > 1:
                    seed = seed_base + vi

                ext = inp.args.get("format", "wav").lstrip(".") or "wav"
                out_name = _output_name(model_name, track_index, f".{ext}")
                out_path = sounds_dir / out_name
                _generate_one(
                    caption=caption,
                    lyrics=lyrics,
                    output_path=out_path,
                    model_name=model_name,
                    duration=_duration_for(model_name),
                    seed=seed,
                    steps=default_steps(inp, model_name),
                    audio_format=ext,
                    extras=merge_synth_extras(inp, model_name),
                )
                link = str(out_path.relative_to(ctx.base_dir)).replace("\\", "/")
                out.sounds.append(link)
                track_index += 1

    out.prompts.clear()
    return out
