"""$text2speech — Kokoro TTS on texts[] / prompts[]."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_arg_list,
    read_bundle_texts,
    read_prompt_texts,
)
from externals.text2speech.model_paths import (
    DEFAULT_VOICE,
    SAMPLE_RATE,
    use_legacy_backend,
)
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$text2speech needs texts[] and/or prompts[].

Example:
  @say: @script -> $text2speech(voice=af_bella)
  @voices: $list
  af_bella, bm_george, am_adam
  @many: @script -> $text2speech(voice=@voices)

Voices: Kokoro names (af_bella, bm_george, …) or paths to .pt under models/kokoro/voices/.
Setup: powershell -File tools\\setup_external_venvs.ps1
AH_EMULATE_TEXT2SPEECH=1 for stub output without kokoro.
"""


def _output_name(voice_id: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", voice_id) or "voice"
    return f"{ts}_{index}_{safe}.wav"


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"$text2speech: invalid {key}={raw!r}") from exc


def _voice_id(voice: str) -> str:
    if voice.endswith(".pt"):
        return Path(voice).stem
    return Path(voice).name if "/" in voice or "\\" in voice else voice


def _collect_texts(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    texts = read_bundle_texts(ctx, inp)
    if texts:
        return texts
    prompts = read_prompt_texts(ctx, inp)
    if prompts:
        return prompts
    if inp.prompt_text.strip():
        return [inp.prompt_text.strip()]
    return []


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    texts: list[str],
    *,
    voices: list[str],
) -> None:
    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for voice in voices:
        vid = _voice_id(voice)
        for text in texts:
            preview = text[:80] + ("…" if len(text) > 80 else "")
            content = (
                f"[emulated $text2speech voice={vid}]\n"
                f"text: {preview}\n"
            )
            dest = sounds_dir / _output_name(vid, idx)
            dest.write_text(content, encoding="utf-8")
            link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
            out.sounds.append(link)
            idx += 1


def _synthesize_one(
    text: str,
    voice: str,
    *,
    device: str,
    speed: float,
    split: str,
    repo_id: str,
    espeak_lib: str,
    chunk_by: str,
) -> np.ndarray:
    if use_legacy_backend():
        from externals.text2speech import legacy_backend

        return legacy_backend.synthesize(
            text,
            voice,
            device=device,
            espeak_lib=espeak_lib,
            chunk_by=chunk_by,
        )
    from externals.text2speech import pipeline_backend

    return pipeline_backend.synthesize(
        text,
        voice,
        device=device,
        speed=speed,
        split=split,
        repo_id=repo_id,
        espeak_lib=espeak_lib,
    )


def _write_audio(ctx: ExternalContext, audio: np.ndarray, filename: str) -> str:
    if use_legacy_backend():
        from externals.text2speech.pipeline_backend import write_wav

        sounds_dir = ctx.op_dir / "sounds"
        sounds_dir.mkdir(parents=True, exist_ok=True)
        dest = sounds_dir / filename
        write_wav(dest, audio, SAMPLE_RATE)
        return str(dest.relative_to(ctx.base_dir)).replace("\\", "/")

    import soundfile as sf

    buf = __import__("io").BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return ctx.new_link("sounds", ".wav", buf.getvalue())


def _require_kokoro() -> None:
    if use_legacy_backend():
        from externals.text2speech.model_paths import legacy_available, kokoro_root

        if not legacy_available():
            raise RuntimeError(
                "$text2speech legacy backend needs models/kokoro/ with models.py and kokoro.py "
                "(clone https://huggingface.co/hexgrad/Kokoro-82M).\n"
                "Or use pipeline backend: AH_TEXT2SPEECH_BACKEND=pipeline"
            )
        kokoro_root()
        return
    try:
        import kokoro  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$text2speech needs kokoro in .venvs/text2speech.\n"
            "  powershell -File tools\\setup_external_venvs.ps1\n"
            "Test stub: AH_EMULATE_TEXT2SPEECH=1"
        ) from exc


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    texts = _collect_texts(ctx, inp)
    if not texts:
        raise ValueError("$text2speech requires at least one text in texts[] or prompts[]")

    voices = read_arg_list(inp, "voice", DEFAULT_VOICE)
    device = inp.args.get("device", "").strip()
    speed = _float_arg(inp.args, "speed", 1.0)
    split = inp.args.get("split", "sentence").strip() or "sentence"
    chunk_by = inp.args.get("chunk_by", split).strip() or split
    repo_id = inp.args.get("repo_id", "hexgrad/Kokoro-82M").strip() or "hexgrad/Kokoro-82M"
    espeak_lib = inp.args.get("espeak_lib", "").strip()

    if os.environ.get("AH_EMULATE_TEXT2SPEECH", "").lower() in ("1", "true", "yes"):
        out.sounds = []
        _emulate(ctx, out, texts, voices=voices)
        return out

    _require_kokoro()
    out.sounds = []
    idx = 0
    backend = "legacy" if use_legacy_backend() else "pipeline"
    print(f"$text2speech: backend={backend} voices={voices!r}", flush=True)

    for voice in voices:
        vid = _voice_id(voice)
        for text in texts:
            print(f"$text2speech: voice={vid} len={len(text)}", flush=True)
            audio = _synthesize_one(
                text,
                voice,
                device=device,
                speed=speed,
                split=split,
                repo_id=repo_id,
                espeak_lib=espeak_lib,
                chunk_by=chunk_by,
            )
            if audio.size == 0:
                raise RuntimeError(f"$text2speech: empty audio for voice={vid!r}")
            link = _write_audio(ctx, audio, _output_name(vid, idx))
            out.sounds.append(link)
            idx += 1

    return out
