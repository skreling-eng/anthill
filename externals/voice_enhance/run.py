"""$voice_enhance — Resemble Enhance on vocal stems (sounds[])."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.voice_enhance.enhance_audio import enhance_file, write_enhanced_wav
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$voice_enhance needs sounds[] (vocal WAV/MP3).

Example:
  @vocals: @track -> $music_separation(model=2stem) -> $select(sounds=[0])
  @clean: @vocals -> $voice_enhance(device=cuda)

Setup: powershell -File tools\\setup_external_venvs.ps1
AH_EMULATE_VOICE_ENHANCE=1 for stub output without resemble-enhance.
"""


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{index}_enhanced.wav"


def _truthy(args: dict[str, str], key: str) -> bool:
    return args.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"$voice_enhance: invalid {key}={raw!r}") from exc


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"$voice_enhance: invalid {key}={raw!r}") from exc


def _sound_path(ctx: ExternalContext, link: str) -> Path:
    path = Path(link)
    if not path.is_absolute():
        path = (ctx.base_dir / link).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"$voice_enhance: sound not found: {path}")
    return path


def _require_resemble() -> None:
    try:
        import resemble_enhance  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$voice_enhance needs resemble-enhance in .venvs/voice_enhance.\n"
            "  powershell -File tools\\setup_external_venvs.ps1\n"
            "Test stub: AH_EMULATE_VOICE_ENHANCE=1"
        ) from exc


def _emulate(ctx: ExternalContext, out: ArrayBundle, sounds: list[str]) -> None:
    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    for i, sound in enumerate(sounds):
        content = f"[emulated $voice_enhance]\nfrom: {sound}\n"
        dest = sounds_dir / _output_name(i)
        dest.write_text(content, encoding="utf-8")
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.sounds.append(link)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    sounds = list(inp.bundle.sounds)
    if not sounds:
        raise RuntimeError(_HELP.strip())

    out = inp.bundle.copy()
    out.sounds = []

    if os.environ.get("AH_EMULATE_VOICE_ENHANCE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        _emulate(ctx, out, sounds)
        return out

    _require_resemble()

    device = inp.args.get("device", "").strip()
    denoise_only = _truthy(inp.args, "denoise_only")
    denoise_before = _truthy(inp.args, "denoise_before")
    nfe = _int_arg(inp.args, "nfe", 64)
    solver = inp.args.get("solver", "midpoint").strip() or "midpoint"
    lambd = _float_arg(inp.args, "lambd", 0.5)
    tau = _float_arg(inp.args, "tau", 0.5)

    print(
        f"$voice_enhance: device={device or 'auto'} "
        f"denoise_only={denoise_only} denoise_before={denoise_before}",
        flush=True,
    )

    for i, link in enumerate(sounds):
        src = _sound_path(ctx, link)
        print(f"$voice_enhance: enhancing {src.name}", flush=True)
        samples, sr = enhance_file(
            src,
            device=device,
            denoise_only=denoise_only,
            denoise_before=denoise_before,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )
        if samples.size == 0:
            raise RuntimeError(f"$voice_enhance: empty output for {src.name}")
        wav = write_enhanced_wav(samples, sr)
        out.sounds.append(ctx.new_link("sounds", ".wav", wav))

    return out
