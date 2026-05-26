"""$change_voice — RVC voice conversion on sounds[]."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list
from externals.change_voice.model_paths import (
    DEFAULT_MODEL,
    detect_rvc_version,
    resolve_model,
    resolve_rvc_version,
)
from ahlib.ah_runtime import ArrayBundle

_HELP = """
$change_voice needs sounds[] (vocals WAV/MP3).

Example:
  @vocals: @track -> $music_separation(model=2stem)
  @mm: @vocals -> $change_voice(model=MuscleMan, f0up_key=2, protect=0.5)
  @voices: $list
  a, b, c
  @many: @vocals -> $change_voice(model=@voices)

Models live in models/rvc/<name>/ (*.pth + optional *.index), e.g.:
  models/rvc/MuscleMan/muscleman.pth
  models/rvc/MuscleMan/added_IVF231_Flat_nprobe_1_MuscleMan_v2.index

Or pass model_path= / index_path= explicitly.
AH_EMULATE_CHANGE_VOICE=1 for stub output without rvc-python.
"""


def _output_name(model_id: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", model_id) or "model"
    return f"{ts}_{index}_{safe}.wav"


def _float_arg(args: dict[str, str], key: str, default: float) -> float:
    raw = args.get(key, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"$change_voice: invalid {key}={raw!r}") from exc


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"$change_voice: invalid {key}={raw!r}") from exc


def _default_device(requested: str) -> str:
    if requested.strip():
        return requested.strip()
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu:0"
    except ImportError:
        return "cpu:0"


def _sound_path(ctx: ExternalContext, link: str) -> Path:
    path = Path(link)
    if not path.is_absolute():
        path = (ctx.base_dir / link).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"$change_voice: sound not found: {path}")
    return path


def _require_rvc() -> None:
    try:
        import rvc_python  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$change_voice needs rvc-python in .venvs/change_voice (Python 3.10).\n"
            "  powershell -File tools\\setup_external_venvs.ps1\n"
            "Test stub: AH_EMULATE_CHANGE_VOICE=1"
        ) from exc


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    sounds: list[str],
    *,
    model_id: str,
) -> None:
    sounds_dir = ctx.op_dir / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    for i, sound in enumerate(sounds):
        content = (
            f"[emulated $change_voice model={model_id}]\n"
            f"from: {sound}\n"
        )
        dest = sounds_dir / _output_name(model_id, i)
        dest.write_text(content, encoding="utf-8")
        link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        out.sounds.append(link)


def _convert_sounds(
    ctx: ExternalContext,
    sounds: list[str],
    *,
    pth: Path,
    index: Path | None,
    model_id: str,
    device: str,
    version: str,
    params: dict[str, float | int | str],
) -> list[str]:
    from rvc_python.infer import RVCInference

    _require_rvc()
    index_str = str(index) if index is not None else ""
    rvc = RVCInference(
        device=device,
        model_path=str(pth),
        index_path=index_str,
        version=version,
    )
    rvc.set_params(**params)

    sounds_dir = ctx.op_dir / "sounds"
    work_dir = ctx.op_dir / "work"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_links: list[str] = []

    try:
        for si, sound_link in enumerate(sounds):
            src = _sound_path(ctx, sound_link)
            dest = sounds_dir / _output_name(model_id, si)
            work_src = work_dir / f"input_{si}{src.suffix}"
            if src.suffix.lower() != ".wav":
                shutil.copy2(src, work_src)
                infer_in = str(work_src)
            else:
                infer_in = str(src)
            rvc.infer_file(infer_in, str(dest))
            link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
            out_links.append(link)
    finally:
        rvc.unload_model()

    return out_links


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    sounds = list(inp.bundle.sounds)
    if not sounds:
        raise RuntimeError(_HELP.strip())

    models = read_arg_list(inp, "model", DEFAULT_MODEL)
    model_path = inp.args.get("model_path", "").strip()
    index_path = inp.args.get("index_path", "").strip()
    use_explicit_paths = len(models) == 1
    device = _default_device(inp.args.get("device", ""))
    version_arg = inp.args.get("version", "").strip()

    params: dict[str, float | int | str] = {
        "f0up_key": _int_arg(inp.args, "f0up_key", 0),
        "protect": _float_arg(inp.args, "protect", 0.33),
        "index_rate": _float_arg(inp.args, "index_rate", 0.5),
    }
    if inp.args.get("f0method", "").strip():
        params["f0method"] = inp.args["f0method"].strip()

    if os.environ.get("AH_EMULATE_CHANGE_VOICE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        out.sounds = []
        for model_name in models:
            if use_explicit_paths and model_path:
                _, _, model_id = resolve_model(
                    "",
                    model_path=model_path,
                    index_path=index_path,
                )
            else:
                model_id = model_name
            _emulate(ctx, out, sounds, model_id=model_id)
        return out

    out.sounds = []
    for model_name in models:
        pth, index, model_id = resolve_model(
            model_name,
            model_path=model_path if use_explicit_paths else "",
            index_path=index_path if use_explicit_paths else "",
        )
        version = resolve_rvc_version(pth, version_arg)
        if not version_arg:
            print(f"$change_voice: {model_id} detected as {version}", flush=True)
        out.sounds.extend(
            _convert_sounds(
                ctx,
                sounds,
                pth=pth,
                index=index,
                model_id=model_id,
                device=device,
                version=version,
                params=params,
            )
        )
    return out
