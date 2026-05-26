"""Run acestep.cpp ace-synth locally (GGUF, no HTTP API server)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from externals.music.ace_bin import ensure_synth_bin, find_synth_bin, synth_bin_candidates
from externals.music.model_paths import resolve_gguf_stack_in


def _is_turbo_dit(dit: Path) -> bool:
    return "turbo" in dit.name.lower()


def _build_request(
    *,
    caption: str,
    lyrics: str,
    duration: float,
    seed: int,
    steps: int,
    dit: Path,
    extras: dict | None = None,
) -> dict:
    req: dict = {
        "caption": caption,
        "lyrics": lyrics or "[Instrumental]",
        "duration": duration if duration > 0 else 0,
        "seed": seed,
        "use_cot_caption": False,
        "batch_size": 1,
    }
    if extras:
        req.update(extras)
    if steps > 0:
        req["inference_steps"] = steps
    elif "inference_steps" not in req:
        if _is_turbo_dit(dit):
            req.setdefault("inference_steps", 8)
            req.setdefault("shift", 3.0)
            req.setdefault("guidance_scale", 1.0)
        else:
            req["inference_steps"] = 50
            req.setdefault("shift", 1.0)
            req.setdefault("guidance_scale", 1.0)
    return req


def _find_output_audio(work_dir: Path, stem: str) -> Path | None:
    patterns = (
        f"{stem}*.wav",
        f"{stem}*.mp3",
        f"{stem}0.wav",
        f"{stem}0.mp3",
        f"{stem}00.wav",
        f"{stem}00.mp3",
    )
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(work_dir.glob(pattern))
    hits = [p for p in hits if p.is_file() and p.stat().st_size > 1000]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def generate_via_synth(
    *,
    caption: str,
    lyrics: str,
    output_path: Path,
    dit_gguf: Path | None = None,
    duration: float = 30.0,
    seed: int = 0,
    steps: int = 0,
    extras: dict | None = None,
    adapter: str | None = None,
    models_dir: Path | None = None,
) -> Path:
    bin_path = ensure_synth_bin() or find_synth_bin()
    if not bin_path:
        tried = "\n  ".join(str(p) for p in synth_bin_candidates()[:6])
        raise FileNotFoundError(
            "ace-synth not found (local GGUF runner, no API server).\n"
            "  Set ACESTEP_SYNTH_BIN, place ace-synth.exe in models/ace-step-1.5/, "
            "or ACESTEP_DOWNLOAD_BIN=1 (Win/Linux/macOS ARM64).\n"
            f"  Checked:\n  {tried}"
        )

    if dit_gguf is not None:
        dit = dit_gguf
        models_dir = models_dir or dit.parent
    if models_dir is None:
        raise ValueError("models_dir or dit_gguf required for ace-synth")
    dit, _embedding, _vae = resolve_gguf_stack_in(models_dir)
    if dit_gguf is not None:
        dit = dit_gguf

    work_dir = output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    req_stem = "music_request"
    req_path = work_dir / f"{req_stem}.json"
    req = _build_request(
        caption=caption,
        lyrics=lyrics,
        duration=duration,
        seed=seed,
        steps=steps,
        dit=dit,
        extras=extras,
    )
    req.pop("adapter", None)

    from externals.music.model_list import resolve_adapter_stem

    adapter_stem = resolve_adapter_stem(adapter, models_dir=models_dir)
    if adapter and not adapter_stem:
        print(f"$music adapter {adapter!r} not in {models_dir / 'adapters'}; skipping")
    adapters_path = None
    if adapter_stem:
        req["adapter"] = adapter_stem
        adapters_path = models_dir / "adapters"

    req_path.write_text(json.dumps(req, indent=2), encoding="utf-8")

    cmd = [
        str(bin_path),
        "--models",
        str(models_dir),
        "--request",
        req_path.name,
    ]
    if adapters_path is not None and adapters_path.is_dir():
        cmd.extend(["--adapters", str(adapters_path)])

    steps_log = req.get("inference_steps", "?")
    shift_log = req.get("shift", "?")
    adapter_log = adapter_stem or "none"
    print(
        f"$music ace-synth ({bin_path.name}) dit={dit.name} "
        f"steps={steps_log} shift={shift_log} adapter={adapter_log}"
    )
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ace-synth failed ({exc.returncode}):\n{exc.stderr or exc.stdout}"
        ) from exc

    produced = _find_output_audio(work_dir, req_stem)
    if produced is None:
        raise FileNotFoundError(
            f"ace-synth finished but no audio found in {work_dir} (expected {req_stem}*.wav/mp3)"
        )
    if produced.resolve() != output_path.resolve():
        shutil.copy2(produced, output_path)
    return output_path


# re-export for model_paths
def synth_bin() -> Path | None:
    return find_synth_bin()
