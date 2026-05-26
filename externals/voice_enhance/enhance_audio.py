"""Resemble Enhance inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from externals.voice_enhance.model_paths import ensure_models


def _resolve_device(requested: str) -> str:
    if requested.strip():
        dev = requested.strip().lower()
        if dev.startswith("cuda") and ":" not in dev:
            return "cuda"
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_mono(path: Path):
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav, int(sr)


def enhance_file(
    path: Path,
    *,
    device: str = "",
    denoise_only: bool = False,
    denoise_before: bool = False,
    nfe: int = 64,
    solver: str = "midpoint",
    lambd: float = 0.5,
    tau: float = 0.5,
    run_dir: Path | None = None,
) -> tuple[np.ndarray, int]:
    """Run Resemble Enhance on one audio file; returns mono float32 samples and sample rate."""
    from externals.voice_enhance.resemble_bootstrap import bootstrap_resemble_inference

    bootstrap_resemble_inference()
    from resemble_enhance.enhancer.inference import denoise, enhance

    dev = _resolve_device(device)
    run = ensure_models(run_dir)
    dwav, sr = _load_mono(path)

    if denoise_only:
        hwav, out_sr = denoise(dwav, sr, dev, run_dir=run)
    else:
        if denoise_before:
            dwav, sr = denoise(dwav, sr, dev, run_dir=run)
        hwav, out_sr = enhance(
            dwav,
            sr,
            dev,
            nfe=nfe,
            solver=solver.lower(),
            lambd=lambd,
            tau=tau,
            run_dir=run,
        )

    samples = hwav.detach().cpu().numpy().astype(np.float32)
    if samples.ndim > 1:
        samples = samples.squeeze()
    return samples, int(out_sr)


def write_enhanced_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
