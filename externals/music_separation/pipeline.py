"""HTDemucs v4 music separation via Intel OpenVINO IR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from externals.music_separation.audio_io import SAMPLE_RATE

STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")
SEGMENT = 343980  # 44100 * 39 / 5
HOP = 1024
N_FFT = 4096
PAD1D = HOP // 2 * 3


@dataclass
class CompiledDemucs:
    compiled: object
    input_x: str
    input_xt: str
    output_x: str
    output_xt: str


_COMPILED: CompiledDemucs | None = None
DEFAULT_SHIFTS = 2  # HTDemucs::Apply default in openvino-plugins-ai-audacity
DEFAULT_OVERLAP = 0.25


def _require_torch():
    import torch

    return torch


def _pad1d_reflect(x, left: int, right: int):
    """Match HTDemucs pad1d in openvino-plugins-ai-audacity htdemucs.cpp."""
    import torch.nn.functional as F

    length = x.shape[-1]
    max_pad = max(left, right)
    if length <= max_pad:
        extra_pad = max_pad - length + 1
        extra_pad_right = min(right, extra_pad)
        extra_pad_left = extra_pad - extra_pad_right
        left -= extra_pad_left
        right -= extra_pad_right
        x = F.pad(x, (extra_pad_left, extra_pad_right))
    return F.pad(x, (left, right), mode="reflect")


def _torch_stft(x):
    torch = _require_torch()
    window = torch.hann_window(N_FFT, device=x.device, dtype=x.dtype)
    return torch.stft(
        x,
        N_FFT,
        HOP,
        N_FFT,
        window,
        center=True,
        pad_mode="reflect",
        normalized=True,
        onesided=True,
        return_complex=True,
    )


def _torch_istft(z, length: int):
    torch = _require_torch()
    window = torch.hann_window(N_FFT, device=z.device, dtype=torch.float32)
    return torch.istft(
        z,
        N_FFT,
        HOP,
        N_FFT,
        window,
        center=True,
        normalized=True,
        onesided=True,
        length=length,
    )


def _normalize_mix(mix: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Match HTDemucs::Apply — normalize stereo mix using mean(L,R) statistics."""
    ref = mix.mean(axis=0)
    ref_mean = float(ref.mean())
    ref_std = float(ref.std())
    if ref_std < 1e-8:
        ref_std = 1e-8
    return ((mix - ref_mean) / ref_std).astype(np.float32), ref_mean, ref_std


def _denormalize_stems(
    stems: dict[str, np.ndarray], ref_mean: float, ref_std: float
) -> dict[str, np.ndarray]:
    return {name: stem * ref_std + ref_mean for name, stem in stems.items()}


def _spec(mix: np.ndarray) -> np.ndarray:
    """Frequency branch input spectrogram ``(B, C, F, T)`` complex — torch STFT like Audacity."""
    torch = _require_torch()
    b, c, length = mix.shape
    le = int(np.ceil(length / HOP))
    pad_right = PAD1D + le * HOP - length
    x = torch.from_numpy(mix.astype(np.float32))
    x = _pad1d_reflect(x, PAD1D, pad_right)
    flat = x.reshape(-1, x.shape[-1])
    z = _torch_stft(flat)
    z = z.reshape(b, c, z.shape[-2], z.shape[-1])
    z = z[..., :-1, :]
    z = z[..., 2 : 2 + le]
    return z.cpu().numpy()


def _magnitude(z: np.ndarray) -> np.ndarray:
    """Match HTDemucs::_magnitude — view_as_real, permute, reshape."""
    b, c, fr, t = z.shape
    stacked = np.stack([np.real(z), np.imag(z)], axis=2)  # (B, C, 2, Fr, T)
    return stacked.reshape(b, c * 2, fr, t).astype(np.float32)


def _mask_to_complex(m: np.ndarray) -> np.ndarray:
    b, s, c4, fr, t = m.shape
    m = m.reshape(b, s, 2, 2, fr, t)
    m = np.transpose(m, (0, 1, 2, 4, 5, 3))
    return m[..., 0] + 1j * m[..., 1]


def _ispec(z: np.ndarray, length: int) -> np.ndarray:
    torch = _require_torch()
    import torch.nn.functional as F

    b, s, c, fr, t = z.shape
    flat = torch.from_numpy(z.astype(np.complex64)).reshape(b * s * c, fr, t)
    flat = F.pad(flat, (0, 0, 0, 1))
    flat = F.pad(flat, (2, 2))
    le = HOP * int(np.ceil(length / HOP)) + 2 * PAD1D
    wave = _torch_istft(flat, le)
    wave = wave.reshape(b, s, c, wave.shape[-1])
    return wave[..., PAD1D : PAD1D + length].cpu().numpy()


def _center_trim(tensor: np.ndarray, reference: int) -> np.ndarray:
    delta = tensor.shape[-1] - reference
    if delta <= 0:
        return tensor
    start = delta // 2
    end = tensor.shape[-1] - (delta - start)
    return tensor[..., start:end]


def _compile_model(model_xml: Path, device: str, *, cache_dir: Path | None = None) -> CompiledDemucs:
    from openvino import Core

    core = Core()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property(device, {"CACHE_DIR": str(cache_dir)})
    model = core.read_model(str(model_xml))
    compiled = core.compile_model(model, device)
    inputs = [i.get_any_name() for i in model.inputs]
    outputs = [o.get_any_name() for o in model.outputs]
    return CompiledDemucs(
        compiled=compiled,
        input_x=inputs[0],
        input_xt=inputs[1],
        output_x=outputs[0],
        output_xt=outputs[1],
    )


def get_compiled(
    model_xml: Path, device: str, *, cache_dir: Path | None = None
) -> CompiledDemucs:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = _compile_model(model_xml, device, cache_dir=cache_dir)
    return _COMPILED


def _run_segment(compiled: CompiledDemucs, mix: np.ndarray) -> np.ndarray:
    """Run one padded segment; returns ``(4, 2, segment_len)``."""
    length = mix.shape[-1]
    if mix.shape[-1] < SEGMENT:
        pad = SEGMENT - mix.shape[-1]
        mix = np.pad(mix, ((0, 0), (0, 0), (0, pad)), mode="constant")
    elif mix.shape[-1] > SEGMENT:
        mix = mix[..., :SEGMENT]

    z = _spec(mix)
    x = _magnitude(z)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True)
    x_in = ((x - mean) / (1e-5 + std)).astype(np.float32)

    xt = mix.astype(np.float32)
    meant = xt.mean(axis=(1, 2), keepdims=True)
    stdt = xt.std(axis=(1, 2), keepdims=True)
    xt_in = ((xt - meant) / (1e-5 + stdt)).astype(np.float32)

    result = compiled.compiled(
        {
            compiled.input_x: x_in,
            compiled.input_xt: xt_in,
        }
    )
    x_out = np.array(result[compiled.output_x])
    xt_out = np.array(result[compiled.output_xt])

    b = x_out.shape[0]
    s = len(STEMS)
    fq = x_out.shape[2]
    t = x_out.shape[3]
    x_out = x_out.reshape(b, s, 4, fq, t)
    x_out = x_out * std[:, None] + mean[:, None]

    z_out = _mask_to_complex(x_out)
    wave = _ispec(z_out, SEGMENT)

    xt_out = xt_out.reshape(b, s, 2, SEGMENT)
    xt_out = xt_out * stdt[:, None] + meant[:, None]
    out = wave + xt_out
    out = _center_trim(out, length)
    return out[0]


def _triangle_weights(segment: int) -> np.ndarray:
    half = segment // 2
    ramp_up = np.arange(1, half + 1, dtype=np.float32)
    ramp_down = np.arange(segment - half, 0, -1, dtype=np.float32)
    return np.concatenate([ramp_up, ramp_down])


def separate(
    mix: np.ndarray,
    compiled: CompiledDemucs,
    *,
    shifts: int = DEFAULT_SHIFTS,
    overlap: float = DEFAULT_OVERLAP,
) -> dict[str, np.ndarray]:
    """Separate stereo ``(2, samples)`` into four stems."""
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError(f"expected stereo (2, samples), got {mix.shape}")

    mix_norm, ref_mean, ref_std = _normalize_mix(mix)
    length = mix_norm.shape[1]
    mix_b = mix_norm[np.newaxis, ...]
    stride = max(1, int((1.0 - overlap) * SEGMENT))
    weight = _triangle_weights(SEGMENT)
    weight = weight / weight.max()
    rng = np.random.default_rng(0)
    max_shift = SAMPLE_RATE // 2

    def _separate_once(source: np.ndarray) -> np.ndarray:
        chunk_out = np.zeros((len(STEMS), 2, source.shape[-1]), dtype=np.float32)
        sum_weight = np.zeros(source.shape[-1], dtype=np.float32)
        for start in range(0, source.shape[-1], stride):
            end = min(start + SEGMENT, source.shape[-1])
            seg_len = end - start
            chunk = source[:, :, start:end]
            if chunk.shape[-1] < SEGMENT:
                chunk = np.pad(
                    chunk,
                    ((0, 0), (0, 0), (0, SEGMENT - chunk.shape[-1])),
                    mode="constant",
                )
            seg_out = _run_segment(compiled, chunk)
            w = weight[:seg_len]
            chunk_out[:, :, start:end] += seg_out[:, :, :seg_len] * w
            sum_weight[start:end] += w
        sum_weight = np.maximum(sum_weight, 1e-8)
        chunk_out /= sum_weight
        return chunk_out

    accum = np.zeros((len(STEMS), 2, length), dtype=np.float32)
    if shifts <= 1:
        accum += _separate_once(mix_b)[:, :, :length]
    else:
        padded = np.pad(mix_b, ((0, 0), (0, 0), (0, 2 * max_shift)), mode="constant")
        for _ in range(shifts):
            offset = int(rng.integers(0, max_shift))
            chunk_len = length + max_shift - offset
            shifted = padded[:, :, offset : offset + chunk_len]
            part = _separate_once(shifted)
            accum += part[:, :, max_shift - offset : max_shift - offset + length]
        accum /= shifts

    stems = {name: accum[i] for i, name in enumerate(STEMS)}
    return _denormalize_stems(stems, ref_mean, ref_std)


def require_openvino() -> None:
    try:
        import openvino  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "$music_separation needs openvino: uv pip install openvino"
        ) from exc
