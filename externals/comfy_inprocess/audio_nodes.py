"""LoadAudio handler for in-process Comfy (core nodes lack LoadAudio)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REGISTERED = False


def _f32_pcm(wav: Any) -> Any:
    import torch

    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2**15)
    if wav.dtype == torch.int32:
        return wav.float() / (2**31)
    raise ValueError(f"LoadAudio: unsupported sample dtype {wav.dtype}")


def _load_audio_file(path: Path) -> tuple[Any, int]:
    """Decode audio to float tensor (channels, samples) via PyAV (same as comfy_lib LoadAudio)."""
    import av
    import torch

    with av.open(str(path)) as af:
        if not af.streams.audio:
            raise RuntimeError(f"LoadAudio: no audio stream in {path.name}")
        stream = af.streams.audio[0]
        sr = int(stream.codec_context.sample_rate)
        n_channels = int(stream.channels)
        frames: list[Any] = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t().contiguous()
            frames.append(buf)
        if not frames:
            raise RuntimeError(f"LoadAudio: could not decode {path.name}")
        wav = _f32_pcm(torch.cat(frames, dim=1))
    if wav.numel() == 0:
        raise RuntimeError(f"LoadAudio: empty audio file: {path.name}")
    return wav, sr


def _load_audio_handler(inputs: dict[str, Any]) -> tuple[Any, ...]:
    import folder_paths

    name = (inputs.get("audio") or inputs.get("audio_file") or "").strip()
    if not name:
        raise RuntimeError("LoadAudio: missing audio filename")

    input_dir = Path(folder_paths.get_input_directory())
    path = Path(name)
    if not path.is_file():
        path = input_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"LoadAudio: file not found: {name!r} (input dir {input_dir})")

    waveform, sample_rate = _load_audio_file(path)
    # Comfy AUDIO: batch x channels x samples
    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    return (audio,)


def register_comfy_audio_handlers() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from externals.comfy_inprocess.executor import register_node_handler

    register_node_handler(
        "LoadAudio",
        _load_audio_handler,
        input_types={
            "required": {"audio": ("STRING",)},
            "optional": {"audio_file": ("STRING",), "seek_seconds": ("FLOAT",), "duration": ("FLOAT",)},
        },
    )
    _REGISTERED = True
