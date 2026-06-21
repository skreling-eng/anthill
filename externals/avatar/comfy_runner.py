"""Comfy-in-process SkyReels V3 avatar runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from externals.avatar.comfy_workflow import build_avatar_prompt, resolve_avatar_size
from externals.comfy_inprocess.bootstrap import bootstrap_comfy, get_nodes_module
from externals.comfy_inprocess.executor import execute_prompt, find_node_id


def _empty_cuda_cache_before_inference() -> None:
    import gc

    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache(force=True)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _tensor_to_frames(image_tensor) -> list[np.ndarray]:
    from externals.comfy_inprocess.prompt_executor import _unwrap_output_slot

    image_tensor = _unwrap_output_slot(image_tensor)
    if not hasattr(image_tensor, "detach"):
        raise RuntimeError(
            f"WanVideoPassImagesFromSamples output is not a tensor "
            f"(got {type(image_tensor).__name__})"
        )
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        frames = arr
    elif arr.ndim == 3:
        frames = arr[np.newaxis, ...]
    else:
        raise RuntimeError(f"Unexpected frame tensor shape: {arr.shape}")
    out: list[np.ndarray] = []
    for frame in frames:
        if frame.max() <= 1.0:
            out.append((frame * 255.0).clip(0, 255).astype(np.uint8))
        else:
            out.append(frame.clip(0, 255).astype(np.uint8))
    return out


def _write_mp4(frames: list[np.ndarray], dest: Path, *, fps: int = 24) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames]
        export_to_video(pil_frames, str(dest), fps=fps)
        return dest
    except ImportError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(dest, np.stack(frames), fps=fps, codec="libx264")
        return dest
    except ImportError as exc:
        raise RuntimeError(
            "$avatar: install imageio or diffusers for MP4 export"
        ) from exc


@dataclass
class ComfyAvatarRunner:
    work_dir: Path
    fps: int = 24
    _nodes: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.resolve()
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        bootstrap_comfy(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            load_wan_wrapper=True,
        )

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = get_nodes_module()
        return self._nodes

    def run_avatar(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        prompt: str,
        output_path: Path,
        negative_prompt: str = "",
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int = 4,
        num_frames: int = 400,
        frame_window_size: int = 81,
        motion_frame: int = 5,
        drop_frames: int = 12,
        cfg: float = 1.0,
        workflow_ref: str = "",
    ) -> Path:
        job_width, job_height = resolve_avatar_size(
            image_path, width=width, height=height
        )
        prompt_dict, _used_seed = build_avatar_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image_path,
            audio_path=audio_path,
            input_dir=self.input_dir,
            seed=seed,
            width=job_width,
            height=job_height,
            steps=steps,
            fps=float(self.fps),
            num_frames=num_frames,
            frame_window_size=frame_window_size,
            motion_frame=motion_frame,
            drop_frames=drop_frames,
            cfg=cfg,
            workflow_ref=workflow_ref,
        )
        from externals.comfy_inprocess.vram_config import apply_comfy_vram_settings

        apply_comfy_vram_settings()
        _empty_cuda_cache_before_inference()
        try:
            import comfy.model_management as mm

            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
        except ImportError:
            pass

        outputs = execute_prompt(prompt_dict, nodes_module=self.nodes)
        decode_id = find_node_id(prompt_dict, "WanVideoPassImagesFromSamples")
        if decode_id is None:
            raise RuntimeError("WanVideoPassImagesFromSamples node missing from workflow")
        frames_tensor = outputs[decode_id][0]
        frames = _tensor_to_frames(frames_tensor)
        silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
        _write_mp4(frames, silent_path, fps=self.fps)

        from externals.video_audio.ffmpeg_io import attach_audio

        attach_audio(silent_path, audio_path, output_path, shortest=True)
        if silent_path.is_file() and silent_path != output_path:
            silent_path.unlink(missing_ok=True)
        return output_path


_RUNNER: ComfyAvatarRunner | None = None
_RUNNER_KEY: tuple[str, int] | None = None


def get_runner(*, work_dir: Path, fps: int = 24) -> ComfyAvatarRunner:
    global _RUNNER, _RUNNER_KEY
    from externals.comfy_inprocess.vram_config import apply_comfy_vram_settings

    key = (str(work_dir.resolve()), fps)
    if _RUNNER is None or _RUNNER_KEY != key:
        print(f"$avatar: comfy_lib backend ({work_dir})", flush=True)
        _RUNNER = ComfyAvatarRunner(work_dir=work_dir, fps=fps)
        _RUNNER_KEY = key
    apply_comfy_vram_settings()
    return _RUNNER


def run_comfy_avatar(
    *,
    work_dir: Path,
    image_path: Path,
    audio_path: Path,
    prompt: str,
    output_path: Path,
    negative_prompt: str = "",
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int = 4,
    num_frames: int = 400,
    frame_window_size: int = 81,
    motion_frame: int = 5,
    drop_frames: int = 12,
    cfg: float = 1.0,
    fps: int = 24,
    workflow_ref: str = "",
) -> Path:
    runner = get_runner(work_dir=work_dir, fps=fps)
    t0 = time.perf_counter()
    result = runner.run_avatar(
        image_path=image_path,
        audio_path=audio_path,
        prompt=prompt,
        output_path=output_path,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        num_frames=num_frames,
        frame_window_size=frame_window_size,
        motion_frame=motion_frame,
        drop_frames=drop_frames,
        cfg=cfg,
        workflow_ref=workflow_ref,
    )
    print(
        f"$avatar: finished in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return result
