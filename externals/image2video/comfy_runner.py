"""Comfy-in-process Wan MEGA I2V runner (comfy_lib + Rapid-AIO-Mega workflow)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from externals.comfy_inprocess.bootstrap import bootstrap_comfy, get_nodes_module
from externals.comfy_inprocess.executor import execute_prompt, find_node_id
from externals.image2video.comfy_nodes import register_i2v_node_handlers
from externals.image2video.comfy_workflow import build_i2v_prompt_for_model


def _tensor_to_frames(image_tensor) -> list[np.ndarray]:
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        frames = arr
    elif arr.ndim == 3:
        frames = arr[np.newaxis, ...]
    else:
        raise RuntimeError(f"Unexpected VAEDecode shape: {arr.shape}")
    out: list[np.ndarray] = []
    for frame in frames:
        out.append((frame * 255.0).clip(0, 255).astype(np.uint8))
    return out


def _write_mp4(frames: list[np.ndarray], dest: Path, *, fps: int = 16) -> Path:
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
            "$image2video: install imageio or use diffusers for MP4 export"
        ) from exc


@dataclass
class ComfyI2VRunner:
    work_dir: Path
    fps: int = 16
    _nodes: object | None = field(default=None, repr=False)
    _handlers_registered: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.resolve()
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        bootstrap_comfy(input_dir=self.input_dir, output_dir=self.output_dir)

    @property
    def nodes(self):
        if not self._handlers_registered:
            register_i2v_node_handlers()
            self._handlers_registered = True
        if self._nodes is None:
            self._nodes = get_nodes_module()
        return self._nodes

    def run_i2v(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        model_arg: str,
        steps: int = 4,
        seed: int | None = None,
        width: int = 768,
        height: int = 1280,
        num_frames: int = 81,
        negative_prompt: str = "",
        guidance: float | None = 1.0,
        workflow_ref: str = "",
    ) -> Path:
        prompt_dict, _used_seed = build_i2v_prompt_for_model(
            prompt=prompt,
            image_path=image_path,
            input_dir=self.input_dir,
            model_arg=model_arg,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
            guidance=guidance,
            workflow_ref=workflow_ref,
        )
        outputs = execute_prompt(prompt_dict, nodes_module=self.nodes)
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from workflow")
        frames_tensor = outputs[decode_id][0]
        frames = _tensor_to_frames(frames_tensor)
        return _write_mp4(frames, output_path, fps=self.fps)


_RUNNER: ComfyI2VRunner | None = None
_RUNNER_KEY: tuple[str, int] | None = None


def get_runner(*, work_dir: Path, fps: int = 16) -> ComfyI2VRunner:
    global _RUNNER, _RUNNER_KEY
    key = (str(work_dir.resolve()), fps)
    if _RUNNER is None or _RUNNER_KEY != key:
        print(f"$image2video: comfy_lib backend ({work_dir})", flush=True)
        _RUNNER = ComfyI2VRunner(work_dir=work_dir, fps=fps)
        _RUNNER_KEY = key
    return _RUNNER


def run_comfy_i2v(
    *,
    work_dir: Path,
    image_path: Path,
    prompt: str,
    output_path: Path,
    model_arg: str,
    steps: int,
    seed: int | None,
    width: int,
    height: int,
    num_frames: int,
    negative_prompt: str = "",
    guidance: float | None = None,
    fps: int = 16,
) -> Path:
    runner = get_runner(work_dir=work_dir, fps=fps)
    t0 = time.perf_counter()
    result = runner.run_i2v(
        image_path=image_path,
        prompt=prompt,
        output_path=output_path,
        model_arg=model_arg,
        steps=steps,
        seed=seed,
        width=width,
        height=height,
        num_frames=num_frames,
        negative_prompt=negative_prompt,
        guidance=guidance,
    )
    print(
        f"$image2video: comfy I2V finished in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return result
