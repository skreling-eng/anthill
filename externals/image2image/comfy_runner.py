"""Comfy-in-process Qwen-Rapid-AIO runner (comfy_lib + workflow JSON)."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from externals.image2image.comfy_bootstrap import bootstrap_comfy, get_nodes_module
from externals.image2image.comfy_executor import execute_prompt, find_node_id
from externals.image2image.comfy_workflow import build_edit_prompt
from externals.image2image.model_paths import resolve_checkpoint


def _tensor_to_pil(image_tensor) -> Image.Image:
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class ComfyRunner:
    """Keeps Comfy bootstrap dirs; checkpoint is loaded each job via workflow."""

    work_dir: Path
    use_gpu: bool = True
    _nodes: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.resolve()
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        bootstrap_comfy(input_dir=self.input_dir, output_dir=self.output_dir)

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = get_nodes_module()
        return self._nodes

    def run_edit(
        self,
        *,
        image_paths: list[Path],
        prompt: str,
        checkpoint: Path,
        steps: int = 4,
        seed: int | None = None,
        width: int = 720,
        height: int = 1280,
        workflow_ref: str = "",
    ) -> Image.Image:
        ckpt_name = checkpoint.name
        prompt_dict, _used_seed = build_edit_prompt(
            prompt=prompt,
            image_paths=image_paths,
            input_dir=self.input_dir,
            checkpoint_name=ckpt_name,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            workflow_ref=workflow_ref,
        )
        outputs = execute_prompt(prompt_dict, nodes_module=self.nodes)
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from workflow")
        vae_out = outputs[decode_id][0]
        return _tensor_to_pil(vae_out)


_RUNNER: ComfyRunner | None = None
_RUNNER_KEY: tuple[str, bool] | None = None


def get_runner(*, work_dir: Path, use_gpu: bool) -> ComfyRunner:
    global _RUNNER, _RUNNER_KEY
    key = (str(work_dir.resolve()), use_gpu)
    if _RUNNER is None or _RUNNER_KEY != key:
        print(f"$image2image: comfy_lib backend ({work_dir})", flush=True)
        _RUNNER = ComfyRunner(work_dir=work_dir, use_gpu=use_gpu)
        _RUNNER_KEY = key
    return _RUNNER


def run_comfy_edit(
    *,
    work_dir: Path,
    image_paths: list[Path],
    prompt: str,
    model_arg: str,
    steps: int,
    seed: int | None,
    width: int,
    height: int,
    use_gpu: bool,
) -> Image.Image:
    checkpoint = resolve_checkpoint(model_arg)
    runner = get_runner(work_dir=work_dir, use_gpu=use_gpu)
    t0 = time.perf_counter()
    result = runner.run_edit(
        image_paths=image_paths,
        prompt=prompt,
        checkpoint=checkpoint,
        steps=steps,
        seed=seed,
        width=width,
        height=height,
    )
    print(
        f"$image2image: comfy edit finished in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return result


def save_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
