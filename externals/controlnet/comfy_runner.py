"""Run Qwen-Image + InstantX Union ControlNet via comfy_lib."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from externals.controlnet.comfy_bootstrap import bootstrap_controlnet_comfy, get_nodes_module
from externals.controlnet.comfy_executor import (  # noqa: F401 — register handlers
    _qwen_clip_loader,
)
from externals.controlnet.comfy_workflow import build_controlnet_workflow
from externals.controlnet.model_paths import ensure_models
from externals.comfy_inprocess.executor import execute_prompt_legacy


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "$controlnet needs Pillow. Run: uv sync --extra media"
        ) from exc
    return Image


def _tensor_to_png_bytes(image_tensor) -> bytes:
    Image = _pil_image()
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _log_comfy_kitchen() -> None:
    try:
        import comfy_kitchen as ck
    except ImportError:
        print(
            "$controlnet: WARNING comfy_kitchen not installed — FP8 Qwen-Image load will fail. "
            "Run: UV_PROJECT_ENVIRONMENT=.venvs/media uv sync --extra media",
            flush=True,
        )
        return
    cuda = ck.list_backends().get("cuda", {})
    if cuda.get("disabled"):
        print(
            "$controlnet: WARNING comfy_kitchen cuda backend disabled — FP8 load may fail",
            flush=True,
        )
    else:
        print("$controlnet: comfy_kitchen cuda backend enabled", flush=True)


@dataclass
class ControlnetRunner:
    work_dir: Path
    use_gpu: bool = True
    _nodes: object | None = field(default=None, repr=False)
    _models_ready: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.resolve()
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        bootstrap_controlnet_comfy(input_dir=self.input_dir, output_dir=self.output_dir)
        _log_comfy_kitchen()
        if self.use_gpu:
            try:
                import comfy.model_management as mm

                print(f"$controlnet: comfy vram_state={mm.vram_state.name}", flush=True)
            except ImportError:
                pass

    @property
    def nodes(self):
        if self._nodes is None:
            print("$controlnet: loading comfy nodes…", flush=True)
            t0 = time.perf_counter()
            self._nodes = get_nodes_module()
            print(f"$controlnet: comfy nodes ready ({time.perf_counter() - t0:.1f}s)", flush=True)
        return self._nodes

    def ensure_models(self) -> None:
        if not self._models_ready:
            ensure_models()
            self._models_ready = True

    def run_job(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        source_image: Path | None,
        control_images: list[tuple[str, Path]],
        width: int,
        height: int,
        steps: int,
        cfg: float,
        denoise: float,
        strength: float,
        seed: int | None,
    ) -> bytes:
        self.ensure_models()
        workflow, _seed = build_controlnet_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            source_image_path=source_image,
            control_images=control_images,
            input_dir=self.input_dir,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            strength=strength,
        )
        output_id = workflow.pop("_output")
        print(
            f"$controlnet: sampling {width}x{height} steps={steps} "
            f"controls={len(control_images)} "
            f"mode={'img2img' if source_image is not None else 'txt2img'} "
            f"denoise={denoise}",
            flush=True,
        )
        t0 = time.perf_counter()
        outputs = execute_prompt_legacy(workflow, nodes_module=self.nodes)
        image_tensor = outputs[output_id][0]
        print(f"$controlnet: done ({time.perf_counter() - t0:.1f}s)", flush=True)
        return _tensor_to_png_bytes(image_tensor)


_RUNNER: ControlnetRunner | None = None


def get_runner(work_dir: Path, *, use_gpu: bool = True) -> ControlnetRunner:
    global _RUNNER
    if _RUNNER is None or _RUNNER.work_dir != work_dir.resolve():
        _RUNNER = ControlnetRunner(work_dir=work_dir, use_gpu=use_gpu)
    return _RUNNER
