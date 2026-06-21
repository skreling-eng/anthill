"""Comfy-in-process Flux.2 Klein face swap runner."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from externals.comfy_inprocess.executor import execute_prompt_legacy, find_node_id
from externals.flux2_klein import comfy_executor  # noqa: F401 — register loaders
from externals.flux2_klein.model_paths import ensure_companion_assets, resolve_unet
from externals.image2image.comfy_bootstrap import bootstrap_comfy, get_nodes_module
from externals.image_face_swap.comfy_workflow import build_face_swap_prompt

_SAMPLE_ONLY = frozenset({"KSampler", "VAEDecode"})


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for $image_face_swap. "
            "Run: powershell -File tools\\setup_external_venvs.ps1"
        ) from exc
    return Image


def _tensor_to_pil(image_tensor):
    Image = _pil_image()
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class FaceSwapRunner:
    work_dir: Path
    use_gpu: bool = True
    _nodes: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.resolve()
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        bootstrap_comfy(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            vram_profile="image2image",
        )

    @property
    def nodes(self):
        if self._nodes is None:
            print(
                "$image_face_swap: loading comfy nodes (first job only)…",
                flush=True,
            )
            t0 = time.perf_counter()
            self._nodes = get_nodes_module()
            print(
                f"$image_face_swap: comfy nodes ready ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        return self._nodes

    def run_swap(
        self,
        *,
        target_path: Path,
        face_path: Path,
        prompt: str,
        model_arg: str,
        steps: int,
        cfg: float,
        seed: int | None,
        width: int,
        height: int,
    ):
        ensure_companion_assets()
        resolve_unet(model_arg)
        prompt_dict, _used_seed = build_face_swap_prompt(
            prompt=prompt,
            target_path=target_path,
            face_path=face_path,
            input_dir=self.input_dir,
            model_arg=model_arg,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
        )
        print(
            f"$image_face_swap: swap ({width}x{height}, {steps} steps) "
            f"target={target_path.name} face={face_path.name}",
            flush=True,
        )
        t_run = time.perf_counter()
        outputs = execute_prompt_legacy(prompt_dict, nodes_module=self.nodes)
        print(
            f"$image_face_swap: done ({time.perf_counter() - t_run:.1f}s)",
            flush=True,
        )
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from face swap workflow")
        return _tensor_to_pil(outputs[decode_id][0])

    def run_swap_variants(
        self,
        *,
        target_path: Path,
        face_path: Path,
        prompt: str,
        model_arg: str,
        steps: int,
        cfg: float,
        seeds: list[int | None],
        width: int,
        height: int,
    ) -> list:
        ensure_companion_assets()
        resolve_unet(model_arg)
        first_seed = seeds[0] if seeds else None
        prompt_dict, used_seed = build_face_swap_prompt(
            prompt=prompt,
            target_path=target_path,
            face_path=face_path,
            input_dir=self.input_dir,
            model_arg=model_arg,
            seed=first_seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
        )
        if seeds and all(s is None for s in seeds):
            seeds = [used_seed + i for i in range(len(seeds))]
        base = execute_prompt_legacy(
            prompt_dict,
            nodes_module=self.nodes,
            stop_before_class="KSampler",
        )
        ks_id = find_node_id(prompt_dict, "KSampler")
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if ks_id is None or decode_id is None:
            raise RuntimeError("KSampler or VAEDecode missing from face swap workflow")

        results: list = []
        for vi, run_seed in enumerate(seeds):
            if run_seed is not None:
                prompt_dict[ks_id]["inputs"]["seed"] = run_seed
            print(
                f"$image_face_swap: sample {vi + 1}/{len(seeds)} "
                f"seed={prompt_dict[ks_id]['inputs'].get('seed', '?')}",
                flush=True,
            )
            out = execute_prompt_legacy(
                prompt_dict,
                nodes_module=self.nodes,
                only_classes=_SAMPLE_ONLY,
                initial_outputs=base,
                prepare_ksampler=True,
            )
            results.append(_tensor_to_pil(out[decode_id][0]))
        return results


_RUNNER: FaceSwapRunner | None = None
_RUNNER_KEY: tuple[str, bool] | None = None


def get_runner(*, work_dir: Path, use_gpu: bool) -> FaceSwapRunner:
    global _RUNNER, _RUNNER_KEY
    key = (str(work_dir.resolve()), use_gpu)
    if _RUNNER is None or _RUNNER_KEY != key:
        _RUNNER = FaceSwapRunner(work_dir=work_dir, use_gpu=use_gpu)
        _RUNNER_KEY = key
    return _RUNNER


def run_face_swap(
    *,
    work_dir: Path,
    target_path: Path,
    face_path: Path,
    prompt: str,
    model_arg: str,
    steps: int,
    cfg: float,
    seed: int | None,
    width: int,
    height: int,
    use_gpu: bool,
):
    runner = get_runner(work_dir=work_dir, use_gpu=use_gpu)
    return runner.run_swap(
        target_path=target_path,
        face_path=face_path,
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seed=seed,
        width=width,
        height=height,
    )


def run_face_swap_variants(
    *,
    work_dir: Path,
    target_path: Path,
    face_path: Path,
    prompt: str,
    model_arg: str,
    steps: int,
    cfg: float,
    seeds: list[int | None],
    width: int,
    height: int,
    use_gpu: bool,
) -> list:
    runner = get_runner(work_dir=work_dir, use_gpu=use_gpu)
    return runner.run_swap_variants(
        target_path=target_path,
        face_path=face_path,
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seeds=seeds,
        width=width,
        height=height,
    )
