"""Comfy-in-process Flux.2 Klein runner for $image and $image2image."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from externals.comfy_inprocess.executor import execute_prompt_legacy, find_node_id
from externals.flux2_klein import comfy_executor  # noqa: F401 — register handlers
from externals.flux2_klein.comfy_workflow import build_edit_prompt, build_txt2img_prompt
from externals.flux2_klein.model_paths import ensure_companion_assets, normalize_klein_steps_cfg, resolve_unet
from externals.image2image.comfy_bootstrap import bootstrap_comfy, get_nodes_module

_SAMPLE_ONLY = frozenset({"KSampler", "VAEDecode"})


def _log_comfy_kitchen_backend() -> None:
    try:
        import comfy_kitchen as ck
    except ImportError:
        print(
            "$flux2_klein: comfy_kitchen not installed — FP8 UNet will be slow",
            flush=True,
        )
        return
    cuda = ck.list_backends().get("cuda", {})
    if cuda.get("disabled"):
        print(
            "$flux2_klein: WARNING comfy_kitchen cuda DISABLED — FP8 sampling will crawl",
            flush=True,
        )
    else:
        print("$flux2_klein: comfy_kitchen cuda backend enabled", flush=True)


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for Flux.2 Klein. "
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
class KleinRunner:
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
        from externals.flux2_klein.comfy_sample_prep import apply_flux2_klein_vram_settings

        apply_flux2_klein_vram_settings()
        try:
            import comfy.model_management as mm

            print(f"$flux2_klein: comfy vram_state={mm.vram_state.name}", flush=True)
        except ImportError:
            pass
        _log_comfy_kitchen_backend()

    @property
    def nodes(self):
        if self._nodes is None:
            print(
                "$flux2_klein: loading comfy nodes (first job only, can take 1–2 min)…",
                flush=True,
            )
            t0 = time.perf_counter()
            self._nodes = get_nodes_module()
            print(
                f"$flux2_klein: comfy nodes ready ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        return self._nodes

    def run_txt2img(
        self,
        *,
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
        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, cfg)
        prompt_dict, _used_seed = build_txt2img_prompt(
            prompt=prompt,
            model_arg=model_arg,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
        )
        latent = next(
            n
            for n in prompt_dict.values()
            if n.get("class_type") == "EmptyFlux2LatentImage"
        )
        out_w = latent["inputs"]["width"]
        out_h = latent["inputs"]["height"]
        print(
            f"$flux2_klein: txt2img ({width}x{height} -> {out_w}x{out_h}, "
            f"{steps} steps, cfg={cfg})…",
            flush=True,
        )
        t_run = time.perf_counter()
        outputs = execute_prompt_legacy(prompt_dict, nodes_module=self.nodes)
        print(
            f"$flux2_klein: txt2img done ({time.perf_counter() - t_run:.1f}s)",
            flush=True,
        )
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from Klein txt2img workflow")
        return _tensor_to_pil(outputs[decode_id][0])

    def run_txt2img_variants(
        self,
        *,
        prompt: str,
        model_arg: str,
        steps: int,
        cfg: float,
        seeds: list[int | None],
        width: int,
        height: int,
        on_sample: Callable[[Any, int], None] | None = None,
    ) -> list:
        ensure_companion_assets()
        resolve_unet(model_arg)
        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, cfg)
        first_seed = seeds[0] if seeds else None
        prompt_dict, used_seed = build_txt2img_prompt(
            prompt=prompt,
            model_arg=model_arg,
            seed=first_seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
        )
        if seeds and all(s is None for s in seeds):
            seeds = [used_seed + i for i in range(len(seeds))]
        latent = next(
            n
            for n in prompt_dict.values()
            if n.get("class_type") == "EmptyFlux2LatentImage"
        )
        out_w = latent["inputs"]["width"]
        out_h = latent["inputs"]["height"]
        print(
            f"$flux2_klein: fast repeat — encode once, {len(seeds)} txt2img sample(s) "
            f"({width}x{height} -> {out_w}x{out_h}, {steps} steps)",
            flush=True,
        )
        base = execute_prompt_legacy(
            prompt_dict,
            nodes_module=self.nodes,
            stop_before_class="KSampler",
        )
        ks_id = find_node_id(prompt_dict, "KSampler")
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if ks_id is None or decode_id is None:
            raise RuntimeError("KSampler or VAEDecode missing from Klein txt2img workflow")

        results: list = []
        for vi, run_seed in enumerate(seeds):
            if run_seed is not None:
                prompt_dict[ks_id]["inputs"]["seed"] = run_seed
            print(
                f"$flux2_klein: sample {vi + 1}/{len(seeds)} "
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
            pil = _tensor_to_pil(out[decode_id][0])
            results.append(pil)
            if on_sample is not None:
                on_sample(pil, vi)
        return results

    def run_edit(
        self,
        *,
        image_path: Path,
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
        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, cfg)
        prompt_dict, _used_seed = build_edit_prompt(
            prompt=prompt,
            image_path=image_path,
            input_dir=self.input_dir,
            model_arg=model_arg,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
        )
        latent = next(
            n
            for n in prompt_dict.values()
            if n.get("class_type") == "EmptyFlux2LatentImage"
        )
        out_w = latent["inputs"]["width"]
        out_h = latent["inputs"]["height"]
        print(
            f"$flux2_klein: edit ({width}x{height} -> {out_w}x{out_h}, {steps} steps) "
            f"prompt={prompt[:80]!r}",
            flush=True,
        )
        t_run = time.perf_counter()
        outputs = execute_prompt_legacy(prompt_dict, nodes_module=self.nodes)
        print(
            f"$flux2_klein: edit done ({time.perf_counter() - t_run:.1f}s)",
            flush=True,
        )
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from Klein edit workflow")
        return _tensor_to_pil(outputs[decode_id][0])

    def run_edit_variants(
        self,
        *,
        image_path: Path,
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
        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, cfg)
        first_seed = seeds[0] if seeds else None
        prompt_dict, used_seed = build_edit_prompt(
            prompt=prompt,
            image_path=image_path,
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
        print(
            f"$flux2_klein: fast repeat — encode once, {len(seeds)} sample(s)",
            flush=True,
        )
        base = execute_prompt_legacy(
            prompt_dict,
            nodes_module=self.nodes,
            stop_before_class="KSampler",
        )
        ks_id = find_node_id(prompt_dict, "KSampler")
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if ks_id is None or decode_id is None:
            raise RuntimeError("KSampler or VAEDecode missing from Klein edit workflow")

        results: list = []
        for vi, run_seed in enumerate(seeds):
            if run_seed is not None:
                prompt_dict[ks_id]["inputs"]["seed"] = run_seed
            print(
                f"$flux2_klein: sample {vi + 1}/{len(seeds)} "
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


_RUNNER: KleinRunner | None = None
_RUNNER_KEY: tuple[str, bool] | None = None


def get_runner(*, work_dir: Path, use_gpu: bool) -> KleinRunner:
    global _RUNNER, _RUNNER_KEY
    key = (str(work_dir.resolve()), use_gpu)
    if _RUNNER is None or _RUNNER_KEY != key:
        print(f"$flux2_klein: comfy_lib backend ({work_dir})", flush=True)
        _RUNNER = KleinRunner(work_dir=work_dir, use_gpu=use_gpu)
        _RUNNER_KEY = key
    return _RUNNER


def run_klein_txt2img(
    *,
    work_dir: Path,
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
    return runner.run_txt2img(
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seed=seed,
        width=width,
        height=height,
    )


def run_klein_txt2img_variants(
    *,
    work_dir: Path,
    prompt: str,
    model_arg: str,
    steps: int,
    cfg: float,
    seeds: list[int | None],
    width: int,
    height: int,
    use_gpu: bool,
    on_sample: Callable[[Any, int], None] | None = None,
) -> list:
    runner = get_runner(work_dir=work_dir, use_gpu=use_gpu)
    return runner.run_txt2img_variants(
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seeds=seeds,
        width=width,
        height=height,
        on_sample=on_sample,
    )


def run_klein_edit(
    *,
    work_dir: Path,
    image_path: Path,
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
    return runner.run_edit(
        image_path=image_path,
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seed=seed,
        width=width,
        height=height,
    )


def run_klein_edit_variants(
    *,
    work_dir: Path,
    image_path: Path,
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
    return runner.run_edit_variants(
        image_path=image_path,
        prompt=prompt,
        model_arg=model_arg,
        steps=steps,
        cfg=cfg,
        seeds=seeds,
        width=width,
        height=height,
    )
