"""Comfy-in-process Qwen-Rapid-AIO runner (comfy_lib + workflow JSON)."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from externals.image2image.comfy_bootstrap import bootstrap_comfy, get_nodes_module
from externals.image2image.comfy_executor import execute_prompt_legacy, find_node_id

_SAMPLE_ONLY = frozenset({"KSampler", "VAEDecode"})
from externals.image2image.comfy_workflow import build_edit_prompt
from externals.image2image.model_paths import is_klein_model, resolve_checkpoint

_PIL_INSTALL_HINT = (
    "Pillow is required for $image2image. "
    "Fresh checkout: powershell -File tools\\init.ps1 "
    "(creates .venvs/media with --extra media). "
    "Or: powershell -File tools\\setup_external_venvs.ps1 "
    "and set AH_EXTERNAL_VENV_image2image=.venvs/media in .env"
)


def _log_comfy_kitchen_backend() -> None:
    """Log whether FP8 fast paths are active (after comfy bootstrap imports quant_ops)."""
    try:
        import comfy_kitchen as ck
    except ImportError:
        print(
            "$image2image: comfy_kitchen not installed — FP8 UNet will be slow",
            flush=True,
        )
        return
    cuda = ck.list_backends().get("cuda", {})
    if cuda.get("disabled"):
        print(
            "$image2image: WARNING comfy_kitchen cuda backend DISABLED — "
            f"FP8 sampling will be very slow (torch.version.cuda, backends={cuda})",
            flush=True,
        )
    else:
        print("$image2image: comfy_kitchen cuda backend enabled", flush=True)


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(_PIL_INSTALL_HINT) from exc
    return Image


def _tensor_to_pil(image_tensor):
    Image = _pil_image()
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
        bootstrap_comfy(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            vram_profile="image2image",
        )
        try:
            import comfy.model_management as mm

            print(f"$image2image: comfy vram_state={mm.vram_state.name}", flush=True)
        except ImportError:
            pass
        _log_comfy_kitchen_backend()

    @property
    def nodes(self):
        if self._nodes is None:
            print(
                "$image2image: loading comfy nodes (first job only, can take 1–2 min)…",
                flush=True,
            )
            t0 = time.perf_counter()
            self._nodes = get_nodes_module()
            print(
                f"$image2image: comfy nodes ready ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
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
        print(
            f"$image2image: executing workflow ({ckpt_name}, {width}x{height}, {steps} steps)…",
            flush=True,
        )
        t_run = time.perf_counter()
        outputs = execute_prompt_legacy(prompt_dict, nodes_module=self.nodes)
        print(
            f"$image2image: workflow done ({time.perf_counter() - t_run:.1f}s)",
            flush=True,
        )
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if decode_id is None:
            raise RuntimeError("VAEDecode node missing from workflow")
        vae_out = outputs[decode_id][0]
        return _tensor_to_pil(vae_out)

    def run_edit_variants(
        self,
        *,
        image_paths: list[Path],
        prompt: str,
        checkpoint: Path,
        steps: int,
        seeds: list[int | None],
        width: int,
        height: int,
        workflow_ref: str = "",
        on_variant: Callable | None = None,
    ) -> list:
        """One vision encode, multiple KSampler+decode runs (repeat with same prompt/image)."""
        ckpt_name = checkpoint.name
        first_seed = seeds[0] if seeds else None
        prompt_dict, used_seed = build_edit_prompt(
            prompt=prompt,
            image_paths=image_paths,
            input_dir=self.input_dir,
            checkpoint_name=ckpt_name,
            seed=first_seed,
            width=width,
            height=height,
            steps=steps,
            workflow_ref=workflow_ref,
        )
        # If caller didn't provide seeds (or provided None placeholders), expand the
        # workflow's actual seed into distinct seeds for each variant.
        if seeds and all(s is None for s in seeds):
            seeds = [used_seed + i for i in range(len(seeds))]
        print(
            f"$image2image: fast repeat — encode once, {len(seeds)} sample(s) "
            f"({ckpt_name}, {width}x{height})",
            flush=True,
        )
        t_enc = time.perf_counter()
        base = execute_prompt_legacy(
            prompt_dict,
            nodes_module=self.nodes,
            stop_before_class="KSampler",
        )
        print(
            f"$image2image: shared encode done ({time.perf_counter() - t_enc:.1f}s)",
            flush=True,
        )
        ks_id = find_node_id(prompt_dict, "KSampler")
        decode_id = find_node_id(prompt_dict, "VAEDecode")
        if ks_id is None or decode_id is None:
            raise RuntimeError("KSampler or VAEDecode missing from workflow")

        results: list = []
        for vi, run_seed in enumerate(seeds):
            if run_seed is not None:
                prompt_dict[ks_id]["inputs"]["seed"] = run_seed
            print(
                f"$image2image: sample {vi + 1}/{len(seeds)} "
                f"seed={prompt_dict[ks_id]['inputs'].get('seed', '?')}",
                flush=True,
            )
            t_s = time.perf_counter()
            out = execute_prompt_legacy(
                prompt_dict,
                nodes_module=self.nodes,
                only_classes=_SAMPLE_ONLY,
                initial_outputs=base,
                # Re-prepare UNet every variant; this avoids slowdowns on later
                # samples when model residency drifts after prior denoise passes.
                prepare_ksampler=True,
            )
            print(
                f"$image2image: sample done ({time.perf_counter() - t_s:.1f}s)",
                flush=True,
            )
            pil = _tensor_to_pil(out[decode_id][0])
            results.append(pil)
            if on_variant is not None:
                on_variant(pil)
        return results


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


def run_comfy_edit_variants(
    *,
    work_dir: Path,
    image_paths: list[Path],
    prompt: str,
    model_arg: str,
    steps: int,
    seeds: list[int | None],
    width: int,
    height: int,
    use_gpu: bool,
    on_variant: Callable | None = None,
) -> list:
    if is_klein_model(model_arg):
        from externals.flux2_klein.comfy_runner import run_klein_edit_variants

        if len(image_paths) != 1:
            image_paths = [image_paths[0]]
        from externals.flux2_klein.model_paths import normalize_klein_steps_cfg

        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, 4.0)
        return run_klein_edit_variants(
            work_dir=work_dir,
            image_path=image_paths[0],
            prompt=prompt,
            model_arg=model_arg,
            steps=steps,
            cfg=cfg,
            seeds=seeds,
            width=width,
            height=height,
            use_gpu=use_gpu,
            on_variant=on_variant,
        )

    checkpoint = resolve_checkpoint(model_arg)
    runner = get_runner(work_dir=work_dir, use_gpu=use_gpu)
    return runner.run_edit_variants(
        image_paths=image_paths,
        prompt=prompt,
        checkpoint=checkpoint,
        steps=steps,
        seeds=seeds,
        width=width,
        height=height,
        on_variant=on_variant,
    )


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
    if is_klein_model(model_arg):
        from externals.flux2_klein.comfy_runner import run_klein_edit

        if len(image_paths) != 1:
            image_paths = [image_paths[0]]
        from externals.flux2_klein.model_paths import normalize_klein_steps_cfg

        steps, cfg = normalize_klein_steps_cfg(model_arg, steps, 4.0)
        t0 = time.perf_counter()
        result = run_klein_edit(
            work_dir=work_dir,
            image_path=image_paths[0],
            prompt=prompt,
            model_arg=model_arg,
            steps=steps,
            cfg=cfg,
            seed=seed,
            width=width,
            height=height,
            use_gpu=use_gpu,
        )
        print(
            f"$image2image: klein edit finished in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return result

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
