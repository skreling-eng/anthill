"""$image — text-to-image via fastimage + model_list."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_arg_list, read_prompt_texts
from ahlib.ah_runtime import ArrayBundle


def _image_file_prefix(model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w.-]", "_", model)
    return f"{ts}_{safe_model}_"


def _repeat_count(inp: ExternalInput) -> int:
    """Variants per prompt ($image(...)[n] or count= arg)."""
    if inp.repeat > 1:
        return inp.repeat
    return max(1, int(inp.args.get("count", "1")))


def _truthy_arg(args: dict[str, str], key: str) -> bool:
    return args.get(key, "").strip().lower() in ("1", "true", "yes")


def _append_model_name_text(
    ctx: ExternalContext, out: ArrayBundle, model_name: str
) -> None:
    out.texts.append(ctx.new_link("texts", ".txt", model_name + "\n"))


def _emulate(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    models: list[str],
    prompts: list[str],
    count: int,
    *,
    model_names_to_texts: bool,
) -> ArrayBundle:
    for model in models:
        for pi, prompt in enumerate(prompts):
            for vi in range(count):
                content = (
                    f"[emulated $image model={model} prompt={pi} variant={vi}]\n"
                    f"{prompt}\n"
                )
                link = ctx.new_link("images", ".png", content)
                out.images.append(link)
                if model_names_to_texts:
                    _append_model_name_text(ctx, out, model)
    return out


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _path_to_link(ctx: ExternalContext, path: str) -> str:
    """Return session-relative link for an absolute or cwd-relative file path."""
    file_path = Path(path).resolve()
    try:
        return str(file_path.relative_to(ctx.base_dir.resolve())).replace("\\", "/")
    except ValueError:
        pass
    dest_name = file_path.name
    existing = len(list((ctx.op_dir / "images").glob("*.png"))) if (ctx.op_dir / "images").exists() else 0
    dest = ctx.op_dir / "images" / f"{existing}_{dest_name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return str(dest.relative_to(ctx.base_dir)).replace("\\", "/")


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_IMAGE", "").lower() in ("1", "true", "yes")


def _help() -> str:
    return (
        "$image uses .venvs/media (torch + diffusers + bitsandbytes for NF4 Flux).\n"
        "  Run once: tools\\setup_external_venvs.ps1\n"
        "  Set AH_EXTERNAL_VENV_image=.venvs/media in .env\n"
        "  Models: models/FLUX.1-dev, models/flux.1-dev-nf4-pkg, models/flux/…\n"
        "  flux_fusion_v2: models/flux/fluxFusionV24StepsGGUFNF4_v1Fp16AIO.safetensors\n"
        "  flux2_klein_fp8: models/flux2klein/flux2Klein9bFp8_fp8.safetensors "
        "(+ text_encoders/qwen_3_8b_fp8mixed + vae/flux2-vae)\n"
        "Test without GPU/models: AH_EMULATE_IMAGE=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    models = read_arg_list(inp, "model", "default")
    prompts = read_prompt_texts(ctx, inp)
    if not prompts:
        prompts = [""]
    count = _repeat_count(inp)
    seed = int(inp.args.get("seed", "0"))
    neg = inp.args.get("neg", "")
    width = _optional_int(inp.args, "width") or 0
    height = _optional_int(inp.args, "height") or 0
    steps = _optional_int(inp.args, "steps")
    model_names_to_texts = _truthy_arg(inp.args, "model_names_to_texts")

    if _emulate_enabled():
        return _emulate(
            ctx, inp, out, models, prompts, count,
            model_names_to_texts=model_names_to_texts,
        )

    try:
        from externals.image.model_list import get_image_gen

        output_dir = str(ctx.op_dir / "images")
        next_index = 0
        for model_name in models:
            gen = get_image_gen(model_name)
            if steps is not None:
                gen.steps = steps
            file_prefix = _image_file_prefix(model_name)
            for i, prompt in enumerate(prompts):
                run_seed = seed + i if seed else seed
                paths = gen.gen(
                    prompt,
                    seed=run_seed,
                    count=count,
                    neg=neg,
                    width=width,
                    height=height,
                    output_dir=output_dir,
                    start_index=next_index,
                    file_prefix=file_prefix,
                )
                next_index += len(paths)
                for path in paths:
                    out.images.append(_path_to_link(ctx, path))
                    if model_names_to_texts:
                        _append_model_name_text(ctx, out, model_name)
    except ImportError as exc:
        raise RuntimeError(_help()) from exc
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc
    except OSError as exc:
        raise RuntimeError(f"{exc}\n\n{_help()}") from exc

    return out
