"""$avatar — talking avatar video from portrait + speech (SkyReels V3 A2V)."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.avatar.model_paths import DEFAULT_NEGATIVE_PROMPT, audio_frame_budget
from ahlib.ah_runtime import ArrayBundle

_SOUND_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _output_name(index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_avatar_{index}.mp4"


def _optional_int(args: dict[str, str], key: str) -> int | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return int(raw)


def _optional_float(args: dict[str, str], key: str) -> float | None:
    raw = args.get(key, "").strip()
    if not raw:
        return None
    return float(raw)


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_text(ctx: ExternalContext, inp: ExternalInput) -> str:
    if inp.args.get("prompt", "").strip():
        return inp.args["prompt"].strip()
    parts = read_prompt_texts(ctx, inp)
    if parts:
        return "\n".join(parts)
    return inp.prompt_text.strip()


def _read_negprompts(ctx: ExternalContext, inp: ExternalInput) -> str:
    neg = inp.args.get("neg", inp.args.get("negative_prompt", "")).strip()
    if neg:
        return neg
    parts: list[str] = []
    for link in inp.bundle.negprompts:
        text = ctx.read_link_text(link)
        if text:
            parts.append(text)
    if parts:
        return "\n\n".join(parts)
    return DEFAULT_NEGATIVE_PROMPT


def _media_paths(ctx: ExternalContext, bundle: ArrayBundle, arr_key: str) -> list[Path]:
    paths: list[Path] = []
    for link in getattr(bundle, arr_key):
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _path_to_link(ctx: ExternalContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ctx.base_dir.resolve())).replace("\\", "/")
    except ValueError:
        pass
    dest = ctx.op_dir / "videos" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return str(dest.relative_to(ctx.base_dir)).replace("\\", "/")


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_AVATAR", "").lower() in ("1", "true", "yes")


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    prompt: str,
    images: list[Path],
    sounds: list[Path],
) -> None:
    pairs = list(zip(images, sounds)) if images and sounds else []
    if not pairs:
        count = max(len(images), len(sounds), 1)
        pairs = [
            (
                images[i] if i < len(images) else None,
                sounds[i] if i < len(sounds) else None,
            )
            for i in range(count)
        ]
    for image_path, sound_path in pairs:
        content = (
            "[emulated $avatar]\n"
            f"prompt: {prompt}\n"
            f"image: {getattr(image_path, 'name', None)}\n"
            f"sound: {getattr(sound_path, 'name', None)}\n"
        )
        out.videos.append(ctx.new_link("videos", ".mp4", content.encode("utf-8")))


def _help() -> str:
    return (
        "$avatar uses comfy_lib in-process with SkyReels-V3-Talking-Avatars_api.json.\n"
        "  Requires portrait in images[] and speech in sounds[].\n"
        "  Worker venv: .venvs/comfy-wan (same as $image2video).\n"
        "  Models: Wan21-SkyReelsV3-A2V_fp8_scaled_mixed.safetensors, "
        "Wan2_1_VAE_bf16.safetensors, umt5-xxl-enc-bf16.safetensors.\n"
        "Test without GPU/model: AH_EMULATE_AVATAR=1"
    )


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.prompts.clear()

    prompt = _prompt_text(ctx, inp)
    negative_prompt = _read_negprompts(ctx, inp)
    image_paths = _media_paths(ctx, inp.bundle, "images")
    sound_paths = _media_paths(ctx, inp.bundle, "sounds")

    if _emulate_enabled():
        _emulate(ctx, out, prompt=prompt, images=image_paths, sounds=sound_paths)
        out.images.clear()
        out.sounds.clear()
        return out

    if not image_paths:
        raise ValueError("$avatar requires at least one image in images[] (portrait)")
    if not sound_paths:
        raise ValueError("$avatar requires at least one sound in sounds[] (speech audio)")
    if not prompt:
        raise ValueError(
            "$avatar requires a prompt (instruction body on the $avatar step, "
            "prompts[], or prompt=). If the wrapper @instruction also runs $llm "
            "upstream, put the motion prompt on a separate @$avatar step — see "
            "examples/example_avatar.ah"
        )

    width = _optional_int(inp.args, "width")
    height = _optional_int(inp.args, "height")
    steps = _optional_int(inp.args, "steps") or 4
    seed = _optional_int(inp.args, "seed")
    num_frames = _optional_int(inp.args, "frames") or _int_arg(inp.args, "max_frames", 400)
    blocks_to_swap = _optional_int(inp.args, "blocks_to_swap")
    frame_window = _int_arg(inp.args, "frame_window", 81)
    motion_frame = _int_arg(inp.args, "motion_frame", 5)
    drop_frames = _int_arg(inp.args, "drop_frames", 12)
    cfg = _optional_float(inp.args, "cfg") or 1.0
    fps = _int_arg(inp.args, "fps", 24)
    workflow_ref = inp.args.get("workflow", inp.args.get("json", "")).strip()

    from externals.avatar.model_paths import configure_avatar_tiled_vae_for_job
    from externals.comfy_inprocess.vram_config import configure_avatar_vram_for_job

    tiled_vae = configure_avatar_tiled_vae_for_job(inp.args)
    configure_avatar_vram_for_job(inp.args)

    from externals.comfy_inprocess.vram_config import apply_avatar_vram_settings

    apply_avatar_vram_settings()

    work_dir = ctx.op_dir / "comfy_work"
    videos_dir = ctx.op_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    if len(image_paths) == len(sound_paths):
        pairs = list(zip(image_paths, sound_paths, strict=True))
    elif len(image_paths) == 1:
        pairs = [(image_paths[0], sound) for sound in sound_paths]
    elif len(sound_paths) == 1:
        pairs = [(image, sound_paths[0]) for image in image_paths]
    else:
        raise ValueError(
            "$avatar needs equal images[] and sounds[], or one side length 1; "
            f"got {len(image_paths)} image(s) and {len(sound_paths)} sound(s)"
        )

    try:
        from externals.avatar.comfy_runner import run_comfy_avatar

        for index, (image_path, sound_path) in enumerate(pairs):
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                from ahlib.ah_runtime import RuntimeCancelled

                raise RuntimeCancelled("$avatar cancelled")
            out_path = videos_dir / _output_name(index)
            run_seed = (seed + index) if seed else seed
            job_frames = min(
                num_frames,
                audio_frame_budget(sound_path, float(fps), cap=num_frames),
            )
            print(
                f"$avatar: job {index + 1}/{len(pairs)} "
                f"image={image_path.name} audio={sound_path.name} "
                f"frames={job_frames}",
                flush=True,
            )
            run_comfy_avatar(
                work_dir=work_dir,
                image_path=image_path,
                audio_path=sound_path,
                prompt=prompt,
                output_path=out_path,
                negative_prompt=negative_prompt,
                seed=run_seed,
                width=width,
                height=height,
                steps=steps,
                num_frames=job_frames,
                frame_window_size=frame_window,
                motion_frame=motion_frame,
                drop_frames=drop_frames,
                cfg=cfg,
                fps=fps,
                workflow_ref=workflow_ref,
                blocks_to_swap=blocks_to_swap,
                tiled_vae=tiled_vae,
            )
            out.videos.append(_path_to_link(ctx, out_path))
    except ImportError as exc:
        raise RuntimeError(_help()) from exc

    out.images.clear()
    out.sounds.clear()
    return out
