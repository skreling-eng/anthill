"""$comfy — run a ComfyUI API workflow with INPUT_* placeholder patching."""

from __future__ import annotations

import copy
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from externals.comfy.client import (
    IMAGE_PLACEHOLDERS,
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_IMAGE_ALT,
    PLACEHOLDER_PROMPT,
    PLACEHOLDER_SEED,
    PLACEHOLDER_SOUND,
    SEED_MAX,
    SEED_MIN,
    ComfyClient,
    load_workflow_json,
    patch_placeholders,
    patch_seed_placeholder,
    resolve_workflow_path,
    workflow_contains,
    workflow_contains_any,
)
from ahlib.ah_runtime import ArrayBundle

_SOUND_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov"}


def _comfy_base_url(inp: ExternalInput) -> str:
    port_raw = inp.args.get("port", "").strip()
    if port_raw:
        return f"http://127.0.0.1:{int(port_raw)}"
    return os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")


def _workflow_ref(inp: ExternalInput) -> str:
    ref = inp.args.get("json", "").strip() or inp.args.get("workflow", "").strip()
    if not ref:
        ref = inp.args.get("_path", "").strip()
    if not ref:
        raise ValueError(
            "$comfy requires json='workflow_api.json' "
            "(API-format export from ComfyUI)."
        )
    return ref


def _file_links(ctx: ExternalContext, bundle: ArrayBundle, arr_key: str) -> list[Path]:
    paths: list[Path] = []
    for link in getattr(bundle, arr_key):
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _prompts_list(ctx: ExternalContext, inp: ExternalInput) -> list[str]:
    parts = read_prompt_texts(ctx, inp)
    if parts:
        return parts
    text = inp.prompt_text.strip()
    return [text] if text else [""]


def _array_for_extension(ext: str) -> str:
    low = ext.lower()
    if low in _IMAGE_EXTS:
        return "images"
    if low in _VIDEO_EXTS:
        return "videos"
    if low in _SOUND_EXTS:
        return "sounds"
    return "files"


def _collect_history_outputs(
    client: ComfyClient,
    ctx: ExternalContext,
    history: dict[str, Any],
    op_dir: Path,
) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {
        "images": [],
        "videos": [],
        "sounds": [],
        "files": [],
    }
    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        return collected

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = 0

    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for bucket in ("images", "gifs", "videos", "audio"):
            items = node_out.get(bucket) or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or item.get("name")
                if not filename:
                    continue
                subfolder = item.get("subfolder", "") or ""
                ftype = item.get("type", "output")
                ext = Path(filename).suffix or ".bin"
                arr_key = _array_for_extension(ext)
                if bucket == "gifs" and arr_key == "images":
                    arr_key = "videos"
                dest_dir = op_dir / arr_key
                dest_dir.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^\w.-]", "_", Path(filename).stem)[:80]
                dest = dest_dir / f"{stamp}_comfy_{idx}_{safe}{ext}"
                idx += 1
                client.download_view(filename, subfolder, ftype, dest)
                link = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
                collected[arr_key].append(link)

    return collected


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    workflow_path: Path,
    prompt: str,
    image_paths: list[Path],
    sound: Path | None,
    needs_image: bool,
) -> ArrayBundle:
    runs = image_paths if needs_image and image_paths else [None]
    total = len(runs)
    for idx, image in enumerate(runs):
        text = (
            f"[emulated $comfy run {idx + 1}/{total}]\n"
            f"workflow: {workflow_path.name}\n"
            f"prompt: {prompt}\n"
            f"image: {image}\n"
            f"sound: {sound}\n"
        )
        out.texts.append(ctx.new_link("texts", ".txt", text))
        if image:
            try:
                link = str(image.relative_to(ctx.base_dir)).replace("\\", "/")
            except ValueError:
                link = str(image).replace("\\", "/")
            out.images.append(link)
    if sound:
        try:
            link = str(sound.relative_to(ctx.base_dir)).replace("\\", "/")
        except ValueError:
            link = str(sound).replace("\\", "/")
        out.sounds.append(link)
    return out


def _run_comfy_job(
    client: ComfyClient,
    ctx: ExternalContext,
    workflow: dict[str, Any],
    *,
    prompt: str,
    image_path: Path | None,
    sound_path: Path | None,
    needs_image: bool,
    needs_sound: bool,
    needs_prompt: bool,
    needs_seed: bool,
    run_index: int,
    run_total: int,
) -> dict[str, list[str]]:
    replacements: dict[str, str] = {}
    if needs_prompt and prompt:
        replacements[PLACEHOLDER_PROMPT] = prompt
    if needs_image and image_path:
        image_name = client.upload_image(image_path)
        replacements[PLACEHOLDER_IMAGE] = image_name
        replacements[PLACEHOLDER_IMAGE_ALT] = image_name
    if needs_sound and sound_path:
        replacements[PLACEHOLDER_SOUND] = client.stage_input_file(
            sound_path, target_name=sound_path.name
        )

    patched = patch_placeholders(copy.deepcopy(workflow), replacements)
    if needs_seed:
        seed = random.randint(SEED_MIN, SEED_MAX)
        patched = patch_seed_placeholder(patched, seed)
        print(f"$comfy: seed={seed}", flush=True)
    suffix = f"_{run_index}" if run_total > 1 else ""
    (ctx.op_dir / f"patched_workflow{suffix}.json").write_text(
        json.dumps(patched, indent=2), encoding="utf-8"
    )

    label = image_path.name if image_path else "no image"
    prompt_hint = (prompt[:40] + "…") if len(prompt) > 40 else prompt
    print(
        f"$comfy: queue {run_index + 1}/{run_total} ({label}) prompt={prompt_hint!r}",
        flush=True,
    )
    prompt_id = client.queue_prompt(patched)
    print(f"$comfy: queued prompt_id={prompt_id}", flush=True)
    history = client.wait_history(prompt_id)
    return _collect_history_outputs(client, ctx, history, ctx.op_dir)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    workflow_path = resolve_workflow_path(_workflow_ref(inp), base_dir=ctx.base_dir)
    workflow = load_workflow_json(workflow_path)
    prompts_list = _prompts_list(ctx, inp)

    image_paths = _file_links(ctx, inp.bundle, "images")
    sound_path = _file_links(ctx, inp.bundle, "sounds")
    sound_path = sound_path[0] if sound_path else None

    needs_image = workflow_contains_any(workflow, IMAGE_PLACEHOLDERS)
    needs_sound = workflow_contains(workflow, PLACEHOLDER_SOUND)
    needs_prompt = workflow_contains(workflow, PLACEHOLDER_PROMPT)
    needs_seed = workflow_contains(workflow, PLACEHOLDER_SEED)

    emulate = os.environ.get("AH_EMULATE_COMFY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if not emulate:
        if needs_image and not image_paths:
            raise ValueError(
                f"$comfy: workflow {workflow_path.name} uses {PLACEHOLDER_IMAGE} "
                f"or {PLACEHOLDER_IMAGE_ALT} but images[] is empty."
            )
        if needs_sound and not sound_path:
            raise ValueError(
                f"$comfy: workflow {workflow_path.name} uses {PLACEHOLDER_SOUND} "
                "but sounds[] is empty."
            )
        if needs_prompt and not any(p.strip() for p in prompts_list):
            raise ValueError(
                f"$comfy: workflow {workflow_path.name} uses {PLACEHOLDER_PROMPT} "
                "but no prompt (prompts[] or instruction body)."
            )

    if emulate:
        for prompt in prompts_list:
            _emulate(
                ctx,
                out,
                workflow_path=workflow_path,
                prompt=prompt,
                image_paths=image_paths,
                sound=sound_path,
                needs_image=needs_image,
            )
        return out

    base_url = _comfy_base_url(inp)
    client = ComfyClient(base_url)
    print(f"$comfy: {base_url} workflow={workflow_path.name}", flush=True)

    if needs_image:
        run_images: list[Path | None] = list(image_paths)
    else:
        run_images = [image_paths[0] if image_paths else None]

    jobs: list[tuple[Path | None, str]] = [
        (image_path, prompt)
        for image_path in run_images
        for prompt in prompts_list
    ]
    all_collected: dict[str, list[str]] = {
        "images": [],
        "videos": [],
        "sounds": [],
        "files": [],
    }
    run_total = len(jobs)
    for run_index, (image_path, prompt) in enumerate(jobs):
        collected = _run_comfy_job(
            client,
            ctx,
            workflow,
            prompt=prompt,
            image_path=image_path,
            sound_path=sound_path,
            needs_image=needs_image,
            needs_sound=needs_sound,
            needs_prompt=needs_prompt,
            needs_seed=needs_seed,
            run_index=run_index,
            run_total=run_total,
        )
        for arr_key, links in collected.items():
            all_collected[arr_key].extend(links)

    for arr_key, links in all_collected.items():
        getattr(out, arr_key).extend(links)

    if not any(all_collected.values()):
        raise RuntimeError(
            "$comfy: workflow finished but produced no downloadable outputs "
            "(images/videos/sounds in history)."
        )

    out.prompts.clear()
    return out
