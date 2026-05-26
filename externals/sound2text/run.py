"""$sound2text — transcribe audio via OpenAI Whisper (openai-whisper package)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from externals.api import ExternalContext, ExternalInput, read_prompt_texts
from ahlib.ah_runtime import ArrayBundle

_WHISPER_MODELS = frozenset(
    {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        "turbo",
    }
)

_MODEL_CACHE: dict[str, object] = {}


def _sound_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.sounds:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _whisper_help() -> str:
    return (
        "$sound2text requires openai-whisper and ffmpeg.\n"
        "  uv sync --extra sound2text\n"
        "  ffmpeg must be on PATH (see https://github.com/openai/whisper)\n"
        "Test without GPU/models: AH_EMULATE_SOUND2TEXT=1"
    )


def _get_model(model_name: str):
    import whisper

    if model_name not in _WHISPER_MODELS:
        raise ValueError(
            f"$sound2text: unknown model {model_name!r}. "
            f"Examples: {', '.join(sorted(_WHISPER_MODELS)[:8])}…"
        )
    if model_name not in _MODEL_CACHE:
        print(f"$sound2text: loading whisper model {model_name!r}", flush=True)
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def _truthy_arg(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def _transcribe_file(
    model,
    path: Path,
    *,
    language: str | None,
    task: str,
    initial_prompt: str,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    kwargs: dict = {"verbose": False, "task": task, "word_timestamps": word_timestamps}
    if language:
        kwargs["language"] = language
    if initial_prompt.strip():
        kwargs["initial_prompt"] = initial_prompt.strip()
    return model.transcribe(str(path), **kwargs)


def _words_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append(
                {
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                }
            )
    return words


def _transcript_payload(
    result: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    text = (result.get("text") or "").strip()
    payload: dict[str, Any] = {
        "file": path.name,
        "text": text,
        "language": result.get("language"),
        "words": _words_from_result(result),
    }
    return payload


def _emulate(
    ctx: ExternalContext,
    out: ArrayBundle,
    *,
    paths: list[Path],
    model: str,
    language: str,
    task: str,
    as_json: bool,
    join: bool,
) -> ArrayBundle:
    payloads: list[dict[str, Any]] = []
    plain_blocks: list[str] = []
    for path in paths:
        if as_json:
            payloads.append(
                {
                    "file": path.name,
                    "text": f"[emulated $sound2text model={model} task={task}]",
                    "language": language or None,
                    "words": [
                        {"word": "[emulated]", "start": 0.0, "end": 0.5},
                        {"word": path.stem, "start": 0.5, "end": 1.0},
                    ],
                }
            )
        else:
            plain_blocks.append(
                f"[emulated $sound2text model={model} task={task}]\n"
                f"file: {path.name}\n"
                f"language: {language or 'auto'}\n"
            )

    if as_json:
        if join:
            if len(payloads) == 1:
                blob = json.dumps(payloads[0], indent=2) + "\n"
            else:
                blob = json.dumps({"files": payloads}, indent=2) + "\n"
            out.texts.append(ctx.new_link("texts", ".json", blob))
        else:
            for payload in payloads:
                blob = json.dumps(payload, indent=2) + "\n"
                out.texts.append(ctx.new_link("texts", ".json", blob))
    elif join:
        out.texts.append(ctx.new_link("texts", ".txt", "\n\n".join(plain_blocks) + "\n"))
    else:
        for block in plain_blocks:
            out.texts.append(ctx.new_link("texts", ".txt", block))
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    paths = _sound_paths(ctx, inp.bundle)
    if not paths:
        raise ValueError("$sound2text requires at least one audio file in sounds[]")

    model_name = inp.args.get("model", "base").strip() or "base"
    language = inp.args.get("language", "").strip() or None
    task = inp.args.get("task", "transcribe").strip().lower() or "transcribe"
    if task not in ("transcribe", "translate"):
        raise ValueError("$sound2text: task= must be transcribe or translate")
    join = _truthy_arg(inp.args.get("join", ""))
    as_json = _truthy_arg(inp.args.get("json", ""))

    guidance_parts = read_prompt_texts(ctx, inp)
    initial_prompt = "\n".join(guidance_parts) if guidance_parts else inp.prompt_text.strip()

    if os.environ.get("AH_EMULATE_SOUND2TEXT", "").lower() in ("1", "true", "yes"):
        return _emulate(
            ctx,
            out,
            paths=paths,
            model=model_name,
            language=language or "",
            task=task,
            as_json=as_json,
            join=join,
        )

    try:
        model = _get_model(model_name)
    except ImportError as exc:
        raise RuntimeError(_whisper_help()) from exc

    transcripts: list[str] = []
    json_payloads: list[dict[str, Any]] = []
    for path in paths:
        print(f"$sound2text: transcribing {path.name}", flush=True)
        result = _transcribe_file(
            model,
            path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=as_json,
        )
        if as_json:
            payload = _transcript_payload(result, path=path)
            if not payload["text"]:
                payload["text"] = f"[empty transcription for {path.name}]"
            json_payloads.append(payload)
            if not join:
                blob = json.dumps(payload, indent=2) + "\n"
                out.texts.append(ctx.new_link("texts", ".json", blob))
        else:
            text = (result.get("text") or "").strip()
            if not text:
                text = f"[empty transcription for {path.name}]"
            transcripts.append(text)
            if not join:
                out.texts.append(ctx.new_link("texts", ".txt", text + "\n"))

    if join:
        if as_json:
            if len(json_payloads) == 1:
                blob = json.dumps(json_payloads[0], indent=2) + "\n"
            else:
                blob = json.dumps({"files": json_payloads}, indent=2) + "\n"
            out.texts.append(ctx.new_link("texts", ".json", blob))
        else:
            joined = "\n\n".join(transcripts) + ("\n" if transcripts else "")
            out.texts.append(ctx.new_link("texts", ".txt", joined))

    return out
