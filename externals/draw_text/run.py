"""$draw_text — overlay multiline text on each input image."""

from __future__ import annotations

import os
import re
from pathlib import Path

from externals.api import (
    ExternalContext,
    ExternalInput,
    read_bundle_texts,
    read_prompt_texts,
)
from ahlib.ah_runtime import ArrayBundle

_DEFAULT_FONT = "ttf/MuseoSansCyrl-700.ttf"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_DRAW_TEXT", "").lower() in ("1", "true", "yes")


def _int_arg(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _resolve_font_path(font_arg: str) -> Path | None:
    for base in (_REPO_ROOT, Path.cwd()):
        path = (base / font_arg).resolve()
        if path.is_file():
            return path
    path = Path(font_arg)
    if path.is_file():
        return path.resolve()
    return None


def _prepare_text(text: str) -> str:
    if len(text) > 10:
        text = re.sub(r"\^", "\n", text)
    return text


def _texts_for_images(ctx: ExternalContext, inp: ExternalInput, image_count: int) -> list[str]:
    if inp.args.get("text", "").strip():
        return [_prepare_text(inp.args["text"].strip())] * image_count
    texts = read_prompt_texts(ctx, inp)
    if not texts:
        texts = read_bundle_texts(ctx, inp)
    if not texts:
        return [""] * image_count
    if len(texts) == 1:
        return [_prepare_text(texts[0])] * image_count
    if len(texts) >= image_count:
        return [_prepare_text(t) for t in texts[:image_count]]
    padded = [_prepare_text(t) for t in texts]
    padded.extend([_prepare_text(texts[-1])] * (image_count - len(texts)))
    return padded


def add_text(
    file_in: Path,
    file_out: Path,
    text: str,
    *,
    font_path: Path | None,
    font_size: int,
    text_left: int,
    text_top: int,
    spacing: int,
    stroke_width: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(file_in).convert("RGBA")
    draw = ImageDraw.Draw(img)

    if font_path is not None:
        font = ImageFont.truetype(str(font_path), font_size)
    else:
        font = ImageFont.load_default()

    text = _prepare_text(text)
    width, height = img.size

    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    max_x = max(0, width - text_width - 20)
    max_y = max(0, height - text_height - 20)
    left = min(text_left, max_x)
    top = min(text_top, max_y)

    print(text, flush=True)

    draw.multiline_text(
        (left, top),
        text,
        font=font,
        fill=(255, 255, 255),
        spacing=spacing,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
        align="left",
    )

    suffix = file_out.suffix.lower() or ".png"
    if suffix in (".jpg", ".jpeg"):
        img.convert("RGB").save(file_out, quality=95)
    else:
        img.save(file_out)


def _emulate(
    ctx: ExternalContext,
    inp: ExternalInput,
    out: ArrayBundle,
    texts: list[str],
) -> ArrayBundle:
    new_images: list[str] = []
    for img_link, text in zip(inp.bundle.images, texts):
        src = ctx.base_dir / img_link
        content = src.read_bytes() if src.exists() else b""
        marker = f"[emulated $draw_text]\n{text}\n".encode("utf-8")
        link = ctx.new_link("images", Path(img_link).suffix or ".png", content + marker)
        new_images.append(link)
    out.images = new_images
    return out


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    if not inp.bundle.images:
        return out

    texts = _texts_for_images(ctx, inp, len(inp.bundle.images))
    font_arg = inp.args.get("font", _DEFAULT_FONT).strip() or _DEFAULT_FONT
    font_size = _int_arg(inp.args, "size", 40)
    text_left = _int_arg(inp.args, "left", 30)
    text_top = _int_arg(inp.args, "top", 350)
    spacing = _int_arg(inp.args, "spacing", 6)
    stroke_width = _int_arg(inp.args, "stroke_width", 2)

    if _emulate_enabled():
        return _emulate(ctx, inp, out, texts)

    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for $draw_text. "
            "Install with: uv sync --extra draw_text "
            "or set AH_EMULATE_DRAW_TEXT=1"
        ) from exc

    font_path = _resolve_font_path(font_arg)
    new_images: list[str] = []
    for img_link, text in zip(inp.bundle.images, texts):
        src = ctx.base_dir / img_link
        suffix = src.suffix.lower() if src.suffix else ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            suffix = ".png"
        dest = ctx.op_dir / "images" / f"draw_{len(new_images)}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        add_text(
            src,
            dest,
            text,
            font_path=font_path,
            font_size=font_size,
            text_left=text_left,
            text_top=text_top,
            spacing=spacing,
            stroke_width=stroke_width,
        )
        rel = str(dest.relative_to(ctx.base_dir)).replace("\\", "/")
        new_images.append(rel)

    out.images = new_images
    return out
