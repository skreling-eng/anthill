"""$ocr — optical character recognition via PaddleOCR."""

from __future__ import annotations

import os
from pathlib import Path

from externals.api import ExternalContext, ExternalInput
from externals.ocr.langs import resolve_lang
from ahlib.ah_runtime import ArrayBundle

_OCR_CACHE: dict[tuple[str, str, str, bool, bool], object] = {}
_PADDLE_INFERENCE_PATCHED = False


def _configure_paddle_env(*, use_gpu: bool) -> None:
    """Disable OneDNN / PIR paths that break PP-OCR CPU inference on Windows."""
    if use_gpu:
        return
    for key, val in (
        ("FLAGS_use_mkldnn", "0"),
        ("FLAGS_enable_mkldnn", "0"),
        ("FLAGS_use_onednn", "0"),
        ("FLAGS_enable_onednn", "0"),
        ("FLAGS_enable_pir_api", "0"),
        ("FLAGS_enable_pir_in_executor", "0"),
        ("PADDLE_PDX_DISABLE_MKLDNN", "1"),
    ):
        os.environ[key] = val
    try:
        import paddle

        paddle.set_flags(
            {
                "FLAGS_use_mkldnn": False,
                "FLAGS_enable_mkldnn": False,
            }
        )
    except Exception:
        pass


def _patch_paddle_inference() -> None:
    """Force legacy IR path — fused_conv2d + OneDNN fails with IR optim on CPU."""
    global _PADDLE_INFERENCE_PATCHED
    if _PADDLE_INFERENCE_PATCHED:
        return
    import paddle.inference as inference

    _orig_switch = inference.Config.switch_ir_optim

    def _switch_ir_optim_off(self, flag=True):
        return _orig_switch(self, False)

    inference.Config.switch_ir_optim = _switch_ir_optim_off
    inference.Config.enable_mkldnn = lambda self, *args, **kwargs: None
    _PADDLE_INFERENCE_PATCHED = True


def _emulate_enabled() -> bool:
    return os.environ.get("AH_EMULATE_OCR", "").lower() in ("1", "true", "yes")


def _truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _image_paths(ctx: ExternalContext, bundle: ArrayBundle) -> list[Path]:
    paths: list[Path] = []
    for link in bundle.images:
        path = Path(link)
        if not path.is_absolute():
            path = (ctx.base_dir / link).resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _ocr_help() -> str:
    return (
        "$ocr requires paddleocr and paddlepaddle.\n"
        "  uv sync --extra ocr\n"
        "  uv run python tools/download_models.py --upstream-fallback  (or ensure_model on first run)\n"
        "Test without models: AH_EMULATE_OCR=1"
    )


def _get_ocr(*, lang: str, use_angle_cls: bool, use_gpu: bool):
    _configure_paddle_env(use_gpu=use_gpu)
    _patch_paddle_inference()
    from externals.ocr.model_paths import ensure_model

    resolved = resolve_lang(lang)
    key = (resolved.rec_pack, resolved.paddle_lang, use_angle_cls, use_gpu)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]

    from paddleocr import PaddleOCR

    det_dir, rec_dir, cls_dir, resolved = ensure_model(lang=lang)
    kwargs: dict = {
        "use_angle_cls": use_angle_cls,
        "lang": resolved.paddle_lang,
        "det_model_dir": str(det_dir),
        "rec_model_dir": str(rec_dir),
        "cls_model_dir": str(cls_dir),
        "use_gpu": use_gpu,
        "enable_mkldnn": False,
        "show_log": False,
        "ocr_version": "PP-OCRv4",
    }
    print(f"$ocr: loading PaddleOCR lang={lang!r} gpu={use_gpu}", flush=True)
    engine = PaddleOCR(**kwargs)
    _OCR_CACHE[key] = engine
    return engine


def _format_ocr_result(result: object) -> str:
    lines: list[str] = []
    if not result:
        return "\n"

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        if not isinstance(page, list):
            continue
        for item in page:
            if not item or len(item) < 2:
                continue
            text_part = item[1]
            if isinstance(text_part, (list, tuple)) and text_part:
                text = str(text_part[0]).strip()
            else:
                text = str(text_part).strip()
            if text:
                lines.append(text)
    if not lines:
        return "\n"
    return "\n".join(lines) + "\n"


def _run_ocr(engine, image_path: Path) -> str:
    result = engine.ocr(str(image_path), cls=True)
    return _format_ocr_result(result)


def run(ctx: ExternalContext, inp: ExternalInput) -> ArrayBundle:
    out = inp.bundle.copy()
    out.texts.clear()

    images = _image_paths(ctx, inp.bundle)
    if not images:
        link = ctx.new_link("texts", ".txt", "[ $ocr: no images[] input ]\n")
        out.texts.append(link)
        return out

    lang = inp.args.get("lang", "en").strip().lower() or "en"
    use_angle_cls = not _truthy(inp.args.get("no_cls", ""))
    use_gpu = _truthy(inp.args.get("gpu", os.environ.get("AH_OCR_GPU", "")))

    if _emulate_enabled():
        for image_path in images:
            text = f"[emulated $ocr lang={lang}] {image_path.name}\n"
            out.texts.append(ctx.new_link("texts", ".txt", text))
        return out

    try:
        engine = _get_ocr(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu)
    except ImportError as exc:
        raise RuntimeError(_ocr_help()) from exc

    for image_path in images:
        text = _run_ocr(engine, image_path)
        out.texts.append(ctx.new_link("texts", ".txt", text))

    return out
