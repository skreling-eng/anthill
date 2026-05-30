"""PaddleOCR inference models under models/ocr/."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from externals.image.model_paths import models_roots
from externals.ocr.langs import (
    CLS_URL,
    DET_PACK_URLS,
    OcrLang,
    REC_PACK_URLS,
    resolve_lang,
)

OCR_VERSION = "PP-OCRv4"

_INFERENCE_MARKERS = ("inference.pdmodel", "inference.json")


def pack_root(rec_pack: str) -> Path:
    for root in models_roots():
        candidate = root / "ocr" / OCR_VERSION / rec_pack
        if candidate.is_dir():
            return candidate
    return models_roots()[0] / "ocr" / OCR_VERSION / rec_pack


def _component_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _INFERENCE_MARKERS)


def model_paths_for_pack(rec_pack: str) -> tuple[Path, Path, Path]:
    root = pack_root(rec_pack)
    return root / "det", root / "rec", root / "cls"


def model_paths_for(lang: OcrLang) -> tuple[Path, Path, Path]:
    return model_paths_for_pack(lang.rec_pack)


def model_ready_pack(rec_pack: str) -> bool:
    det, rec, cls = model_paths_for_pack(rec_pack)
    return _component_ready(det) and _component_ready(rec) and _component_ready(cls)


def model_ready(lang: str = "en") -> bool:
    return model_ready_pack(resolve_lang(lang).rec_pack)


def _download_tar(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar"
        print(f"$ocr: downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path, "r:*") as archive:
            archive.extractall(dest_dir)
        subdirs = [p for p in dest_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            nested = subdirs[0]
            for item in nested.iterdir():
                target = dest_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            nested.rmdir()


def ensure_pack(rec_pack: str, det_pack: str, *, force: bool = False) -> tuple[Path, Path, Path]:
    det_dir, rec_dir, cls_dir = model_paths_for_pack(rec_pack)
    if model_ready_pack(rec_pack) and not force:
        return det_dir, rec_dir, cls_dir

    rec_url = REC_PACK_URLS.get(rec_pack)
    det_url = DET_PACK_URLS.get(det_pack)
    if not rec_url or not det_url:
        raise ValueError(f"$ocr: missing model URLs for rec={rec_pack!r} det={det_pack!r}")

    _download_tar(det_url, det_dir)
    _download_tar(rec_url, rec_dir)
    _download_tar(CLS_URL, cls_dir)

    if not model_ready_pack(rec_pack):
        raise FileNotFoundError(
            f"PaddleOCR models missing under {pack_root(rec_pack)} after download"
        )
    return det_dir, rec_dir, cls_dir


def ensure_model(*, lang: str = "en", force: bool = False) -> tuple[Path, Path, Path, OcrLang]:
    """Download det/rec/cls for the language's shared recognition pack."""
    resolved = resolve_lang(lang)
    det_dir, rec_dir, cls_dir = ensure_pack(
        resolved.rec_pack, resolved.det_pack, force=force
    )
    return det_dir, rec_dir, cls_dir, resolved


def ensure_all_packs(*, force: bool = False) -> list[str]:
    """Download all recognition packs (8 packs covering top-20 lang= codes)."""
    from externals.ocr.langs import REC_PACK_DET

    done: list[str] = []
    for rec_pack in sorted(REC_PACK_URLS):
        det_pack = REC_PACK_DET[rec_pack]
        ensure_pack(rec_pack, det_pack, force=force)
        done.append(rec_pack)
    return done


def ensure_all_core_packs(*, force: bool = False) -> None:
    """Alias for ensure_all_packs."""
    ensure_all_packs(force=force)
