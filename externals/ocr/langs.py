"""Language codes and PaddleOCR model pack mapping."""

from __future__ import annotations

from dataclasses import dataclass

# Top ~20 languages by speakers; maps user lang= code -> PaddleOCR rec pack + det pack.
# Bengali (bn) and Thai (th) are not supported by PP-OCRv4 rec packs.


@dataclass(frozen=True)
class OcrLang:
    code: str
    label: str
    rec_pack: str
    det_pack: str
    paddle_lang: str


def _latin(code: str, label: str) -> OcrLang:
    return OcrLang(code, label, "latin", "en", code)


def _arabic(code: str, label: str) -> OcrLang:
    return OcrLang(code, label, "arabic", "ml", code)


def _cyrillic(code: str, label: str) -> OcrLang:
    return OcrLang(code, label, "cyrillic", "ml", code)


def _devanagari(code: str, label: str) -> OcrLang:
    return OcrLang(code, label, "devanagari", "ml", code)


OCR_LANGUAGES: tuple[OcrLang, ...] = (
    OcrLang("en", "English", "en", "en", "en"),
    OcrLang("zh", "Chinese (Simplified)", "ch", "ch", "ch"),
    OcrLang("ch", "Chinese (Simplified)", "ch", "ch", "ch"),
    _devanagari("hi", "Hindi"),
    _latin("es", "Spanish"),
    _latin("fr", "French"),
    _arabic("ar", "Arabic"),
    _latin("pt", "Portuguese"),
    _cyrillic("ru", "Russian"),
    OcrLang("ja", "Japanese", "japan", "ml", "japan"),
    _latin("de", "German"),
    _latin("id", "Indonesian"),
    _latin("vi", "Vietnamese"),
    _latin("tr", "Turkish"),
    _latin("it", "Italian"),
    OcrLang("ko", "Korean", "korean", "ml", "korean"),
    _cyrillic("uk", "Ukrainian"),
    _latin("pl", "Polish"),
    _latin("nl", "Dutch"),
    _arabic("ur", "Urdu"),
    _arabic("fa", "Persian (Farsi)"),
)

OCR_LANG_BY_CODE: dict[str, OcrLang] = {}
for entry in OCR_LANGUAGES:
    OCR_LANG_BY_CODE.setdefault(entry.code.lower(), entry)

REC_PACK_URLS: dict[str, str] = {
    "en": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
    "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
    "latin": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
    "arabic": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar",
    "cyrillic": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
    "devanagari": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar",
    "korean": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
    "japan": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
}

# Detection model shared per recognition pack.
REC_PACK_DET: dict[str, str] = {
    "en": "en",
    "ch": "ch",
    "latin": "en",
    "arabic": "ml",
    "cyrillic": "ml",
    "devanagari": "ml",
    "korean": "ml",
    "japan": "ml",
}

DET_PACK_URLS: dict[str, str] = {
    "en": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
    "ml": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar",
}

CLS_URL = (
    "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
)


def resolve_lang(code: str) -> OcrLang:
    key = code.strip().lower() or "en"
    if key not in OCR_LANG_BY_CODE:
        supported = ", ".join(sorted({e.code for e in OCR_LANGUAGES}))
        raise ValueError(
            f"$ocr: unsupported lang {code!r}. Supported: {supported}"
        )
    return OCR_LANG_BY_CODE[key]


def supported_lang_codes() -> list[str]:
    return sorted(OCR_LANG_BY_CODE.keys())
