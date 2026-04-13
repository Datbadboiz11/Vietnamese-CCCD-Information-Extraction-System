from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.ocr.types import EnsembleResult, OCRResult
from src.ocr.utils import (
    canonicalize_field_name,
    empty_ocr_result,
    is_valid_date,
    is_valid_id_number,
    looks_suspicious_for_field,
    normalize_text_for_field,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_VIETOCR = None
_DEFAULT_PADDLEOCR = None

NUMERIC_DATE_FIELDS = {"id", "id_number", "birth", "date_of_birth"}
LONG_TEXT_FIELDS = {"place_of_origin", "place_of_residence"}


def _score(result: OCRResult) -> float:
    return float(result.confidence or 0.0)


def _normalized_text(result: OCRResult, field_name: str | None) -> str:
    if result.normalized_text is None:
        result.normalized_text = normalize_text_for_field(result.text, field_name)
    return result.normalized_text


def _passes_pattern(field_name: str | None, result: OCRResult) -> bool:
    canonical_field = canonicalize_field_name(field_name)
    normalized = _normalized_text(result, canonical_field)
    if canonical_field == "id_number":
        return is_valid_id_number(normalized)
    if canonical_field == "date_of_birth":
        return is_valid_date(normalized)
    return False


def _is_suspicious(field_name: str | None, result: OCRResult) -> bool:
    return looks_suspicious_for_field(result.text, field_name)


def _pick_for_long_text(
    field_name: str | None,
    vietocr_result: OCRResult,
    paddleocr_result: OCRResult,
    confidence_margin: float,
) -> tuple[OCRResult, str, str]:
    viet_score = _score(vietocr_result)
    paddle_score = _score(paddleocr_result)
    viet_text = _normalized_text(vietocr_result, field_name)
    paddle_text = _normalized_text(paddleocr_result, field_name)
    viet_bad = _is_suspicious(field_name, vietocr_result)
    paddle_bad = _is_suspicious(field_name, paddleocr_result)

    if paddle_text and not viet_text:
        return paddleocr_result, "paddleocr", "long_text_nonempty"
    if viet_text and not paddle_text:
        return vietocr_result, "vietocr", "long_text_nonempty"
    if viet_bad and not paddle_bad:
        return paddleocr_result, "paddleocr", "viet_suspicious_long_text"
    if paddle_bad and not viet_bad:
        return vietocr_result, "vietocr", "paddle_suspicious_long_text"
    if paddle_score >= viet_score - 0.12:
        return paddleocr_result, "paddleocr", "paddle_long_text_default"
    if viet_score > paddle_score + max(confidence_margin + 0.20, 0.35):
        return vietocr_result, "vietocr", "viet_long_text_confidence"
    return paddleocr_result, "paddleocr", "paddle_long_text_default"


def _pick_for_name(
    field_name: str | None,
    vietocr_result: OCRResult,
    paddleocr_result: OCRResult,
    confidence_margin: float,
) -> tuple[OCRResult, str, str]:
    viet_score = _score(vietocr_result)
    paddle_score = _score(paddleocr_result)
    viet_text = _normalized_text(vietocr_result, field_name)
    paddle_text = _normalized_text(paddleocr_result, field_name)
    viet_bad = _is_suspicious(field_name, vietocr_result)
    paddle_bad = _is_suspicious(field_name, paddleocr_result)

    if viet_text and not paddle_text:
        return vietocr_result, "vietocr", "name_nonempty"
    if paddle_text and not viet_text:
        return paddleocr_result, "paddleocr", "name_nonempty"
    if paddle_bad and not viet_bad:
        return vietocr_result, "vietocr", "paddle_suspicious_name"
    if viet_bad and not paddle_bad:
        return paddleocr_result, "paddleocr", "viet_suspicious_name"
    if viet_score >= paddle_score - 0.08:
        return vietocr_result, "vietocr", "viet_name_bias"
    if len(viet_text) >= len(paddle_text) and viet_score >= paddle_score - 0.12:
        return vietocr_result, "vietocr", "viet_name_length_bias"
    if paddle_score > viet_score + confidence_margin:
        return paddleocr_result, "paddleocr", "paddle_name_confidence"
    return vietocr_result, "vietocr", "viet_name_bias"


def select_best_ocr_result(
    field_name: str | None,
    vietocr_result: OCRResult,
    paddleocr_result: OCRResult,
    confidence_margin: float = 0.08,
) -> EnsembleResult:
    """Pick the best OCR output from VietOCR and PaddleOCR."""

    canonical_field = canonicalize_field_name(field_name)
    viet_text = _normalized_text(vietocr_result, canonical_field)
    paddle_text = _normalized_text(paddleocr_result, canonical_field)
    viet_score = _score(vietocr_result)
    paddle_score = _score(paddleocr_result)

    if viet_text and viet_text == paddle_text:
        chosen = vietocr_result if viet_score >= paddle_score else paddleocr_result
        return EnsembleResult(
            text=chosen.text,
            score=max(viet_score, paddle_score),
            engine="agree",
            needs_review=False,
            candidates={"vietocr": vietocr_result, "paddleocr": paddleocr_result},
            normalized_text=viet_text,
            raw={"reason": "same_text"},
        )

    chosen = vietocr_result
    chosen_engine = "vietocr"
    reason = "higher_confidence"

    if canonical_field in {"id_number", "date_of_birth"}:
        viet_ok = _passes_pattern(canonical_field, vietocr_result)
        paddle_ok = _passes_pattern(canonical_field, paddleocr_result)
        if viet_ok and not paddle_ok:
            chosen = vietocr_result
            chosen_engine = "vietocr"
            reason = "pattern_match"
        elif paddle_ok and not viet_ok:
            chosen = paddleocr_result
            chosen_engine = "paddleocr"
            reason = "pattern_match"
        elif paddle_score > viet_score:
            chosen = paddleocr_result
            chosen_engine = "paddleocr"
            reason = "higher_confidence"
    elif canonical_field in LONG_TEXT_FIELDS:
        chosen, chosen_engine, reason = _pick_for_long_text(
            canonical_field,
            vietocr_result,
            paddleocr_result,
            confidence_margin,
        )
    elif canonical_field == "full_name":
        chosen, chosen_engine, reason = _pick_for_name(
            canonical_field,
            vietocr_result,
            paddleocr_result,
            confidence_margin,
        )
    elif paddle_score > viet_score:
        chosen = paddleocr_result
        chosen_engine = "paddleocr"
        reason = "higher_confidence"

    margin = abs(viet_score - paddle_score)
    chosen_normalized = _normalized_text(chosen, canonical_field)
    needs_review = chosen.confidence < 0.5 or chosen_normalized == ""
    if _is_suspicious(canonical_field, chosen):
        needs_review = True
    if viet_text != paddle_text and margin < confidence_margin:
        needs_review = True

    return EnsembleResult(
        text=chosen.text,
        score=_score(chosen),
        engine=chosen_engine,
        needs_review=needs_review,
        candidates={"vietocr": vietocr_result, "paddleocr": paddleocr_result},
        normalized_text=chosen_normalized,
        raw={"reason": reason, "confidence_margin": margin},
    )


def ensemble_predictions(
    field_name: str | None,
    vietocr: OCRResult,
    paddleocr: OCRResult,
    confidence_margin: float = 0.08,
) -> EnsembleResult:
    return select_best_ocr_result(
        field_name=field_name,
        vietocr_result=vietocr,
        paddleocr_result=paddleocr,
        confidence_margin=confidence_margin,
    )


def _get_default_recognizers() -> tuple[Any, Any]:
    global _DEFAULT_PADDLEOCR, _DEFAULT_VIETOCR
    if _DEFAULT_VIETOCR is None:
        from src.ocr.vietocr_adapter import VietOCRRecognizer

        _DEFAULT_VIETOCR = VietOCRRecognizer()
    if _DEFAULT_PADDLEOCR is None:
        from src.ocr.paddleocr_adapter import PaddleOCRRecognizer

        _DEFAULT_PADDLEOCR = PaddleOCRRecognizer()
    return _DEFAULT_VIETOCR, _DEFAULT_PADDLEOCR


def ensemble_recognize(
    field_name: str | None,
    image: np.ndarray,
    *,
    vietocr_recognizer: Any | None = None,
    paddleocr_recognizer: Any | None = None,
    confidence_margin: float = 0.08,
) -> EnsembleResult:
    """Run both OCR engines on a crop and return the ensemble decision."""

    if image is None or image.size == 0:
        empty = empty_ocr_result("ensemble")
        return EnsembleResult(
            text=empty.text,
            score=empty.score,
            engine=empty.engine,
            needs_review=True,
            candidates={"vietocr": empty_ocr_result("vietocr"), "paddleocr": empty_ocr_result("paddleocr")},
            normalized_text="",
            raw={"reason": "empty_image"},
        )

    try:
        if vietocr_recognizer is None or paddleocr_recognizer is None:
            default_vietocr, default_paddleocr = _get_default_recognizers()
            vietocr_recognizer = vietocr_recognizer or default_vietocr
            paddleocr_recognizer = paddleocr_recognizer or default_paddleocr

        viet_result = vietocr_recognizer.recognize(image, field_name=field_name)
        paddle_result = paddleocr_recognizer.recognize(image, field_name=field_name)
        return select_best_ocr_result(field_name, viet_result, paddle_result, confidence_margin=confidence_margin)
    except Exception as exc:
        LOGGER.warning("Ensemble recognition failed: %s", exc)
        failed = empty_ocr_result("ensemble")
        return EnsembleResult(
            text=failed.text,
            score=0.0,
            engine="ensemble",
            needs_review=True,
            candidates={"vietocr": empty_ocr_result("vietocr"), "paddleocr": empty_ocr_result("paddleocr")},
            normalized_text="",
            raw={"reason": "exception", "error": str(exc)},
        )


__all__ = ["ensemble_predictions", "ensemble_recognize", "select_best_ocr_result"]
