from __future__ import annotations

from src.ocr.types import EnsembleResult, OCRResult
from src.ocr.utils import is_valid_date, is_valid_id_number, normalize_text_for_field


def _candidate_confidence(result: OCRResult) -> float:
    return float(result.confidence if result.confidence is not None else 0.0)


def _resolve_same_text(field_name: str, vietocr: OCRResult, paddleocr: OCRResult) -> EnsembleResult:
    chosen = vietocr if _candidate_confidence(vietocr) >= _candidate_confidence(paddleocr) else paddleocr
    return EnsembleResult(
        text=chosen.text,
        confidence=max(_candidate_confidence(vietocr), _candidate_confidence(paddleocr)),
        source="agree",
        needs_review=False,
        candidates={"vietocr": vietocr, "paddleocr": paddleocr},
        normalized_text=normalize_text_for_field(chosen.text, field_name),
    )


def _pick_by_confidence(vietocr: OCRResult, paddleocr: OCRResult) -> tuple[OCRResult, str]:
    if _candidate_confidence(vietocr) >= _candidate_confidence(paddleocr):
        return vietocr, "confidence_tiebreak"
    return paddleocr, "confidence_tiebreak"


def _pick_with_validity(field_name: str, vietocr: OCRResult, paddleocr: OCRResult) -> tuple[OCRResult, str]:
    if field_name == "id_number":
        viet_valid = is_valid_id_number(vietocr.normalized_text or vietocr.text)
        pad_valid = is_valid_id_number(paddleocr.normalized_text or paddleocr.text)
    elif field_name == "date_of_birth":
        viet_valid = is_valid_date(vietocr.normalized_text or vietocr.text)
        pad_valid = is_valid_date(paddleocr.normalized_text or paddleocr.text)
    else:
        return _pick_by_confidence(vietocr, paddleocr)

    if viet_valid and not pad_valid:
        return vietocr, "vietocr"
    if pad_valid and not viet_valid:
        return paddleocr, "paddleocr"
    return _pick_by_confidence(vietocr, paddleocr)


def ensemble_predictions(
    field_name: str,
    vietocr: OCRResult,
    paddleocr: OCRResult,
    confidence_margin: float = 0.1,
) -> EnsembleResult:
    viet_normalized = normalize_text_for_field(vietocr.text, field_name)
    pad_normalized = normalize_text_for_field(paddleocr.text, field_name)
    vietocr.normalized_text = viet_normalized
    paddleocr.normalized_text = pad_normalized

    if viet_normalized and viet_normalized == pad_normalized:
        return _resolve_same_text(field_name, vietocr, paddleocr)

    if field_name in {"id_number", "date_of_birth"}:
        chosen, source = _pick_with_validity(field_name, vietocr, paddleocr)
    else:
        chosen, source = _pick_by_confidence(vietocr, paddleocr)

    margin = abs(_candidate_confidence(vietocr) - _candidate_confidence(paddleocr))
    needs_review = (
        _candidate_confidence(chosen) < 0.5
        or (viet_normalized != pad_normalized and margin < confidence_margin)
        or normalize_text_for_field(chosen.text, field_name) == ""
    )

    return EnsembleResult(
        text=chosen.text,
        confidence=_candidate_confidence(chosen),
        source=source,
        needs_review=needs_review,
        candidates={"vietocr": vietocr, "paddleocr": paddleocr},
        normalized_text=normalize_text_for_field(chosen.text, field_name),
    )
