"""OCR adapters, crop utilities, ensemble logic, and OCR metrics."""

from .cropping import PreparedCard, prepare_card_for_ocr, project_bbox
from .ensemble import ensemble_predictions, ensemble_recognize, select_best_ocr_result
from .paddleocr_adapter import PaddleOCRRecognizer
from .types import EnsembleResult, OCRResult, OCRSegment
from .vietocr_adapter import VietOCRRecognizer

__all__ = [
    "EnsembleResult",
    "OCRResult",
    "OCRSegment",
    "PaddleOCRRecognizer",
    "PreparedCard",
    "VietOCRRecognizer",
    "ensemble_predictions",
    "ensemble_recognize",
    "prepare_card_for_ocr",
    "project_bbox",
    "select_best_ocr_result",
]
