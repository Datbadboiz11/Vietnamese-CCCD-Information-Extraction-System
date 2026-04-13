"""OCR evaluation utilities."""

from .ocr_metrics import (
    TextNormalizationOptions,
    build_error_analysis_markdown,
    evaluate_predictions,
    match_prediction_rows,
    normalize_metric_text,
)

__all__ = [
    "TextNormalizationOptions",
    "build_error_analysis_markdown",
    "evaluate_predictions",
    "match_prediction_rows",
    "normalize_metric_text",
]
