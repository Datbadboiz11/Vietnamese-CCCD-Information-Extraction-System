"""Backward-compatible wrapper for OCR metrics."""

from __future__ import annotations

from src.evaluation.ocr_metrics import (
    TextNormalizationOptions,
    build_error_analysis_markdown,
    character_error_rate as cer,
    evaluate_predictions,
    levenshtein_distance,
    match_prediction_rows,
    normalize_metric_text,
    word_error_rate as wer,
)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


def evaluate_rows(
    rows: list[dict],
    prediction_key: str,
    ground_truth_key: str = "ground_truth_text",
) -> dict:
    summary, _ = evaluate_predictions(
        rows,
        prediction_keys={"default": prediction_key},
        ground_truth_key=ground_truth_key,
        normalization=TextNormalizationOptions(),
    )
    return summary["engines"]["default"]


def collect_error_samples(
    rows: list[dict],
    prediction_key: str,
    ground_truth_key: str = "ground_truth_text",
    backend_name: str | None = None,
) -> list[dict]:
    _, errors = evaluate_predictions(
        rows,
        prediction_keys={backend_name or "default": prediction_key},
        ground_truth_key=ground_truth_key,
        normalization=TextNormalizationOptions(),
    )
    return errors


__all__ = [
    "TextNormalizationOptions",
    "build_error_analysis_markdown",
    "cer",
    "collect_error_samples",
    "evaluate_predictions",
    "evaluate_rows",
    "exact_match",
    "levenshtein_distance",
    "match_prediction_rows",
    "normalize_metric_text",
    "wer",
]
