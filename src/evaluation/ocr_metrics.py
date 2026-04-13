from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import unicodedata
from typing import Any


@dataclass(frozen=True)
class TextNormalizationOptions:
    strip_extra_spaces: bool = True
    unicode_form: str = "NFC"
    case_sensitive: bool = True


@dataclass
class MetricAccumulator:
    count: int = 0
    cer_total: float = 0.0
    wer_total: float = 0.0
    exact_total: int = 0

    def update(self, reference: str, hypothesis: str) -> None:
        self.count += 1
        self.cer_total += character_error_rate(reference, hypothesis)
        self.wer_total += word_error_rate(reference, hypothesis)
        self.exact_total += int(reference == hypothesis)

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0, "cer": 0.0, "wer": 0.0, "exact_match_rate": 0.0}
        return {
            "count": self.count,
            "cer": self.cer_total / self.count,
            "wer": self.wer_total / self.count,
            "exact_match_rate": self.exact_total / self.count,
        }


def normalize_metric_text(text: str, options: TextNormalizationOptions) -> str:
    value = text or ""
    if options.unicode_form:
        value = unicodedata.normalize(options.unicode_form, value)
    if options.strip_extra_spaces:
        value = " ".join(value.split())
    if not options.case_sensitive:
        value = value.lower()
    return value


def levenshtein_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            insertion = previous[hyp_index] + 1
            deletion = current[hyp_index - 1] + 1
            substitution = previous[hyp_index - 1] + (ref_token != hyp_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / max(1, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / max(1, len(ref_words))


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    crop_path = str(row.get("crop_path") or row.get("path") or row.get("id") or "").strip().replace("\\", "/")
    field_name = str(row.get("field_name") or row.get("class") or "").strip()
    return crop_path, field_name


def match_prediction_rows(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_index = {_row_key(row): row for row in pred_rows}
    matched_rows: list[dict[str, Any]] = []
    for gt_row in gt_rows:
        key = _row_key(gt_row)
        pred_row = pred_index.get(key)
        if pred_row is None:
            continue
        merged = dict(pred_row)
        merged.setdefault("crop_path", gt_row.get("crop_path") or gt_row.get("path"))
        merged.setdefault("field_name", gt_row.get("field_name") or gt_row.get("class"))
        merged["ground_truth_text"] = gt_row.get("ground_truth_text") or gt_row.get("text") or gt_row.get("transcript") or ""
        matched_rows.append(merged)
    return matched_rows


def evaluate_predictions(
    rows: list[dict[str, Any]],
    prediction_keys: dict[str, str],
    *,
    ground_truth_key: str = "ground_truth_text",
    normalization: TextNormalizationOptions | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    options = normalization or TextNormalizationOptions()
    summary: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []

    for engine_name, prediction_key in prediction_keys.items():
        overall = MetricAccumulator()
        per_field: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

        for row in rows:
            field_name = str(row.get("field_name") or row.get("class") or "")
            reference_raw = str(row.get(ground_truth_key, "") or "")
            if not reference_raw.strip():
                continue
            prediction_raw = str(row.get(prediction_key, "") or "")

            reference = normalize_metric_text(reference_raw, options)
            prediction = normalize_metric_text(prediction_raw, options)

            overall.update(reference, prediction)
            per_field[field_name].update(reference, prediction)

            if reference != prediction:
                error_rows.append(
                    {
                        "engine": engine_name,
                        "field_name": field_name,
                        "crop_path": row.get("crop_path"),
                        "ground_truth_text": reference_raw,
                        "prediction_text": prediction_raw,
                        "ground_truth_normalized": reference,
                        "prediction_normalized": prediction,
                        "cer": character_error_rate(reference, prediction),
                        "wer": word_error_rate(reference, prediction),
                    }
                )

        summary[engine_name] = {
            "overall": overall.as_dict(),
            "per_field": {field: accumulator.as_dict() for field, accumulator in sorted(per_field.items())},
        }

    payload = {
        "normalization": asdict(options),
        "engines": summary,
    }
    error_rows.sort(key=lambda row: (row["engine"], -row["cer"], -row["wer"], row["field_name"]))
    return payload, error_rows


def build_error_analysis_markdown(
    error_rows: list[dict[str, Any]],
    *,
    max_samples_per_engine: int = 20,
) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in error_rows:
        grouped[str(row["engine"])].append(row)

    lines: list[str] = ["# OCR Error Analysis", ""]
    if not grouped:
        lines.extend(["No OCR mismatches were found.", ""])
        return "\n".join(lines)

    for engine_name in sorted(grouped):
        lines.append(f"## {engine_name}")
        lines.append("")
        for sample in grouped[engine_name][:max_samples_per_engine]:
            lines.append(f"- Field: `{sample['field_name']}` | CER={sample['cer']:.4f} | WER={sample['wer']:.4f}")
            lines.append(f"  Crop: `{sample.get('crop_path') or ''}`")
            lines.append(f"  GT: `{sample['ground_truth_text']}`")
            lines.append(f"  Pred: `{sample['prediction_text']}`")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "TextNormalizationOptions",
    "build_error_analysis_markdown",
    "character_error_rate",
    "evaluate_predictions",
    "levenshtein_distance",
    "match_prediction_rows",
    "normalize_metric_text",
    "word_error_rate",
]
