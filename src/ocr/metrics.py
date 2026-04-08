from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.ocr.utils import normalize_text_for_field


def levenshtein_distance(a: list[str] | str, b: list[str] | str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for index_a, item_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, item_b in enumerate(b, start=1):
            insertions = previous[index_b] + 1
            deletions = current[index_b - 1] + 1
            substitutions = previous[index_b - 1] + (item_a != item_b)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / max(1, len(reference))


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / max(1, len(ref_words))


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


@dataclass
class MetricAccumulator:
    count: int = 0
    cer_total: float = 0.0
    wer_total: float = 0.0
    exact_total: int = 0

    def update(self, reference: str, hypothesis: str) -> None:
        self.count += 1
        self.cer_total += cer(reference, hypothesis)
        self.wer_total += wer(reference, hypothesis)
        self.exact_total += int(exact_match(reference, hypothesis))

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0, "cer": 0.0, "wer": 0.0, "exact_match": 0.0}
        return {
            "count": self.count,
            "cer": self.cer_total / self.count,
            "wer": self.wer_total / self.count,
            "exact_match": self.exact_total / self.count,
        }


def evaluate_rows(
    rows: list[dict[str, Any]],
    prediction_key: str,
    ground_truth_key: str = "ground_truth_text",
) -> dict[str, Any]:
    overall = MetricAccumulator()
    per_field: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    for row in rows:
        field_name = row.get("field_name") or row.get("class")
        reference_raw = str(row.get(ground_truth_key, "") or "").strip()
        if not reference_raw:
            continue

        hypothesis_raw = str(row.get(prediction_key, "") or "").strip()
        reference = normalize_text_for_field(reference_raw, field_name)
        hypothesis = normalize_text_for_field(hypothesis_raw, field_name)

        overall.update(reference, hypothesis)
        per_field[str(field_name)].update(reference, hypothesis)

    return {
        "overall": overall.as_dict(),
        "per_field": {field_name: metrics.as_dict() for field_name, metrics in sorted(per_field.items())},
    }


def collect_error_samples(
    rows: list[dict[str, Any]],
    prediction_key: str,
    ground_truth_key: str = "ground_truth_text",
    backend_name: str | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        field_name = row.get("field_name") or row.get("class")
        reference_raw = str(row.get(ground_truth_key, "") or "").strip()
        if not reference_raw:
            continue

        prediction_raw = str(row.get(prediction_key, "") or "").strip()
        reference = normalize_text_for_field(reference_raw, field_name)
        prediction = normalize_text_for_field(prediction_raw, field_name)
        if reference == prediction:
            continue

        samples.append(
            {
                "backend": backend_name or prediction_key,
                "field_name": field_name,
                "crop_path": row.get("crop_path"),
                "prediction": prediction_raw,
                "prediction_normalized": prediction,
                "ground_truth": reference_raw,
                "ground_truth_normalized": reference,
                "cer": cer(reference, prediction),
                "wer": wer(reference, prediction),
            }
        )
    return samples
