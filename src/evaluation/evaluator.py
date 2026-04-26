"""
OCREvaluator — wrapper cấp cao cho evaluation framework.

Hỗ trợ:
- Field-level: CER, WER, Exact Match Rate
- End-to-end: đánh giá từ PipelineResult so với ground truth
- Tổng hợp báo cáo markdown

Usage::
    evaluator = OCREvaluator()

    # Eval từng sample OCR
    evaluator.add(field_name="id_number", reference="079...", hypothesis="079...")

    # Tổng kết
    report = evaluator.summary()
    print(report["overall"]["exact_match_rate"])
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ocr_metrics import (
    MetricAccumulator,
    TextNormalizationOptions,
    build_error_analysis_markdown,
    character_error_rate,
    evaluate_predictions,
    match_prediction_rows,
    normalize_metric_text,
    word_error_rate,
)


class OCREvaluator:
    """
    Đánh giá OCR kết hợp field-level và end-to-end.

    Sử dụng::
        evaluator = OCREvaluator()
        evaluator.add("id_number", reference="079201001234", hypothesis="079201001234")
        evaluator.add("full_name", reference="NGUYEN VAN A", hypothesis="NGUYEN VAN A")
        report = evaluator.summary()
    """

    def __init__(
        self,
        normalization: TextNormalizationOptions | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self._options = normalization or TextNormalizationOptions(
            strip_extra_spaces=True,
            unicode_form="NFC",
            case_sensitive=case_sensitive,
        )
        self._overall = MetricAccumulator()
        self._per_field: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
        self._error_rows: list[dict[str, Any]] = []

    def add(
        self,
        field_name: str,
        reference: str,
        hypothesis: str,
        sample_id: str | None = None,
    ) -> None:
        """Thêm 1 cặp (reference, hypothesis) vào accumulator."""
        ref = normalize_metric_text(reference, self._options)
        hyp = normalize_metric_text(hypothesis, self._options)

        self._overall.update(ref, hyp)
        self._per_field[field_name].update(ref, hyp)

        if ref != hyp:
            self._error_rows.append(
                {
                    "engine": "pipeline",
                    "field_name": field_name,
                    "crop_path": sample_id,
                    "ground_truth_text": reference,
                    "prediction_text": hypothesis,
                    "ground_truth_normalized": ref,
                    "prediction_normalized": hyp,
                    "cer": character_error_rate(ref, hyp),
                    "wer": word_error_rate(ref, hyp),
                }
            )

    def add_batch(self, rows: list[dict[str, Any]]) -> None:
        """
        Thêm batch rows, mỗi row cần:
            - field_name / class
            - ground_truth_text / text / transcript
            - predicted_text / best_text
        """
        for row in rows:
            field_name = str(row.get("field_name") or row.get("class") or "")
            reference = str(
                row.get("ground_truth_text")
                or row.get("text")
                or row.get("transcript")
                or ""
            )
            hypothesis = str(
                row.get("predicted_text")
                or row.get("best_text")
                or row.get("prediction")
                or ""
            )
            sample_id = str(row.get("crop_path") or row.get("id") or "")
            if reference:
                self.add(field_name, reference, hypothesis, sample_id)

    def add_from_pipeline_result(
        self,
        pipeline_result: Any,
        ground_truth: dict[str, str],
    ) -> None:
        """
        So sánh PipelineResult với ground truth dict.

        ground_truth ví dụ::
            {
                "id_number": "079...",
                "full_name": "NGUYEN VAN A",
                ...
            }
        """
        if pipeline_result.parsed_info is None:
            return

        info = pipeline_result.parsed_info
        field_map = {
            "id_number": info.id_number,
            "full_name": info.full_name,
            "date_of_birth": info.date_of_birth,
            "place_of_origin": info.place_of_origin,
            "place_of_residence": info.place_of_residence,
        }
        for field_name, predicted in field_map.items():
            reference = ground_truth.get(field_name)
            if reference is None:
                continue
            self.add(field_name, reference, predicted or "")

    def summary(self) -> dict[str, Any]:
        """Trả về dict summary: overall + per_field metrics."""
        return {
            "overall": self._overall.as_dict(),
            "per_field": {
                fname: acc.as_dict()
                for fname, acc in sorted(self._per_field.items())
            },
        }

    def error_rows(self) -> list[dict[str, Any]]:
        """Trả về danh sách sample bị dự đoán sai."""
        return list(self._error_rows)

    def error_analysis_markdown(self, max_samples: int = 20) -> str:
        """Tạo báo cáo markdown cho các lỗi."""
        return build_error_analysis_markdown(self._error_rows, max_samples_per_engine=max_samples)

    def reset(self) -> None:
        """Reset toàn bộ accumulator."""
        self._overall = MetricAccumulator()
        self._per_field = defaultdict(MetricAccumulator)
        self._error_rows = []

    # ── static helpers ────────────────────────────────────────────────────

    @staticmethod
    def evaluate_jsonl(
        gt_path: str | Path,
        pred_path: str | Path,
        *,
        prediction_key: str = "best_text",
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate 2 file JSONL (ground truth vs predictions).

        Args:
            gt_path: file reviewed.jsonl (có ground_truth_text)
            pred_path: file pseudo_labels.jsonl (có best_text hoặc predicted_text)
            prediction_key: key chứa text dự đoán trong pred file
            output_dir: nếu cung cấp, lưu báo cáo vào thư mục này

        Returns:
            dict summary metrics
        """
        import jsonlines  # type: ignore

        def _load_jsonl(path: Path) -> list[dict[str, Any]]:
            rows = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return rows

        gt_rows = _load_jsonl(Path(gt_path))
        pred_rows = _load_jsonl(Path(pred_path))

        matched = match_prediction_rows(gt_rows, pred_rows)
        summary, error_rows = evaluate_predictions(
            matched,
            prediction_keys={"pipeline": prediction_key},
        )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            md = build_error_analysis_markdown(error_rows)
            (out / "error_analysis.md").write_text(md, encoding="utf-8")

        return summary
