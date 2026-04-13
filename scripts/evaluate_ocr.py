"""Evaluate OCR predictions against reviewed transcripts.

Default workflow:
python scripts/evaluate_ocr.py --gt data/processed/ocr/reviewed.jsonl --pred data/processed/ocr/pseudo_labels.jsonl

This writes:
- outputs/ocr_eval/ocr_metrics_summary.json
- outputs/ocr_eval/ocr_metrics_summary.csv
- outputs/ocr_eval/ocr_error_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.evaluation.ocr_metrics import TextNormalizationOptions, build_error_analysis_markdown, evaluate_predictions, match_prediction_rows

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VietOCR, PaddleOCR, and ensemble OCR results.")
    parser.add_argument("--gt", required=True, help="JSONL with reviewed ground-truth transcripts.")
    parser.add_argument("--pred", required=True, help="JSONL with prediction rows.")
    parser.add_argument("--output-dir", default="outputs/ocr_eval")
    parser.add_argument("--gt-text-key", default="ground_truth_text")
    parser.add_argument("--keep-case", action="store_true")
    parser.add_argument("--no-strip-extra-spaces", action="store_true")
    parser.add_argument("--unicode-form", default="NFC")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["engine", "scope", "field_name", "count", "cer", "wer", "exact_match_rate"],
        )
        writer.writeheader()
        for engine_name, engine_summary in summary["engines"].items():
            overall = engine_summary["overall"]
            writer.writerow({"engine": engine_name, "scope": "overall", "field_name": "*", **overall})
            for field_name, metrics in engine_summary["per_field"].items():
                writer.writerow({"engine": engine_name, "scope": "per_field", "field_name": field_name, **metrics})


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    gt_rows = read_jsonl(Path(args.gt))
    pred_rows = read_jsonl(Path(args.pred))
    matched_rows = match_prediction_rows(gt_rows, pred_rows)
    if not matched_rows:
        raise ValueError("No matching rows found between ground truth and predictions.")
    LOGGER.info("Matched %d rows for OCR evaluation.", len(matched_rows))

    for row in matched_rows:
        if args.gt_text_key != "ground_truth_text":
            row["ground_truth_text"] = row.get(args.gt_text_key) or row.get("ground_truth_text") or ""
        if "text_paddleocr" not in row:
            row["text_paddleocr"] = row.get("text_paddle", "")

    normalization = TextNormalizationOptions(
        strip_extra_spaces=not args.no_strip_extra_spaces,
        unicode_form=args.unicode_form,
        case_sensitive=args.keep_case,
    )
    prediction_keys = {
        "vietocr": "text_vietocr",
        "paddleocr": "text_paddleocr",
        "ensemble": "best_text",
    }
    summary, error_rows = evaluate_predictions(
        matched_rows,
        prediction_keys,
        ground_truth_key="ground_truth_text",
        normalization=normalization,
    )
    LOGGER.info("Computed metrics for %d OCR engines.", len(prediction_keys))

    output_dir = Path(args.output_dir)
    write_json(output_dir / "ocr_metrics_summary.json", summary)
    write_summary_csv(output_dir / "ocr_metrics_summary.csv", summary)
    write_markdown(output_dir / "ocr_error_analysis.md", build_error_analysis_markdown(error_rows))
    LOGGER.info("Saved OCR metrics to %s", output_dir)


if __name__ == "__main__":
    main()
