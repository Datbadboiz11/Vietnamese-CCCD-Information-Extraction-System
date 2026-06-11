"""Evaluate the end-to-end CCCD pipeline against 5-field validation GT.

Example:
    python scripts/evaluate_full_pipeline_5fields.py --limit 10
    python scripts/evaluate_full_pipeline_5fields.py

Outputs:
    outputs/full_pipeline_eval/predictions.jsonl
    outputs/full_pipeline_eval/metrics_summary.json
    outputs/full_pipeline_eval/metrics_summary.csv
    outputs/full_pipeline_eval/error_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.evaluation.ocr_metrics import (
    TextNormalizationOptions,
    build_error_analysis_markdown,
    evaluate_predictions,
    normalize_metric_text,
)
from src.pipeline import CCCDPipeline

LOGGER = logging.getLogger(__name__)

FIELD_ALIASES = {
    "id": "id_number",
    "name": "full_name",
    "birth": "date_of_birth",
    "origin": "place_of_origin",
    "address": "place_of_residence",
}
CANONICAL_TO_CLASS = {
    "id_number": "id",
    "full_name": "name",
    "date_of_birth": "birth",
    "place_of_origin": "origin",
    "place_of_residence": "address",
}
REQUIRED_FIELDS = tuple(CANONICAL_TO_CLASS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full CCCD pipeline on 5-field GT JSONL.")
    parser.add_argument(
        "--gt",
        default="data/processed/eval/reviewed_finalval_clean.jsonl",
        help="JSONL containing source_image/image_path, field_name, and ground_truth_text.",
    )
    parser.add_argument("--output-dir", default="outputs/full_pipeline_eval")
    parser.add_argument("--card-detector", default="model/card_detector/best.pt")
    parser.add_argument("--field-detector", default="model/field_detector/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N images.")
    parser.add_argument("--use-ensemble", action="store_true")
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--no-text-region-refinement", action="store_true")
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


def canonical_field(field_name: str | None) -> str:
    value = (field_name or "").strip()
    return FIELD_ALIASES.get(value, value)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["engine", "scope", "field_name", "count", "cer", "wer", "exact_match_rate"],
        )
        writer.writeheader()
        for engine_name, engine_summary in summary["engines"].items():
            writer.writerow({"engine": engine_name, "scope": "overall", "field_name": "*", **engine_summary["overall"]})
            for field_name, metrics in engine_summary["per_field"].items():
                writer.writerow({"engine": engine_name, "scope": "per_field", "field_name": field_name, **metrics})


def group_ground_truth(rows: list[dict[str, Any]], root: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gt = str(row.get("ground_truth_text") or "").strip()
        if not gt:
            continue
        image_path = str(row.get("source_image") or row.get("image_path") or "").strip()
        if not image_path:
            continue
        image = Path(image_path)
        if not image.is_absolute():
            image = root / image
        row = dict(row)
        row["field_name"] = canonical_field(row.get("field_name") or row.get("class"))
        row["resolved_image_path"] = str(image)
        grouped[str(image)].append(row)
    return dict(grouped)


def prediction_map_from_result(result: Any) -> tuple[dict[str, str], dict[str, float]]:
    if result.parsed_info is None:
        return {}, {}
    info = result.parsed_info
    predictions = {
        "id_number": info.id_number or "",
        "full_name": info.full_name or "",
        "date_of_birth": info.date_of_birth or "",
        "place_of_origin": info.place_of_origin or "",
        "place_of_residence": info.place_of_residence or "",
    }
    confidences = dict(getattr(info, "confidence_scores", {}) or {})
    return predictions, confidences


def raw_ocr_for_field(result: Any, field_name: str) -> tuple[str, float]:
    cls_name = CANONICAL_TO_CLASS.get(field_name, field_name)
    text, score = result.ocr_results.get(cls_name, ("", 0.0))
    return str(text or ""), float(score or 0.0)


def debug_model_for_field(result: Any, field_name: str) -> str:
    cls_name = CANONICAL_TO_CLASS.get(field_name, field_name)
    debug = (result.field_debug or {}).get(cls_name, {})
    return str(debug.get("vietocr_model") or debug.get("engine") or "")


def image_exact_summary(rows: list[dict[str, Any]], options: TextNormalizationOptions) -> dict[str, Any]:
    by_image: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        field = row["field_name"]
        if field not in REQUIRED_FIELDS or field in by_image[row["image_path"]]:
            continue
        by_image[row["image_path"]][field] = {
            "gt": row["ground_truth_text"],
            "pred": row["predicted_text"],
        }

    complete = 0
    all_exact = 0
    for fields in by_image.values():
        if not all(field in fields for field in REQUIRED_FIELDS):
            continue
        complete += 1
        if all(
            normalize_metric_text(fields[field]["gt"], options)
            == normalize_metric_text(fields[field]["pred"], options)
            for field in REQUIRED_FIELDS
        ):
            all_exact += 1

    return {
        "complete_5field_images": complete,
        "all_5_fields_exact": all_exact,
        "all_5_fields_exact_rate": all_exact / complete if complete else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    gt_path = (root / args.gt).resolve() if not Path(args.gt).is_absolute() else Path(args.gt)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    gt_rows = read_jsonl(gt_path)
    grouped = group_ground_truth(gt_rows, root)
    image_items = list(grouped.items())
    if args.limit and args.limit > 0:
        image_items = image_items[: args.limit]

    LOGGER.info("Loaded %d GT rows from %s", len(gt_rows), gt_path)
    LOGGER.info("Evaluating %d images", len(image_items))

    pipeline = CCCDPipeline(
        card_detector_path=args.card_detector,
        field_detector_path=args.field_detector,
        device=args.device,
        use_ensemble=args.use_ensemble,
        use_tta=args.use_tta,
        use_text_region_refinement=not args.no_text_region_refinement,
    )

    prediction_rows: list[dict[str, Any]] = []
    for idx, (image_path, image_gt_rows) in enumerate(image_items, start=1):
        LOGGER.info("[%d/%d] %s", idx, len(image_items), image_path)
        result = pipeline(image_path)
        predictions, confidences = prediction_map_from_result(result)
        errors = list(result.errors or [])
        warnings = list(result.warnings or [])

        for row in image_gt_rows:
            field = row["field_name"]
            raw_text, raw_conf = raw_ocr_for_field(result, field)
            prediction_rows.append(
                {
                    "image_path": image_path,
                    "crop_path": row.get("crop_path", ""),
                    "field_name": field,
                    "class": CANONICAL_TO_CLASS.get(field, field),
                    "ground_truth_text": row.get("ground_truth_text", ""),
                    "predicted_text": predictions.get(field, ""),
                    "confidence": confidences.get(field, raw_conf),
                    "raw_ocr_text": raw_text,
                    "raw_ocr_confidence": raw_conf,
                    "vietocr_model": debug_model_for_field(result, field),
                    "card_detected": bool(result.card_detected),
                    "field_detected": CANONICAL_TO_CLASS.get(field, field) in result.ocr_results,
                    "needs_review": bool(result.parsed_info.needs_review) if result.parsed_info else True,
                    "errors": errors,
                    "warnings": warnings,
                }
            )

    options = TextNormalizationOptions(
        strip_extra_spaces=not args.no_strip_extra_spaces,
        unicode_form=args.unicode_form,
        case_sensitive=args.keep_case,
    )
    summary, error_rows = evaluate_predictions(
        prediction_rows,
        prediction_keys={"full_pipeline": "predicted_text"},
        normalization=options,
    )
    summary["dataset"] = {
        "gt_path": str(gt_path),
        "gt_rows_total": len(gt_rows),
        "evaluated_images": len(image_items),
        "evaluated_rows": len(prediction_rows),
    }
    summary["pipeline"] = {
        "card_detector": args.card_detector,
        "field_detector": args.field_detector,
        "device": args.device,
        "use_ensemble": bool(args.use_ensemble),
        "use_tta": bool(args.use_tta),
        "use_text_region_refinement": not args.no_text_region_refinement,
    }
    summary["image_level"] = image_exact_summary(prediction_rows, options)

    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    write_json(output_dir / "metrics_summary.json", summary)
    write_summary_csv(output_dir / "metrics_summary.csv", summary)
    (output_dir / "error_analysis.md").write_text(
        build_error_analysis_markdown(error_rows, max_samples_per_engine=50),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "error_rows.jsonl", error_rows)
    LOGGER.info("Saved evaluation outputs to %s", output_dir)
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    summary = evaluate(args)
    metrics = summary["engines"]["full_pipeline"]["overall"]
    image_metrics = summary["image_level"]
    print(
        "overall: "
        f"count={metrics['count']} "
        f"exact={metrics['exact_match_rate']:.4f} "
        f"CER={metrics['cer']:.4f} "
        f"WER={metrics['wer']:.4f}"
    )
    print(
        "image-level: "
        f"all_5_exact={image_metrics['all_5_fields_exact']}/"
        f"{image_metrics['complete_5field_images']} "
        f"({image_metrics['all_5_fields_exact_rate']:.4f})"
    )


if __name__ == "__main__":
    main()
