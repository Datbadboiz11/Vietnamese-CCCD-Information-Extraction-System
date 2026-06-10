"""Evaluate address/origin OCR and post-processing against reviewed GT.

This script turns ``data/processed/ocr/reviewed.jsonl`` into a fixed
regression benchmark for the two hardest CCCD fields:

  - address  -> place_of_residence
  - origin   -> place_of_origin

It compares raw OCR candidates and current post-processors, then writes:

  - summary.json
  - rows.csv
  - errors.json
  - report.md

Usage:
    python scripts/evaluate_address_origin_gt.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import _bootstrap  # noqa: F401

from src.evaluation.ocr_metrics import character_error_rate, word_error_rate
from src.ocr.ensemble import select_best_ocr_result
from src.ocr.types import OCRResult
from src.ocr.utils import cleanup_ocr_text, normalize_text_for_field
from src.parsing.validators import CCCDParser

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_INPUT = Path("data/processed/ocr/reviewed.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/address_origin_gt")
DEFAULT_FIELDS = ("address", "origin")

FIELD_TO_CANONICAL = {
    "address": "place_of_residence",
    "origin": "place_of_origin",
}

LABEL_RE = re.compile(
    r"\b(?:place|resid\w*|origin|orgin|address|"
    r"nguy[eê]n\s*qu[aáâ]n|qu[eê]\s*qu[aáâ]n|"
    r"n[oơ]i\s*(?:d?k?h?k?\s*)?th[uư]?[oơờ]?ng\s*tr[uú])\b",
    re.IGNORECASE,
)

VIET_DIAC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


def ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", normalize_text(text).lower())
    folded = folded.replace("đ", "d").replace("Đ", "d")
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")


def punctuation_fold(text: str) -> str:
    return re.sub(r"[^\w\s]", "", normalize_text(text).lower(), flags=re.UNICODE)


def numeric_groups(text: str) -> list[str]:
    return re.findall(r"\d+", text or "")


def levenshtein(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_ch in enumerate(reference, start=1):
        current = [i]
        for j, hyp_ch in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(ref_ch != hyp_ch),
                )
            )
        previous = current
    return previous[-1]


def classify_error(gt: str, pred: str) -> str:
    gt_n = normalize_text(gt)
    pred_n = normalize_text(pred)

    if pred_n == gt_n:
        return "correct"
    if not pred_n:
        return "empty_prediction"
    if pred_n.lower() == gt_n.lower():
        return "case_only"
    if punctuation_fold(pred_n) == punctuation_fold(gt_n):
        return "punctuation_only"
    if ascii_fold(pred_n) == ascii_fold(gt_n):
        return "diacritic_only"
    if numeric_groups(gt_n) != numeric_groups(pred_n) and numeric_groups(gt_n):
        return "numeric_mismatch"
    if LABEL_RE.search(pred_n):
        return "label_bleed"

    cer = character_error_rate(gt_n, pred_n)
    if len(pred_n) < len(gt_n) * 0.55:
        return "truncated"
    if len(pred_n) > len(gt_n) * 1.45:
        return "hallucination"
    if cer <= 0.15:
        return "near_match_cer15"
    if cer <= 0.30:
        return "moderate_cer30"
    return "major_error"


@dataclass
class EngineStats:
    count: int = 0
    exact: int = 0
    cer_sum: float = 0.0
    wer_sum: float = 0.0
    changed_from_best: int = 0
    fixed_best_error: int = 0
    broke_best_correct: int = 0
    improved_cer: int = 0
    worsened_cer: int = 0
    same_cer: int = 0
    categories: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        gt: str,
        pred: str,
        *,
        best_pred: str,
        best_cer: float,
        best_exact: bool,
    ) -> None:
        gt_n = normalize_text(gt)
        pred_n = normalize_text(pred)
        cer = character_error_rate(gt_n, pred_n)
        wer = word_error_rate(gt_n, pred_n)
        exact = pred_n == gt_n

        self.count += 1
        self.exact += int(exact)
        self.cer_sum += cer
        self.wer_sum += wer
        self.categories[classify_error(gt_n, pred_n)] += 1

        if pred_n != normalize_text(best_pred):
            self.changed_from_best += 1
        if exact and not best_exact:
            self.fixed_best_error += 1
        if best_exact and not exact:
            self.broke_best_correct += 1

        if cer < best_cer:
            self.improved_cer += 1
        elif cer > best_cer:
            self.worsened_cer += 1
        else:
            self.same_cer += 1

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "exact": 0,
                "exact_rate": 0.0,
                "cer": 0.0,
                "wer": 0.0,
                "categories": {},
            }
        return {
            "count": self.count,
            "exact": self.exact,
            "exact_rate": round(self.exact / self.count, 4),
            "cer": round(self.cer_sum / self.count, 4),
            "wer": round(self.wer_sum / self.count, 4),
            "changed_from_best": self.changed_from_best,
            "fixed_best_error": self.fixed_best_error,
            "broke_best_correct": self.broke_best_correct,
            "improved_cer": self.improved_cer,
            "worsened_cer": self.worsened_cer,
            "same_cer": self.same_cer,
            "categories": dict(self.categories.most_common()),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parser_value(parser: CCCDParser, field_name: str, text: str, confidence: float) -> str:
    result = parser.parse_field(field_name, text, confidence)
    if result is None:
        return text
    return result.value or ""


def ocr_result(engine: str, field_name: str, text: str, confidence: float) -> OCRResult:
    cleaned = cleanup_ocr_text(text or "", field_name)
    return OCRResult(
        text=cleaned,
        score=float(confidence or 0.0),
        engine=engine,
        normalized_text=normalize_text_for_field(cleaned, field_name),
    )


def build_predictions(row: dict[str, Any], parser: CCCDParser) -> dict[str, str]:
    field_name = str(row.get("class") or row.get("field_name") or "")
    best_text = str(row.get("best_text") or "")
    best_conf = float(row.get("best_conf") or 0.0)
    viet_text = str(row.get("text_vietocr") or "")
    viet_conf = float(row.get("conf_vietocr") or 0.0)
    paddle_text = str(row.get("text_paddleocr") or row.get("text_paddle") or "")
    paddle_conf = float(row.get("conf_paddleocr") or row.get("conf_paddle") or 0.0)

    selector_result = select_best_ocr_result(
        field_name,
        ocr_result("vietocr", field_name, viet_text, viet_conf),
        ocr_result("paddleocr", field_name, paddle_text, paddle_conf),
    )

    predictions: dict[str, str] = {
        "best_text": best_text,
        "vietocr": viet_text,
        "paddleocr": paddle_text,
        "selector_current": selector_result.text,
        "parser_on_best": parser_value(parser, field_name, best_text, best_conf),
    }

    try:
        from src.parsing.vn_places import correct_place_text

        predictions["vn_places_on_best"] = correct_place_text(best_text)
    except Exception as exc:
        predictions["vn_places_on_best"] = best_text
        predictions["vn_places_error"] = str(exc)

    try:
        from src.parsing.address_corrector import correct_address

        predictions["legacy_address_corrector"] = correct_address(best_text)
    except Exception as exc:
        predictions["legacy_address_corrector"] = best_text
        predictions["legacy_address_corrector_error"] = str(exc)

    return predictions


def summarize(
    rows: list[dict[str, Any]],
    engine_names: list[str],
) -> dict[str, Any]:
    stats_by_engine: dict[str, EngineStats] = {
        engine: EngineStats() for engine in engine_names
    }
    stats_by_field: dict[str, dict[str, EngineStats]] = {
        field: {engine: EngineStats() for engine in engine_names}
        for field in DEFAULT_FIELDS
    }

    for row in rows:
        gt = str(row["ground_truth_text"])
        best_pred = str(row["pred_best_text"])
        best_cer = float(row["cer_best_text"])
        best_exact = bool(row["exact_best_text"])
        field_name = str(row["class"])

        for engine in engine_names:
            pred = str(row[f"pred_{engine}"])
            stats_by_engine[engine].add(
                gt,
                pred,
                best_pred=best_pred,
                best_cer=best_cer,
                best_exact=best_exact,
            )
            stats_by_field[field_name][engine].add(
                gt,
                pred,
                best_pred=best_pred,
                best_cer=best_cer,
                best_exact=best_exact,
            )

    return {
        "overall": {
            engine: stats.as_dict()
            for engine, stats in stats_by_engine.items()
        },
        "by_field": {
            field_name: {
                engine: stats.as_dict()
                for engine, stats in field_stats.items()
            }
            for field_name, field_stats in stats_by_field.items()
        },
    }


def build_error_samples(
    rows: list[dict[str, Any]],
    engine_names: list[str],
    max_per_engine: int,
) -> dict[str, list[dict[str, Any]]]:
    errors: dict[str, list[dict[str, Any]]] = {}
    for engine in engine_names:
        engine_errors = [
            {
                "field": row["class"],
                "crop_path": row["crop_path"],
                "review_bucket": row.get("review_bucket", ""),
                "ground_truth_text": row["ground_truth_text"],
                "prediction": row[f"pred_{engine}"],
                "cer": row[f"cer_{engine}"],
                "category": row[f"category_{engine}"],
            }
            for row in rows
            if not row[f"exact_{engine}"]
        ]
        engine_errors.sort(key=lambda item: (-float(item["cer"]), item["field"], item["crop_path"]))
        errors[engine] = engine_errors[:max_per_engine]
    return errors


def write_report(
    path: Path,
    summary: dict[str, Any],
    errors: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = [
        "# Address/Origin GT Benchmark",
        "",
        "## Overall",
        "",
        "| Engine | Count | Exact | Exact % | CER | WER | Fixed Best | Broke Best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for engine, stats in summary["overall"].items():
        lines.append(
            f"| {engine} | {stats['count']} | {stats['exact']} | "
            f"{stats['exact_rate'] * 100:.1f} | {stats['cer']:.4f} | "
            f"{stats['wer']:.4f} | {stats.get('fixed_best_error', 0)} | "
            f"{stats.get('broke_best_correct', 0)} |"
        )

    lines.extend(["", "## By Field", ""])
    for field_name, field_stats in summary["by_field"].items():
        lines.extend([
            f"### {field_name}",
            "",
            "| Engine | Count | Exact | Exact % | CER | WER |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for engine, stats in field_stats.items():
            lines.append(
                f"| {engine} | {stats['count']} | {stats['exact']} | "
                f"{stats['exact_rate'] * 100:.1f} | {stats['cer']:.4f} | "
                f"{stats['wer']:.4f} |"
            )
        lines.append("")

    lines.extend(["## Worst Errors", ""])
    for engine, samples in errors.items():
        lines.extend([f"### {engine}", ""])
        if not samples:
            lines.append("No errors.")
            lines.append("")
            continue
        for sample in samples[:10]:
            lines.append(
                f"- `{sample['field']}` CER={sample['cer']:.4f} "
                f"category={sample['category']} bucket={sample['review_bucket']}"
            )
            lines.append(f"  GT: {sample['ground_truth_text']}")
            lines.append(f"  Pred: {sample['prediction']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate address/origin predictions against reviewed GT.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS))
    parser.add_argument("--max-error-samples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = {field.strip() for field in args.fields if field.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.input)
    parser = CCCDParser()
    benchmark_rows: list[dict[str, Any]] = []
    engine_names: list[str] = []

    for row in source_rows:
        field_name = str(row.get("class") or row.get("field_name") or "")
        gt = str(row.get("ground_truth_text") or "")
        if field_name not in fields or not gt.strip():
            continue

        predictions = build_predictions(row, parser)
        engine_names = [key for key in predictions if not key.endswith("_error")]

        output_row: dict[str, Any] = {
            "crop_path": row.get("crop_path") or "",
            "image_path": row.get("image_path") or "",
            "split": row.get("split") or "",
            "class": field_name,
            "canonical_field": FIELD_TO_CANONICAL.get(field_name, field_name),
            "review_bucket": row.get("review_bucket") or "",
            "needs_review": bool(row.get("needs_review")),
            "best_engine": row.get("best_engine") or "",
            "best_conf": float(row.get("best_conf") or 0.0),
            "ground_truth_text": normalize_text(gt),
        }

        for engine, pred in predictions.items():
            if engine.endswith("_error"):
                continue
            pred_n = normalize_text(pred)
            gt_n = output_row["ground_truth_text"]
            output_row[f"pred_{engine}"] = pred_n
            output_row[f"exact_{engine}"] = pred_n == gt_n
            output_row[f"cer_{engine}"] = character_error_rate(gt_n, pred_n)
            output_row[f"wer_{engine}"] = word_error_rate(gt_n, pred_n)
            output_row[f"category_{engine}"] = classify_error(gt_n, pred_n)

        benchmark_rows.append(output_row)

    if not benchmark_rows:
        raise SystemExit("No address/origin rows with ground_truth_text were found.")

    summary = summarize(benchmark_rows, engine_names)
    summary["metadata"] = {
        "input": str(args.input),
        "num_source_rows": len(source_rows),
        "num_benchmark_rows": len(benchmark_rows),
        "fields": sorted(fields),
        "engines": engine_names,
    }

    errors = build_error_samples(benchmark_rows, engine_names, args.max_error_samples)

    rows_path = args.output_dir / "rows.csv"
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", summary, errors)

    print(f"Input rows:      {len(source_rows)}")
    print(f"Benchmark rows:  {len(benchmark_rows)}")
    print(f"Output dir:      {args.output_dir}")
    print()
    print("Overall:")
    for engine, stats in summary["overall"].items():
        print(
            f"  {engine:<25s} "
            f"exact={stats['exact']:>3}/{stats['count']} "
            f"({stats['exact_rate'] * 100:5.1f}%) "
            f"CER={stats['cer']:.4f} "
            f"fixed={stats.get('fixed_best_error', 0):>3} "
            f"broke={stats.get('broke_best_correct', 0):>3}"
        )


if __name__ == "__main__":
    main()
