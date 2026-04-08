from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.ocr.metrics import collect_error_samples, evaluate_rows


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_per_field_csv(path: Path, summary: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["backend", "field_name", "count", "cer", "wer", "exact_match"])
        writer.writeheader()
        for backend, backend_summary in summary.items():
            for field_name, metrics in backend_summary["per_field"].items():
                writer.writerow({"backend": backend, "field_name": field_name, **metrics})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR results using reviewed transcripts.")
    parser.add_argument("--input", default="data/processed/ocr/reviewed.jsonl")
    parser.add_argument("--output-dir", default="outputs/ocr_eval")
    parser.add_argument("--ground-truth-key", default="ground_truth_text")
    parser.add_argument("--error-limit", type=int, default=300)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    if not any(str(row.get(args.ground_truth_key, "")).strip() for row in rows):
        raise ValueError(
            "No ground-truth text found. Fill `ground_truth_text` in reviewed.jsonl before running OCR evaluation."
        )

    backends = {
        "vietocr": "text_vietocr",
        "paddleocr": "text_paddle",
        "ensemble": "best_text",
    }

    summary: dict[str, dict] = {}
    all_errors: list[dict] = []
    for backend_name, key in backends.items():
        backend_summary = evaluate_rows(rows, prediction_key=key, ground_truth_key=args.ground_truth_key)
        summary[backend_name] = backend_summary
        backend_errors = collect_error_samples(
            rows,
            prediction_key=key,
            ground_truth_key=args.ground_truth_key,
            backend_name=backend_name,
        )
        all_errors.extend(backend_errors[: args.error_limit])

    output_dir = Path(args.output_dir)
    write_json(output_dir / "metrics_summary.json", summary)
    write_per_field_csv(output_dir / "per_field_metrics.csv", summary)
    write_jsonl(output_dir / "error_samples.jsonl", all_errors)
    print(f"Saved OCR evaluation to {output_dir}")


if __name__ == "__main__":
    main()
