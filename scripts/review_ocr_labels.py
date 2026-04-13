"""Filter pseudo-labels by review bucket for manual review.

Example:
python scripts/review_ocr_labels.py --input data/processed/ocr/pseudo_labels.jsonl --bucket review --output outputs/review_bucket_review.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter OCR pseudo-labels by review bucket.")
    parser.add_argument("--input", default="data/processed/ocr/pseudo_labels.jsonl")
    parser.add_argument("--bucket", default="review", choices=["accept", "review", "reject"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("outputs") / f"ocr_{args.bucket}.jsonl"
    rows = [row for row in read_jsonl(input_path) if row.get("review_bucket") == args.bucket]
    if args.limit is not None:
        rows = rows[: args.limit]
    write_jsonl(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
