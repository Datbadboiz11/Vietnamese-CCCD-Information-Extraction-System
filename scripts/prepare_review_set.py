from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed.jsonl from pseudo OCR outputs.")
    parser.add_argument("--input", default="data/processed/ocr/pseudo_labels.jsonl")
    parser.add_argument("--output", default="data/processed/ocr/reviewed.jsonl")
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument("--needs-review-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_jsonl(input_path)
    if args.splits:
        split_set = set(args.splits)
        rows = [row for row in rows if row.get("split") in split_set]
    if args.buckets:
        bucket_set = set(args.buckets)
        rows = [row for row in rows if row.get("review_bucket") in bucket_set]
    if args.needs_review_only:
        rows = [row for row in rows if bool(row.get("needs_review"))]
    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        row.setdefault("ground_truth_text", "")

    write_jsonl(output_path, rows)
    print(f"Prepared {len(rows)} rows at {output_path}")


if __name__ == "__main__":
    main()
