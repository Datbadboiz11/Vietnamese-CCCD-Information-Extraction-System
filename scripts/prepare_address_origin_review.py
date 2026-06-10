"""Prepare a review set focused on address/origin fields for ground truth annotation.

Reads pseudo_labels.jsonl, deduplicates by source image (1 crop per unique card
per field), prioritizes test→val splits, and sorts by confidence ascending so
the hardest cases are reviewed first.

Usage:
    python scripts/prepare_address_origin_review.py [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

DEFAULT_INPUT = Path("data/processed/ocr/pseudo_labels.jsonl")
DEFAULT_OUTPUT = Path("data/processed/ocr/address_origin_review.jsonl")

TARGET_CLASSES = {"address", "origin"}
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _source_base(source_image: str) -> str:
    m = re.match(r"(image\d+)", source_image)
    return m.group(1) if m else source_image


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare address/origin review set for GT annotation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=200, help="Max records per field class")
    args = parser.parse_args()

    all_rows = read_jsonl(args.input)

    selected: list[dict] = []
    for cls in sorted(TARGET_CLASSES):
        cls_rows = [r for r in all_rows if r.get("class") == cls]

        cls_rows.sort(key=lambda r: (
            SPLIT_PRIORITY.get(r.get("split", "train"), 99),
            float(r.get("best_conf", 0) or 0),
        ))

        seen_bases: set[str] = set()
        deduped: list[dict] = []
        for row in cls_rows:
            base = _source_base(row.get("source_image", ""))
            if base in seen_bases:
                continue
            seen_bases.add(base)
            deduped.append(row)

        picked = deduped[: args.limit]
        for row in picked:
            row.setdefault("ground_truth_text", "")
        selected.extend(picked)

    write_jsonl(args.output, selected)

    addr_count = sum(1 for r in selected if r.get("class") == "address")
    orig_count = sum(1 for r in selected if r.get("class") == "origin")
    splits = {}
    for r in selected:
        s = r.get("split", "?")
        splits[s] = splits.get(s, 0) + 1

    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Total:   {len(selected)} records (address={addr_count}, origin={orig_count})")
    print(f"Splits:  {splits}")
    print(f"\nNext step: python review_tool.py --input {args.output}")


if __name__ == "__main__":
    main()
