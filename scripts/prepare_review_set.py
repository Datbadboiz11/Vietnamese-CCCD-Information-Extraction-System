from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import _bootstrap  # noqa: F401


DEFAULT_INPUT_CANDIDATES = (
    Path("data/processed/ocr/pseudo_labels.jsonl"),
    Path("data/processed/ocr_full/pseudo_labels.jsonl"),
    Path("data/processed/datacu/pseudo_labels.jsonl"),
)

REVIEW_PRESETS = {
    "custom": {
        "output": Path("data/processed/ocr/reviewed_val.jsonl"),
        "splits": ["val"],
        "sort_by": "none",
    },
    "test_deep": {
        "output": Path("data/processed/ocr/review_test_deep.jsonl"),
        "splits": ["test"],
        "sort_by": "source_image",
    },
    "val_medium": {
        "output": Path("data/processed/ocr/review_val_medium.jsonl"),
        "splits": ["val"],
        "max_conf": 0.85,
        "sort_by": "best_conf_asc",
    },
    "train_confidence": {
        "output": Path("data/processed/ocr/review_train_confidence.jsonl"),
        "splits": ["train"],
        "max_conf": 0.85,
        "sort_by": "best_conf_asc",
    },
}


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


def resolve_input_path(raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value)
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]


def apply_preset(args: argparse.Namespace) -> tuple[Path, Path]:
    preset = REVIEW_PRESETS[args.preset]
    input_path = resolve_input_path(args.input)
    output_path = Path(args.output) if args.output else preset["output"]

    if args.splits is None:
        args.splits = list(preset.get("splits", []))
    if args.max_conf is None and "max_conf" in preset:
        args.max_conf = float(preset["max_conf"])
    if args.sort_by is None:
        args.sort_by = preset.get("sort_by", "none")

    return input_path, output_path


def sort_rows(rows: list[dict], sort_by: str) -> list[dict]:
    if sort_by == "best_conf_asc":
        return sorted(rows, key=lambda row: float(row.get("best_conf", 0.0) or 0.0))
    if sort_by == "best_conf_desc":
        return sorted(rows, key=lambda row: float(row.get("best_conf", 0.0) or 0.0), reverse=True)
    if sort_by == "source_image":
        return sorted(rows, key=lambda row: (row.get("source_image", ""), row.get("ann_id", 0)))
    return rows


def print_summary(rows: list[dict]) -> None:
    split_counts = collections.Counter(row.get("split", "missing") for row in rows)
    bucket_counts = collections.Counter(row.get("review_bucket", "missing") for row in rows)
    print("Summary by split:", dict(sorted(split_counts.items())))
    print("Summary by bucket:", dict(sorted(bucket_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a reviewed OCR set with presets for test/val/train review workflows."
    )
    parser.add_argument("--preset", choices=sorted(REVIEW_PRESETS), default="custom")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument("--needs-review-only", action="store_true")
    parser.add_argument("--min-conf", type=float, default=None)
    parser.add_argument("--max-conf", type=float, default=None)
    parser.add_argument(
        "--sort-by",
        choices=("none", "best_conf_asc", "best_conf_desc", "source_image"),
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path, output_path = apply_preset(args)

    rows = read_jsonl(input_path)
    if args.splits:
        split_set = set(args.splits)
        rows = [row for row in rows if row.get("split") in split_set]
    if args.buckets:
        bucket_set = set(args.buckets)
        rows = [row for row in rows if row.get("review_bucket") in bucket_set]
    if args.needs_review_only:
        rows = [row for row in rows if bool(row.get("needs_review"))]
    if args.min_conf is not None:
        rows = [row for row in rows if float(row.get("best_conf", 0.0) or 0.0) >= args.min_conf]
    if args.max_conf is not None:
        rows = [row for row in rows if float(row.get("best_conf", 0.0) or 0.0) < args.max_conf]
    rows = sort_rows(rows, args.sort_by or "none")
    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        row.setdefault("ground_truth_text", "")

    write_jsonl(output_path, rows)
    print(f"Input: {input_path}")
    print(f"Preset: {args.preset}")
    if args.max_conf is not None or args.min_conf is not None:
        print(f"Confidence filter: min={args.min_conf}, max={args.max_conf}")
    print_summary(rows)
    print(f"Prepared {len(rows)} rows at {output_path}")


if __name__ == "__main__":
    main()
