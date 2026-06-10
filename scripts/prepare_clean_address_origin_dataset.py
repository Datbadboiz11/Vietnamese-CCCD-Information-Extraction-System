from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    ("reviewed.jsonl", 1),
    ("data/review_part1.jsonl", 2),
    ("data/review_part3.jsonl", 3),
]


def clean_label(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def suspicious_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if any(marker in text for marker in ("Ã", "áº", "á»", "Ä", "Æ")):
        reasons.append("possible_mojibake")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        reasons.append("control_character")
    if len(text) < 5:
        reasons.append("too_short")
    if re.search(r"\baaps\b|\baps\b", text, flags=re.IGNORECASE):
        reasons.append("possible_ap_typo")
    if re.search(r"[A-Za-zÀ-ỹ]{25,}", text):
        reasons.append("very_long_token")
    if re.search(r"([A-Za-zÀ-ỹ])\1\1", text, flags=re.IGNORECASE):
        reasons.append("triple_repeated_letter")
    return reasons


def path_key(path_value: str) -> str:
    return Path(path_value.replace("\\", "/")).name


def group_key(row: dict[str, Any], resolved_crop: Path) -> str:
    image_path = str(row.get("image_path") or "").replace("\\", "/")
    if image_path:
        return Path(image_path).stem

    crop_stem = resolved_crop.stem
    if "_" in crop_stem:
        return crop_stem.rsplit("_", 1)[0]
    return crop_stem


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = str(path.relative_to(PROJECT_ROOT))
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def index_crops(crop_root: Path, fields: set[str]) -> tuple[dict[tuple[str, str], Path], Counter]:
    crop_index: dict[tuple[str, str], Path] = {}
    stats: Counter = Counter()

    for field in sorted(fields):
        field_dir = crop_root / field
        if not field_dir.exists():
            stats["missing_field_dirs"] += 1
            continue
        for crop_path in field_dir.iterdir():
            if not crop_path.is_file() or crop_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            key = (field, crop_path.name)
            if key in crop_index:
                stats["duplicate_crop_filenames"] += 1
                continue
            crop_index[key] = crop_path

    return crop_index, stats


def split_groups(groups: list[str], seed: int, val_ratio: float, test_ratio: float) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = round(total * test_ratio)
    val_count = round(total * val_ratio)

    split_by_group: dict[str, str] = {}
    for idx, key in enumerate(shuffled):
        if idx < test_count:
            split_by_group[key] = "test"
        elif idx < test_count + val_count:
            split_by_group[key] = "val"
        else:
            split_by_group[key] = "train"
    return split_by_group


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_label_file(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row.get('dataset_crop_path', row['crop_path'])}\t{row['label']}\n")


def write_character_dict(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    chars = sorted({char for row in rows for char in row["label"] if char not in {"\t", "\n", "\r", " "}})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for char in chars:
            handle.write(char + "\n")
    return chars


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_specs = [(PROJECT_ROOT / source, priority) for source, priority in DEFAULT_SOURCES]
    fields = set(args.fields)
    crop_root = PROJECT_ROOT / args.crop_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_crops_dir = output_dir / "crops"

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if output_dir.exists() and args.overwrite:
        resolved_output = output_dir.resolve()
        resolved_project = PROJECT_ROOT.resolve()
        if resolved_output == resolved_project or resolved_project not in resolved_output.parents:
            raise ValueError(f"Refusing to remove unsafe output directory: {resolved_output}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    crop_index, crop_index_stats = index_crops(crop_root, fields)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    source_counts: Counter = Counter()

    for source_path, priority in source_specs:
        for row in read_jsonl(source_path):
            field = str(row.get("field_name") or row.get("class") or "").strip()
            if field not in fields:
                skipped["field_not_selected"] += 1
                continue

            label = clean_label(str(row.get("ground_truth_text") or ""))
            if not label:
                skipped["empty_label"] += 1
                continue

            original_crop_path = str(row.get("crop_path") or "")
            crop_name = path_key(original_crop_path)
            resolved_crop = crop_index.get((field, crop_name))
            if resolved_crop is None:
                skipped["missing_crop"] += 1
                continue

            key = (field, crop_name)
            item = {
                "field_name": field,
                "label": label,
                "resolved_source_crop_path": str(resolved_crop.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "original_crop_path": original_crop_path.replace("\\", "/"),
                "image_path": str(row.get("image_path") or "").replace("\\", "/"),
                "source_file": row["_source_file"].replace("\\", "/"),
                "source_line": row["_source_line"],
                "source_priority": priority,
                "crop_filename": crop_name,
                "group_key": group_key(row, resolved_crop),
            }

            previous = selected.get(key)
            if previous is not None and previous["label"] != label:
                conflicts.append(
                    {
                        "field_name": field,
                        "crop_filename": crop_name,
                        "kept_label": label if priority >= previous["source_priority"] else previous["label"],
                        "replaced_label": previous["label"] if priority >= previous["source_priority"] else label,
                        "kept_source": item["source_file"] if priority >= previous["source_priority"] else previous["source_file"],
                        "replaced_source": previous["source_file"] if priority >= previous["source_priority"] else item["source_file"],
                    }
                )

            if previous is None or priority >= previous["source_priority"]:
                selected[key] = item
                source_counts[item["source_file"]] += 1

    rows = []
    rejected_suspicious: list[dict[str, Any]] = []
    for row in selected.values():
        reasons = suspicious_reasons(row["label"])
        if reasons and not args.keep_suspicious:
            rejected_row = dict(row)
            rejected_row["suspicious_reasons"] = reasons
            rejected_suspicious.append(rejected_row)
            continue
        rows.append(row)

    rows.sort(key=lambda row: (row["group_key"], row["field_name"], row["crop_filename"]))

    split_by_group = split_groups(
        sorted({row["group_key"] for row in rows}),
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    for row in rows:
        split = split_by_group[row["group_key"]]
        row["split"] = split
        target_crop = output_crops_dir / row["field_name"] / row["crop_filename"]
        target_crop.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / row["resolved_source_crop_path"], target_crop)
        row["crop_path"] = str(target_crop.relative_to(PROJECT_ROOT)).replace("\\", "/")
        row["dataset_crop_path"] = str(target_crop.relative_to(output_dir)).replace("\\", "/")
        row["label_length"] = len(row["label"])

    write_jsonl(output_dir / "manifest.jsonl", rows)
    write_jsonl(output_dir / "conflicts.jsonl", conflicts)
    write_jsonl(output_dir / "rejected_suspicious_labels.jsonl", rejected_suspicious)

    suspicious_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons = suspicious_reasons(row["label"])
        if reasons:
            suspicious_row = dict(row)
            suspicious_row["suspicious_reasons"] = reasons
            suspicious_rows.append(suspicious_row)
    write_jsonl(output_dir / "suspicious_labels.jsonl", suspicious_rows)

    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        write_label_file(output_dir / f"{split}_label.txt", split_rows)
        write_jsonl(output_dir / f"{split}.jsonl", split_rows)

    by_field_dir = output_dir / "by_field"
    by_field_dir.mkdir(exist_ok=True)
    for field in sorted(fields):
        for split in ["train", "val", "test"]:
            split_rows = [row for row in rows if row["field_name"] == field and row["split"] == split]
            write_label_file(by_field_dir / f"{field}_{split}_label.txt", split_rows)

    dict_chars = write_character_dict(output_dir / "dict.txt", rows)

    report = {
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_files": [{"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "priority": priority} for path, priority in source_specs],
        "fields": sorted(fields),
        "total_samples": len(rows),
        "samples_by_field": Counter(row["field_name"] for row in rows),
        "samples_by_split": Counter(row["split"] for row in rows),
        "samples_by_field_split": Counter(f"{row['field_name']}:{row['split']}" for row in rows),
        "selected_by_source": Counter(row["source_file"] for row in rows),
        "unique_groups": len(split_by_group),
        "conflicts_resolved": len(conflicts),
        "rejected_suspicious_labels": len(rejected_suspicious),
        "rejected_suspicious_by_reason": Counter(reason for row in rejected_suspicious for reason in row["suspicious_reasons"]),
        "suspicious_labels": len(suspicious_rows),
        "suspicious_by_reason": Counter(reason for row in suspicious_rows for reason in row["suspicious_reasons"]),
        "skipped": skipped,
        "crop_index": crop_index_stats,
        "character_count": len(dict_chars),
        "max_label_length": max((row["label_length"] for row in rows), default=0),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (output_dir / "dataset_meta.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "name": "cccd_address_origin_clean",
                "label_format": "relative_image_path<TAB>text",
                "fields": sorted(fields),
                "splits": dict(report["samples_by_split"]),
                "samples_by_field": dict(report["samples_by_field"]),
                "character_count": len(dict_chars),
                "max_label_length": report["max_label_length"],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean OCR fine-tuning dataset for address/origin fields.")
    parser.add_argument("--output-dir", default="data/processed/ocr_address_origin_clean")
    parser.add_argument("--crop-root", default="data/processed/ocr/field_crops")
    parser.add_argument("--fields", nargs="+", default=["address", "origin"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--keep-suspicious", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = build_dataset(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
