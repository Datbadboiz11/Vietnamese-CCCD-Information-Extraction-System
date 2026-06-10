"""
Prepare clean address/origin dataset for PP-OCRv5 fine-tuning.

Reads from:
  - data/processed/ocr/finetune/train_annotation.txt (real field crops)
  - data/processed/ocr/finetune/val_annotation.txt
  - data/synthetic/images/ (synthetic augmentation)
  - data/processed/ocr/ppocrv5_rec/train_label.txt (existing synthetic labels)

Outputs to:
  - data/processed/ocr/ppocrv5_finetune_v2/train_label.txt
  - data/processed/ocr/ppocrv5_finetune_v2/val_label.txt
  - data/processed/ocr/ppocrv5_finetune_v2/dict.txt
  - data/processed/ocr/ppocrv5_finetune_v2/stats.json

Usage:
  python scripts/ocr/prepare_ppocrv5_finetune_data.py
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINETUNE_DIR = ROOT / "data" / "processed" / "ocr" / "finetune"
SYNTHETIC_LABEL = ROOT / "data" / "processed" / "ocr" / "ppocrv5_rec" / "train_label.txt"
OUTPUT_DIR = ROOT / "data" / "processed" / "ocr" / "ppocrv5_finetune_v2"

VIET_DIAC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

ADMIN_KEYWORDS = {
    "xã", "phường", "thị", "huyện", "quận", "tỉnh", "thành",
    "phố", "trấn", "xa", "tp", "tt", "tx",
}

LABEL_BLEED_RE = re.compile(
    r"place|origin|orgin|residen|nguyen\s*quan|nguyên\s*quán|nơi.*trú|"
    r"piace|phlace|placo|ofonig|dforig",
    re.IGNORECASE,
)


def _is_valid_label(label: str, field: str) -> bool:
    """Filter out garbage pseudo-labels."""
    if not label or len(label.strip()) < 3:
        return False
    label = label.strip()

    if LABEL_BLEED_RE.search(label):
        return False

    if len(label) > 200:
        return False

    words = label.split()
    if len(words) > 20:
        return False

    digit_count = sum(c.isdigit() for c in label)
    alpha_count = sum(c.isalpha() for c in label)
    if alpha_count == 0:
        return False
    if digit_count / max(alpha_count, 1) > 2.0:
        return False

    if field == "address":
        if len(label) < 10:
            return False
        if "," not in label and len(words) < 3:
            return False

    if field == "origin":
        if len(label) < 3:
            return False

    return True


def _normalize_label(label: str) -> str:
    """Clean up label text."""
    label = unicodedata.normalize("NFC", label.strip())
    label = re.sub(r"\s+", " ", label)
    label = label.strip(" ,;.")
    return label


def _collect_chars(labels: list[str]) -> list[str]:
    """Build character dictionary from all labels."""
    chars: set[str] = set()
    for label in labels:
        chars.update(label)
    chars.discard("\n")
    chars.discard("\r")
    chars.discard("\t")
    return sorted(chars)


def _load_finetune_annotations(
    path: Path, target_fields: set[str],
) -> list[tuple[str, str, str]]:
    """Load annotations, filter to target fields, validate."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts

            field = None
            for fld in target_fields:
                if f"/{fld}/" in img_path:
                    field = fld
                    break
            if field is None:
                continue

            label = _normalize_label(label)
            if not _is_valid_label(label, field):
                continue

            full_img_path = ROOT / img_path
            if not full_img_path.exists():
                continue

            results.append((img_path, label, field))
    return results


def _load_synthetic_labels(path: Path) -> list[tuple[str, str]]:
    """Load synthetic label file."""
    results = []
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            img_path, label = parts
            label = _normalize_label(label)
            if len(label) < 3:
                continue
            full_path = ROOT / img_path
            if not full_path.exists():
                continue
            results.append((img_path, label))
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_fields = {"address", "origin"}

    print("Loading finetune train annotations...")
    train_data = _load_finetune_annotations(
        FINETUNE_DIR / "train_annotation.txt", target_fields,
    )
    print(f"  Valid address/origin train samples: {len(train_data)}")

    print("Loading finetune val annotations...")
    val_data = _load_finetune_annotations(
        FINETUNE_DIR / "val_annotation.txt", target_fields,
    )
    print(f"  Valid address/origin val samples: {len(val_data)}")

    print("Loading synthetic data...")
    synthetic = _load_synthetic_labels(SYNTHETIC_LABEL)
    print(f"  Valid synthetic samples: {len(synthetic)}")

    train_entries = [(p, l) for p, l, _ in train_data]
    train_entries.extend(synthetic)

    val_entries = [(p, l) for p, l, _ in val_data]

    if len(val_entries) < 50:
        import random
        random.seed(42)
        random.shuffle(train_entries)
        split_idx = max(50, int(len(train_entries) * 0.05))
        val_entries.extend(train_entries[:split_idx])
        train_entries = train_entries[split_idx:]

    all_labels = [l for _, l in train_entries] + [l for _, l in val_entries]
    chars = _collect_chars(all_labels)

    train_path = OUTPUT_DIR / "train_label.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        for img_path, label in train_entries:
            f.write(f"{img_path}\t{label}\n")

    val_path = OUTPUT_DIR / "val_label.txt"
    with open(val_path, "w", encoding="utf-8") as f:
        for img_path, label in val_entries:
            f.write(f"{img_path}\t{label}\n")

    dict_path = OUTPUT_DIR / "dict.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        for ch in chars:
            f.write(f"{ch}\n")

    field_counts = Counter(f for _, _, f in train_data)
    label_lengths = [len(l) for _, l in train_entries]

    stats = {
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "real_address": field_counts.get("address", 0),
        "real_origin": field_counts.get("origin", 0),
        "synthetic_count": len(synthetic),
        "char_count": len(chars),
        "avg_label_length": round(sum(label_lengths) / max(len(label_lengths), 1), 1),
        "max_label_length": max(label_lengths) if label_lengths else 0,
    }
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Dataset prepared at: {OUTPUT_DIR}")
    print(f"  Train: {stats['train_count']} samples")
    print(f"    - Real address: {stats['real_address']}")
    print(f"    - Real origin: {stats['real_origin']}")
    print(f"    - Synthetic: {stats['synthetic_count']}")
    print(f"  Val: {stats['val_count']} samples")
    print(f"  Characters: {stats['char_count']}")
    print(f"  Avg label length: {stats['avg_label_length']}")
    print(f"  Max label length: {stats['max_label_length']}")


if __name__ == "__main__":
    main()
