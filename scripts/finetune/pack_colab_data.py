"""
Pack OCR fine-tuning data into a zip for Google Colab / remote training.

This script is generic and works for both the legacy VietOCR dataset layout
and the newer PaddleOCR recognition layout.

It reads annotation files in ``path<TAB>text`` format, copies every referenced
image into a staging directory while preserving relative paths, then writes a
zip archive that can be uploaded to Colab.

Included files when present:
  - train_label.txt / train_annotation.txt
  - val_label.txt / val_annotation.txt
  - dict.txt / vi_dict.txt
  - dataset_meta.json
  - README.md

Usage:
    python scripts/finetune/pack_colab_data.py
    python scripts/finetune/pack_colab_data.py --input-dir data/processed/ocr/ppocrv5_rec
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ocr" / "ppocrv5_rec"
LEGACY_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ocr" / "finetune"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "ppocrv5_rec_data"

ANNOTATION_CANDIDATES = (
    "train_label.txt",
    "val_label.txt",
    "train_annotation.txt",
    "val_annotation.txt",
)
EXTRA_FILES = (
    "dict.txt",
    "vi_dict.txt",
    "dataset_meta.json",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack OCR fine-tuning data into a zip archive.")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Dataset directory containing annotation files and referenced images.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(DEFAULT_OUTPUT_PREFIX),
        help="Output archive prefix without .zip suffix.",
    )
    return parser.parse_args()


def resolve_input_dir(input_dir_arg: str | None) -> Path:
    if input_dir_arg:
        path = Path(input_dir_arg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if DEFAULT_INPUT_DIR.exists():
        return DEFAULT_INPUT_DIR
    return LEGACY_INPUT_DIR


def collect_annotation_files(input_dir: Path) -> list[Path]:
    files = [input_dir / name for name in ANNOTATION_CANDIDATES if (input_dir / name).exists()]
    if not files:
        raise FileNotFoundError(
            f"No annotation files found in {input_dir}. "
            f"Expected one of: {', '.join(ANNOTATION_CANDIDATES)}"
        )
    return files


def copy_annotation_assets(ann_path: Path, input_dir: Path, staging_dir: Path) -> tuple[list[str], int]:
    copied = 0
    output_lines: list[str] = []

    lines: list[str] | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1258"):
        try:
            lines = ann_path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {ann_path}")

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue

        rel_path, text = parts
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            src = input_dir / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Referenced image not found: {rel_path}")

        normalized_rel = Path(rel_path).as_posix()
        dst = staging_dir / normalized_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

        output_lines.append(f"{normalized_rel}\t{text}\n")

    return output_lines, copied


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    annotation_files = collect_annotation_files(input_dir)
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = PROJECT_ROOT / output_prefix

    staging = PROJECT_ROOT / "_colab_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    total_images_copied = 0
    total_rows = 0

    print(f"Packing dataset from: {input_dir}")
    for ann_path in annotation_files:
        lines, copied = copy_annotation_assets(ann_path, input_dir, staging)
        total_images_copied += copied
        total_rows += len(lines)
        with (staging / ann_path.name).open("w", encoding="utf-8") as handle:
            handle.writelines(lines)
        print(f"  {ann_path.name}: {len(lines)} rows, {copied} new images copied")

    for extra_name in EXTRA_FILES:
        src = input_dir / extra_name
        if src.exists():
            shutil.copy2(src, staging / extra_name)
            print(f"  Included {extra_name}")

    print(f"\nTotal annotation rows: {total_rows}")
    print(f"Total unique images copied: {total_images_copied}")

    print("\nCreating zip archive...")
    shutil.make_archive(str(output_prefix), "zip", staging)
    zip_path = output_prefix.with_suffix(".zip")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"Created: {zip_path} ({size_mb:.1f} MB)")

    shutil.rmtree(staging)
    print("Done.")


if __name__ == "__main__":
    main()
