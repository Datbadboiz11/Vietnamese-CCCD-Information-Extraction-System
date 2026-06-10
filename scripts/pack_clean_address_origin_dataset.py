from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack the clean address/origin OCR dataset for Colab.")
    parser.add_argument("--input-dir", default="data/processed/ocr_address_origin_clean")
    parser.add_argument("--output", default="ocr_address_origin_clean.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = PROJECT_ROOT / args.input_dir
    output_path = PROJECT_ROOT / args.output

    required = ["train_label.txt", "val_label.txt", "test_label.txt", "dict.txt", "dataset_meta.json"]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {input_dir}: {missing}")

    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(input_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(input_dir).as_posix())

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
