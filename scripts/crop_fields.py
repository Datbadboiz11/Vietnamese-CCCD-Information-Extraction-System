from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from src.ocr.utils import FIELD_NAME_MAP


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


def build_image_index(dataset_root: Path, image_roots: list[Path] | None = None) -> dict[str, Path]:
    candidates = image_roots or []
    if not candidates:
        for name in ("train", "valid", "test"):
            candidate = dataset_root / name
            if candidate.exists():
                candidates.append(candidate)
        if dataset_root.exists():
            candidates.append(dataset_root)

    index: dict[str, Path] = {}
    for root in candidates:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            index.setdefault(path.name, path)
    return index


def crop_from_split(
    split_json: Path,
    split_name: str,
    image_index: dict[str, Path],
    output_dir: Path,
    padding: int,
    min_size: int,
    allowed_classes: set[str],
) -> list[dict]:
    coco = read_json(split_json)
    category_lookup = {category["id"]: category["name"] for category in coco["categories"]}
    image_lookup = {image["id"]: image for image in coco["images"]}
    rows: list[dict] = []

    for ann in coco["annotations"]:
        class_name = category_lookup.get(ann["category_id"])
        if class_name not in allowed_classes:
            continue

        image_info = image_lookup[ann["image_id"]]
        image_path = image_index.get(image_info["file_name"])
        if image_path is None:
            continue

        image = imread_unicode(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        x, y, w, h = ann["bbox"]
        x1 = max(0, int(x) - padding)
        y1 = max(0, int(y) - padding)
        x2 = min(width, int(x + w) + padding)
        y2 = min(height, int(y + h) + padding)
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_info["file_name"]).stem
        file_name = f"{stem}_{ann['id']}.jpg"
        crop_path = target_dir / file_name
        if not imwrite_unicode(crop_path, crop):
            continue

        field_name = FIELD_NAME_MAP.get(class_name, class_name)
        rows.append(
            {
                "path": str(crop_path.as_posix()),
                "crop_path": str(crop_path.as_posix()),
                "class": class_name,
                "field_name": field_name,
                "split": split_name,
                "source_image": image_info["file_name"],
                "ann_id": ann["id"],
                "bbox": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop field regions from COCO split files.")
    parser.add_argument("--dataset-root", default="data/cccd.v1i.coco")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--output-dir", default="data/interim/cropped_fields")
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=["id", "name", "birth", "origin", "address", "title"],
        help="COCO class names to crop.",
    )
    parser.add_argument(
        "--image-roots",
        nargs="*",
        default=None,
        help="Optional explicit image roots. If omitted, the script searches inside dataset-root.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    image_roots = [Path(path) for path in args.image_roots] if args.image_roots else None
    image_index = build_image_index(dataset_root, image_roots=image_roots)
    if not image_index:
        raise FileNotFoundError(
            "Could not find source images. Provide --dataset-root or --image-roots that contain the raw images."
        )

    all_rows: list[dict] = []
    for split_name in ("train", "val", "test"):
        split_json = splits_dir / f"{split_name}.json"
        if not split_json.exists():
            continue
        all_rows.extend(
            crop_from_split(
                split_json=split_json,
                split_name=split_name,
                image_index=image_index,
                output_dir=output_dir,
                padding=args.padding,
                min_size=args.min_size,
                allowed_classes=set(args.classes),
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, all_rows)

    class_counts = Counter(row["class"] for row in all_rows)
    print(f"Saved {len(all_rows)} crops to {output_dir}")
    print(f"Manifest: {manifest_path}")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
