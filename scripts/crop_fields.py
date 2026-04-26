"""Crop OCR field images from COCO annotations.

Default workflow:
python scripts/crop_fields.py

This writes:
- crops to data/processed/ocr/field_crops/
- manifest to data/processed/ocr/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
from tqdm import tqdm

from src.ocr.cropping import bbox_xywh_to_xyxy, imread_unicode, imwrite_unicode, prepare_card_for_ocr, project_bbox

LOGGER = logging.getLogger(__name__)
TARGET_FIELDS = ("id", "name", "birth", "origin", "address")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop OCR-ready field images from COCO annotations.")
    parser.add_argument("--dataset-root", default="data/cccd.v1i.coco")
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--annotations", nargs="*", default=None, help="Optional explicit COCO JSON files.")
    parser.add_argument("--output-dir", default="data/processed/ocr/field_crops")
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--padding-ratio", type=float, default=0.04)
    parser.add_argument("--rectification-padding-ratio", type=float, default=0.05)
    parser.add_argument("--disable-rectify", action="store_true")
    parser.add_argument("--disable-enhance", action="store_true")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--fields", nargs="*", default=list(TARGET_FIELDS))
    parser.add_argument("--min-width", type=int, default=12)
    parser.add_argument("--min-height", type=int, default=12)
    parser.add_argument("--limit-crops", type=int, default=None, help="Stop early after writing this many crops.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def infer_annotation_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.annotations:
        return [(Path(path).stem, Path(path)) for path in args.annotations]

    splits_dir = Path(args.splits_dir)
    files: list[tuple[str, Path]] = []
    for split_name in args.splits:
        candidate = splits_dir / f"{split_name}.json"
        if candidate.exists():
            files.append((split_name, candidate))
    if not files:
        raise FileNotFoundError("No annotation files found. Use --annotations or provide valid files in --splits-dir.")
    return files


def build_image_index(dataset_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in dataset_root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        index.setdefault(path.name, path)
    return index


def clamp_bbox(xyxy: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = xyxy.tolist()
    clamped = np.array(
        [
            np.clip(x1, 0.0, max(0.0, float(width - 1))),
            np.clip(y1, 0.0, max(0.0, float(height - 1))),
            np.clip(x2, 1.0, float(width)),
            np.clip(y2, 1.0, float(height)),
        ],
        dtype=np.float32,
    )
    if clamped[2] <= clamped[0]:
        clamped[2] = min(float(width), clamped[0] + 1.0)
    if clamped[3] <= clamped[1]:
        clamped[3] = min(float(height), clamped[1] + 1.0)
    return clamped


def add_padding(xyxy: np.ndarray, image_shape: tuple[int, ...], padding_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.tolist()
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    padded = np.array(
        [
            x1 - width * padding_ratio,
            y1 - height * padding_ratio,
            x2 + width * padding_ratio,
            y2 + height * padding_ratio,
        ],
        dtype=np.float32,
    )
    return clamp_bbox(padded, image_shape)


def select_card_bbox(annotations: list[dict[str, Any]], category_lookup: dict[int, str]) -> dict[str, Any] | None:
    candidates = [ann for ann in annotations if category_lookup.get(ann["category_id"]) == "card"]
    if not candidates:
        return None
    return max(candidates, key=lambda ann: float(ann["bbox"][2] * ann["bbox"][3]))


def crop_split(
    split_name: str,
    annotation_path: Path,
    image_index: dict[str, Path],
    output_dir: Path,
    project_root: Path,
    *,
    field_names: set[str],
    padding_ratio: float,
    rectify_cards: bool,
    enhance_cards: bool,
    rectification_padding_ratio: float,
    min_width: int,
    min_height: int,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    coco = read_json(annotation_path)
    category_lookup = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    image_lookup = {int(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    manifest_rows: list[dict[str, Any]] = []
    image_items = list(image_lookup.items())
    progress = tqdm(image_items, desc=f"crop:{split_name}", unit="image")
    for image_id, image_info in progress:
        if max_rows is not None and len(manifest_rows) >= max_rows:
            break
        image_path = image_index.get(str(image_info["file_name"]))
        if image_path is None:
            LOGGER.warning("Missing image file for %s", image_info["file_name"])
            continue

        image = imread_unicode(image_path)
        if image is None:
            LOGGER.warning("Could not read image %s", image_path)
            continue

        image_annotations = annotations_by_image.get(image_id, [])
        card_annotation = select_card_bbox(image_annotations, category_lookup) if rectify_cards else None
        prepared_card = prepare_card_for_ocr(
            image=image,
            card_bbox=card_annotation["bbox"] if card_annotation is not None else None,
            enhance=enhance_cards,
            rectification_padding_ratio=rectification_padding_ratio,
        )

        for annotation in image_annotations:
            field_name = category_lookup.get(int(annotation["category_id"]))
            if field_name not in field_names:
                continue

            source_bbox = add_padding(
                bbox_xywh_to_xyxy(annotation["bbox"]),
                image.shape,
                padding_ratio=padding_ratio,
            )
            projected_bbox = project_bbox(
                source_bbox,
                prepared_card.perspective_matrix,
                prepared_card.crop_source_image.shape,
            )
            crop_width = int(round(projected_bbox[2] - projected_bbox[0]))
            crop_height = int(round(projected_bbox[3] - projected_bbox[1]))
            if crop_width < min_width or crop_height < min_height:
                continue

            crop = prepared_card.crop_source_image[
                int(round(projected_bbox[1])) : int(round(projected_bbox[3])),
                int(round(projected_bbox[0])) : int(round(projected_bbox[2])),
            ]
            if crop.size == 0:
                continue

            destination_dir = output_dir / field_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{Path(str(image_info['file_name'])).stem}_{annotation['id']}.jpg"
            crop_path = destination_dir / crop_name
            if not imwrite_unicode(crop_path, crop):
                LOGGER.warning("Failed to write crop %s", crop_path)
                continue

            manifest_rows.append(
                {
                    "crop_path": to_portable_path(crop_path, project_root),
                    "image_path": to_portable_path(image_path, project_root),
                    "field_name": field_name,
                    "image_id": image_id,
                    "source_image_id": image_id,
                    "bbox": [float(value) for value in projected_bbox.tolist()],
                    "source_bbox": [float(value) for value in source_bbox.tolist()],
                    "split": split_name,
                    "annotation_id": int(annotation["id"]),
                    "rectification_method": prepared_card.rectification_method,
                    "image_quality_score": float(prepared_card.quality_score),
                }
            )
            if max_rows is not None and len(manifest_rows) >= max_rows:
                break
        progress.set_postfix(crops=len(manifest_rows))

    return manifest_rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    started_at = perf_counter()
    project_root = Path.cwd().resolve()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    manifest_output = Path(args.manifest_output) if args.manifest_output else output_dir / "manifest.jsonl"

    annotation_files = infer_annotation_files(args)
    image_index = build_image_index(dataset_root)
    if not image_index:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    field_names = {field.strip() for field in args.fields}
    all_rows: list[dict[str, Any]] = []
    for split_name, annotation_path in annotation_files:
        LOGGER.info("Cropping split '%s' from %s", split_name, annotation_path)
        split_started_at = perf_counter()
        remaining_limit = None if args.limit_crops is None else max(0, args.limit_crops - len(all_rows))
        if remaining_limit == 0:
            break
        rows = crop_split(
            split_name=split_name,
            annotation_path=annotation_path,
            image_index=image_index,
            output_dir=output_dir,
            project_root=project_root,
            field_names=field_names,
            padding_ratio=args.padding_ratio,
            rectify_cards=not args.disable_rectify,
            enhance_cards=not args.disable_enhance,
            rectification_padding_ratio=args.rectification_padding_ratio,
            min_width=args.min_width,
            min_height=args.min_height,
            max_rows=remaining_limit,
        )
        all_rows.extend(rows)
        LOGGER.info("Finished split '%s' with %d crops in %.2fs", split_name, len(rows), perf_counter() - split_started_at)
        if args.limit_crops is not None and len(all_rows) >= args.limit_crops:
            LOGGER.info("Reached --limit-crops=%d; stopping early.", args.limit_crops)
            break

    write_jsonl(manifest_output, all_rows)
    LOGGER.info("Saved %d crops to %s", len(all_rows), output_dir)
    LOGGER.info("Manifest written to %s", manifest_output)
    LOGGER.info("Total runtime: %.2fs", perf_counter() - started_at)


if __name__ == "__main__":
    main()
