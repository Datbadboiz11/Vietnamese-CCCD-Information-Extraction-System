from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
from tqdm import tqdm

from src.ocr.ensemble import ensemble_predictions
from src.ocr.paddleocr_adapter import PaddleOCRAdapter
from src.ocr.utils import FIELD_NAME_MAP, TARGET_FIELDS, review_bucket
from src.ocr.vietocr_adapter import VietOCRAdapter


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


def write_jsonl_line(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def to_portable_path(path: Path, project_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
        return relative.as_posix()
    except ValueError:
        return path.as_posix()


def disable_broken_proxy_env() -> list[str]:
    disabled: list[str] = []
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key, "")
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(key, None)
            disabled.append(key)
    return disabled


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def configure_local_model_cache(project_root: Path) -> Path:
    cache_root = project_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "torch").mkdir(parents=True, exist_ok=True)
    (cache_root / "paddle").mkdir(parents=True, exist_ok=True)
    (cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
    home_root = cache_root / "home"
    home_root.mkdir(parents=True, exist_ok=True)
    (home_root / ".cache" / "paddle" / "dataset").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str((cache_root / "torch").resolve()))
    os.environ.setdefault("PADDLE_HOME", str((cache_root / "paddle").resolve()))
    os.environ.setdefault("HF_HOME", str((cache_root / "huggingface").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
    resolved_home = str(home_root.resolve())
    resolved_home_path = home_root.resolve()
    os.environ["HOME"] = resolved_home
    os.environ["USERPROFILE"] = resolved_home
    os.environ["HOMEDRIVE"] = resolved_home_path.drive
    os.environ["HOMEPATH"] = str(resolved_home_path).replace(resolved_home_path.drive, "", 1)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return cache_root


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def get_crop_path(row: dict, project_root: Path) -> Path:
    crop_key = row.get("crop_path") or row.get("path")
    if not crop_key:
        raise KeyError("Manifest row is missing both `crop_path` and `path`.")
    return resolve_path(crop_key, project_root)


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def build_record(row: dict, project_root: Path, viet_adapter: VietOCRAdapter, paddle_adapter: PaddleOCRAdapter) -> dict:
    field_name = row.get("field_name") or FIELD_NAME_MAP.get(row.get("class", ""), row.get("class"))
    crop_path = get_crop_path(row, project_root)
    image = imread_unicode(crop_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read crop image: {crop_path}")

    viet_result = viet_adapter.predict(image, field_name)
    paddle_result = paddle_adapter.predict(image, field_name)
    best_result = ensemble_predictions(field_name, viet_result, paddle_result)
    best_bucket = review_bucket(best_result.confidence)

    return {
        "crop_path": to_portable_path(crop_path, project_root),
        "field_name": field_name,
        "class": row.get("class"),
        "split": row.get("split"),
        "source_image": row.get("source_image"),
        "ann_id": row.get("ann_id"),
        "text_vietocr": viet_result.text,
        "conf_vietocr": viet_result.confidence,
        "text_paddle": paddle_result.text,
        "conf_paddle": paddle_result.confidence,
        "best_text": best_result.text,
        "best_conf": best_result.confidence,
        "best_source": best_result.source,
        "best_normalized_text": best_result.normalized_text,
        "review_bucket": best_bucket,
        "needs_review": best_result.needs_review,
        "ground_truth_text": row.get("ground_truth_text", ""),
        "candidates": {
            "vietocr": asdict(viet_result),
            "paddleocr": asdict(paddle_result),
        },
    }


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from cropped field images.")
    parser.add_argument("--manifest", default="data/interim/cropped_fields/manifest.jsonl")
    parser.add_argument("--output-dir", default="data/processed/ocr")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--vietocr-config", default="vgg_transformer")
    parser.add_argument("--paddle-lang", default="vi")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--write-buckets",
        action="store_true",
        help="Also export accept/review/reject JSONL files. By default only pseudo_labels.jsonl is written.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = resolve_path(args.manifest, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    rows = read_jsonl(manifest_path)
    rows = [row for row in rows if (row.get("field_name") or FIELD_NAME_MAP.get(row.get("class", ""), "")) in TARGET_FIELDS]
    if args.splits:
        split_set = set(args.splits)
        rows = [row for row in rows if row.get("split") in split_set]
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No target OCR rows found in manifest.")

    missing_paths: list[str] = []
    for row in rows[: min(len(rows), 100)]:
        crop_path = get_crop_path(row, project_root)
        if not crop_path.exists():
            missing_paths.append(str(crop_path))
    if missing_paths:
        sample = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            "Manifest exists but crop image files are missing.\n"
            "Re-run scripts/crop_fields.py to materialize crop images, then run pseudo-label generation again.\n"
            f"Example missing files:\n{sample}"
        )

    disabled_proxy_keys = disable_broken_proxy_env()
    if disabled_proxy_keys:
        print(f"Disabled broken proxy env vars for OCR downloads: {', '.join(disabled_proxy_keys)}")
    configure_local_model_cache(project_root)
    print("Using project-local model cache for OCR downloads.")

    viet_adapter = VietOCRAdapter(config_name=args.vietocr_config)
    paddle_adapter = PaddleOCRAdapter(language=args.paddle_lang)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = len(rows)
    print(f"Processing {total_rows} crop(s) for pseudo-label generation...")

    counts = {"accept": 0, "review": 0, "reject": 0}
    pseudo_path = output_dir / "pseudo_labels.jsonl"
    accept_path = output_dir / "accept.jsonl"
    review_path = output_dir / "review.jsonl"
    reject_path = output_dir / "reject.jsonl"

    if not args.write_buckets:
        remove_if_exists(accept_path)
        remove_if_exists(review_path)
        remove_if_exists(reject_path)

    with ExitStack() as stack:
        pseudo_handle = stack.enter_context(pseudo_path.open("w", encoding="utf-8"))
        bucket_handles: dict[str, object] = {}
        if args.write_buckets:
            bucket_handles = {
                "accept": stack.enter_context(accept_path.open("w", encoding="utf-8")),
                "review": stack.enter_context(review_path.open("w", encoding="utf-8")),
                "reject": stack.enter_context(reject_path.open("w", encoding="utf-8")),
            }

        for index, row in enumerate(
            tqdm(rows, total=total_rows, desc="Pseudo-labels", unit="crop"),
            start=1,
        ):
            record = build_record(row, project_root, viet_adapter, paddle_adapter)
            write_jsonl_line(pseudo_handle, record)

            bucket = record["review_bucket"]
            counts[bucket] += 1
            if args.write_buckets:
                write_jsonl_line(bucket_handles[bucket], record)

            if index % 50 == 0:
                pseudo_handle.flush()
                for handle in bucket_handles.values():
                    handle.flush()

    print(f"Generated {total_rows} pseudo-label rows in {output_dir}")
    print(f"Buckets summary: accept={counts['accept']}, review={counts['review']}, reject={counts['reject']}")
    if args.write_buckets:
        print("Wrote: pseudo_labels.jsonl + accept/review/reject JSONL files")
    else:
        print("Wrote: pseudo_labels.jsonl")


if __name__ == "__main__":
    main()
