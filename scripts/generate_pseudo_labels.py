"""Generate OCR pseudo-labels from cropped field images.

Default workflow:
python scripts/generate_pseudo_labels.py

This reads:
- manifest from data/processed/ocr/manifest.jsonl

This writes:
- pseudo-labels to data/processed/ocr/pseudo_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.ocr.ensemble import select_best_ocr_result
from src.ocr.hybrid_line_pick import run_hybrid_field_ocr
from src.ocr.paddleocr_adapter import PaddleOCRRecognizer
from src.ocr.utils import canonicalize_field_name, empty_ocr_result, looks_suspicious_for_field
from src.ocr.vietocr_adapter import VietOCRRecognizer

LOGGER = logging.getLogger(__name__)
HYBRID_FIELDS = {"place_of_origin", "place_of_residence"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VietOCR and PaddleOCR to generate pseudo-labels.")
    parser.add_argument("--manifest", default="data/processed/ocr/manifest.jsonl")
    parser.add_argument("--output", default="data/processed/ocr/pseudo_labels.jsonl")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--vietocr-config", default="vgg_transformer")
    parser.add_argument("--vietocr-device", default="auto")
    parser.add_argument("--paddle-lang", default="vi")
    parser.add_argument("--paddle-device", default="auto")
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use-hybrid-long-text",
        action="store_true",
        help="Optionally refine origin/address with hybrid line-picking. Disabled by default because it can truncate valid text.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def review_bucket(confidence: float) -> str:
    if confidence >= 0.9:
        return "accept"
    if confidence >= 0.5:
        return "review"
    return "reject"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def to_portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _pick_safer_result(field_name: str | None, baseline, hybrid):
    canonical_field = canonicalize_field_name(field_name)
    if hybrid is None or not hybrid.text.strip():
        return baseline
    if not baseline.text.strip():
        return hybrid

    baseline_bad = looks_suspicious_for_field(baseline.text, canonical_field)
    hybrid_bad = looks_suspicious_for_field(hybrid.text, canonical_field)
    baseline_score = float(baseline.confidence or 0.0)
    hybrid_score = float(hybrid.confidence or 0.0)
    baseline_len = len(baseline.text.strip())
    hybrid_len = len(hybrid.text.strip())

    if baseline_bad and not hybrid_bad:
        return hybrid
    if hybrid_bad and not baseline_bad:
        return baseline

    # Reject aggressive truncation unless confidence improves materially.
    if hybrid_len < max(4, int(round(baseline_len * 0.60))) and hybrid_score < baseline_score + 0.10:
        return baseline

    if canonical_field in HYBRID_FIELDS:
        if hybrid_score > baseline_score + 0.06 and hybrid_len >= max(4, int(round(baseline_len * 0.70))):
            return hybrid
        if hybrid_len > baseline_len + 6 and hybrid_score >= baseline_score - 0.02:
            return hybrid
        return baseline

    return hybrid if hybrid_score > baseline_score + 0.05 else baseline


def run_batch(
    manifest_rows: list[dict[str, Any]],
    *,
    project_root: Path,
    vietocr: VietOCRRecognizer,
    paddleocr: PaddleOCRRecognizer,
    use_hybrid_long_text: bool = False,
) -> list[dict[str, Any]]:
    from src.ocr.cropping import imread_unicode

    output_rows: list[dict[str, Any]] = []
    progress = tqdm(manifest_rows, desc="pseudo-labels", unit="crop")
    for index, row in enumerate(progress, start=1):
        crop_path = resolve_path(str(row["crop_path"]), project_root)
        image = imread_unicode(crop_path)
        if image is None:
            LOGGER.warning("Could not read crop %s", crop_path)
            viet_result = empty_ocr_result("vietocr")
            paddle_result = empty_ocr_result("paddleocr")
        else:
            field_name = str(row.get("field_name") or "")
            viet_result = vietocr.recognize(image, field_name=field_name)
            paddle_result = paddleocr.recognize(image, field_name=field_name)
            if use_hybrid_long_text and canonicalize_field_name(field_name) in HYBRID_FIELDS:
                hybrid_viet_result, hybrid_paddle_result = run_hybrid_field_ocr(
                    image=image,
                    field_name=field_name,
                    paddle_adapter=paddleocr,
                    viet_adapter=vietocr,
                )
                viet_result = _pick_safer_result(field_name, viet_result, hybrid_viet_result)
                paddle_result = _pick_safer_result(field_name, paddle_result, hybrid_paddle_result)

        best_result = select_best_ocr_result(row.get("field_name"), viet_result, paddle_result)
        output_rows.append(
            {
                "crop_path": to_portable_path(crop_path, project_root),
                "field_name": row.get("field_name"),
                "image_path": row.get("image_path"),
                "image_id": row.get("image_id"),
                "split": row.get("split"),
                "annotation_id": row.get("annotation_id"),
                "ground_truth_text": row.get("ground_truth_text", ""),
                "text_vietocr": viet_result.text,
                "conf_vietocr": round(float(viet_result.confidence), 4),
                "text_paddle": paddle_result.text,
                "text_paddleocr": paddle_result.text,
                "conf_paddle": round(float(paddle_result.confidence), 4),
                "conf_paddleocr": round(float(paddle_result.confidence), 4),
                "best_text": best_result.text,
                "best_conf": round(float(best_result.confidence), 4),
                "best_engine": best_result.engine,
                "best_source": best_result.engine,
                "needs_review": bool(best_result.needs_review),
                "review_bucket": review_bucket(float(best_result.confidence)),
            }
        )

        progress.set_postfix(
            field=row.get("field_name"),
            bucket=output_rows[-1]["review_bucket"],
        )

    return output_rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    started_at = perf_counter()
    project_root = Path(args.project_root).resolve()
    manifest_path = resolve_path(args.manifest, project_root)
    output_path = resolve_path(args.output, project_root)

    manifest_rows = read_jsonl(manifest_path)
    if args.fields:
        field_filter = {field.strip() for field in args.fields}
        manifest_rows = [row for row in manifest_rows if row.get("field_name") in field_filter]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    if not manifest_rows:
        raise ValueError("Manifest does not contain any crop rows to process.")

    LOGGER.info("Initializing OCR recognizers")
    vietocr = VietOCRRecognizer(config_name=args.vietocr_config, device=args.vietocr_device)
    paddleocr = PaddleOCRRecognizer(language=args.paddle_lang, device=args.paddle_device)
    rows = run_batch(
        manifest_rows,
        project_root=project_root,
        vietocr=vietocr,
        paddleocr=paddleocr,
        use_hybrid_long_text=args.use_hybrid_long_text,
    )

    write_jsonl(output_path, rows)
    LOGGER.info("Pseudo-labels written to %s", output_path)
    LOGGER.info("Generated %d rows in %.2fs", len(rows), perf_counter() - started_at)


if __name__ == "__main__":
    main()
