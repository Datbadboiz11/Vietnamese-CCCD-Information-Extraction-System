"""
Text Line Detection for multi-line CCCD fields (Module 7).

Address and place-of-origin on Vietnamese CCCD cards contain 2 lines of text.
VietOCR is a single-line recognition model — feeding a multi-line crop directly
causes hallucination (English gibberish, repeated tokens, or MRZ digits).

This module detects individual text lines within a field crop so each line can
be recognised separately by VietOCR.

Applied to:  place_of_origin, place_of_residence
Skipped for: id_number, full_name, date_of_birth (always single-line)
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.ocr.cropping import clamp_bbox, crop_image_xyxy

LOGGER = logging.getLogger(__name__)

MULTI_LINE_FIELDS = {"place_of_origin", "place_of_residence", "origin", "address"}
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LONG_TEXT_REC_MODEL_DIR = os.path.join(
    _REPO_ROOT,
    "model",
    "ocr",
    "ppocrv5_cccd_address_origin",
    "latin_ppocrv5_cccd",
    "inference",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TextLine:
    """A single detected text line within a field crop."""

    crop: np.ndarray
    bbox_xyxy: np.ndarray
    line_index: int
    detection_confidence: float
    method: str
    det_text: str | None = None


@dataclass
class TextLineDetectionResult:
    """Result of text line detection on a field crop."""

    lines: list[TextLine]
    method: str
    num_lines_raw: int
    num_lines_filtered: int
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _is_mrz_text(text: str) -> bool:
    """Text that is mostly digits (MRZ / barcode contamination)."""
    if not text:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    if alnum == 0:
        return False
    digits = sum(ch.isdigit() for ch in text)
    return digits / alnum > 0.70 and digits >= 8


def _is_noise_box(
    bbox_xyxy: np.ndarray,
    image_shape: tuple[int, ...],
    min_height: int = 10,
) -> bool:
    """Box too small or geometrically implausible."""
    h = float(bbox_xyxy[3] - bbox_xyxy[1])
    w = float(bbox_xyxy[2] - bbox_xyxy[0])
    if h < min_height or w < 15:
        return True
    img_h = image_shape[0]
    if h > img_h * 0.90 and w < image_shape[1] * 0.25:
        return True
    return False


def _sort_reading_order(
    items: list[tuple[np.ndarray, float, str | None]],
) -> list[tuple[np.ndarray, float, str | None]]:
    """Top-to-bottom, then left-to-right."""
    return sorted(items, key=lambda t: (
        float((t[0][1] + t[0][3]) / 2.0),
        float(t[0][0]),
    ))


# ---------------------------------------------------------------------------
# PaddleOCR text detector
# ---------------------------------------------------------------------------

def _configure_headless_matplotlib() -> None:
    backend = (os.environ.get("MPLBACKEND") or "").strip().lower()
    if backend.startswith("module://matplotlib_inline") or not backend:
        os.environ["MPLBACKEND"] = "Agg"


class PaddleTextDetector:
    """PaddleOCR wrapper that returns per-line boxes (and optionally rec text)."""

    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._client: Any | None = None

    @staticmethod
    def _has_custom_long_text_model() -> bool:
        return os.path.isdir(_LONG_TEXT_REC_MODEL_DIR) and os.path.isfile(
            os.path.join(_LONG_TEXT_REC_MODEL_DIR, "inference.pdiparams")
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        _configure_headless_matplotlib()

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for text line detection. "
                "Install with: pip install paddleocr paddlepaddle"
            ) from exc

        kwargs: dict[str, Any] = {
            "lang": "vi",
            "enable_mkldnn": False,
            "enable_hpi": False,
        }

        sig: inspect.Signature | None = None
        try:
            sig = inspect.signature(PaddleOCR.__init__)
        except (TypeError, ValueError):
            pass

        if sig is None or "show_log" in (sig.parameters if sig else {}):
            kwargs["show_log"] = False

        for key, val in {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }.items():
            if sig is not None and key in sig.parameters:
                kwargs[key] = val

        if self._has_custom_long_text_model():
            if sig is None or "text_recognition_model_dir" in sig.parameters:
                kwargs["text_recognition_model_dir"] = _LONG_TEXT_REC_MODEL_DIR
            if sig is None or "text_recognition_model_name" in sig.parameters:
                kwargs["text_recognition_model_name"] = "latin_PP-OCRv5_mobile_rec"

        if self._device:
            kwargs["device"] = self._device

        try:
            self._client = PaddleOCR(**kwargs)
        except ValueError as exc:
            if "Unknown argument: show_log" not in str(exc):
                raise
            kwargs.pop("show_log", None)
            self._client = PaddleOCR(**kwargs)
        return self._client

    # ------------------------------------------------------------------ #

    def detect(
        self, image: np.ndarray, *, with_rec: bool = True,
    ) -> list[tuple[np.ndarray, float, str | None]]:
        """Return ``[(bbox_xyxy, confidence, text_or_None), ...]``."""
        client = self._get_client()
        results: list[tuple[np.ndarray, float, str | None]] = []

        try:
            if hasattr(client, "predict"):
                raw = self._run_predict(client, image, with_rec)
            else:
                raw = client.ocr(image, det=True, rec=with_rec, cls=False)
        except Exception as exc:
            LOGGER.warning("PaddleOCR detection failed: %s", exc)
            return []

        self._parse_raw(raw, results, with_rec)
        return results

    # -- internal parsers ------------------------------------------------- #

    @staticmethod
    def _run_predict(client: Any, image: np.ndarray, with_rec: bool) -> Any:
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(client.predict)
        except (TypeError, ValueError):
            sig = None
        for key, val in {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "return_word_box": False,
            "text_rec_score_thresh": 0.0,
        }.items():
            if sig is not None and key in sig.parameters:
                kwargs[key] = val
        return client.predict(image, **kwargs)

    def _parse_raw(
        self,
        node: Any,
        out: list[tuple[np.ndarray, float, str | None]],
        with_rec: bool,
    ) -> None:
        if node is None:
            return

        if hasattr(node, "json"):
            try:
                self._parse_raw(node.json, out, with_rec)
                return
            except Exception:
                pass

        if isinstance(node, dict):
            self._parse_dict(node, out, with_rec)
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                if item is None:
                    continue
                if isinstance(item, dict):
                    self._parse_dict(item, out, with_rec)
                elif isinstance(item, (list, tuple)):
                    self._parse_list_item(item, out, with_rec)
                elif hasattr(item, "json"):
                    self._parse_raw(item, out, with_rec)

    def _parse_dict(
        self,
        d: dict,
        out: list[tuple[np.ndarray, float, str | None]],
        with_rec: bool,
    ) -> None:
        if "res" in d:
            self._parse_raw(d["res"], out, with_rec)
            return

        box_key = next(
            (k for k in ("dt_polys", "dt_boxes", "polys", "boxes", "rec_polys") if k in d),
            None,
        )
        if box_key is not None:
            boxes = d.get(box_key) or []
            scores = d.get("dt_scores", d.get("rec_scores", []))
            texts = d.get("rec_texts", []) if with_rec else []
            for i, box in enumerate(boxes):
                xyxy = self._poly_to_xyxy(box)
                if xyxy is None:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.8
                text = str(texts[i]) if i < len(texts) else None
                out.append((xyxy, score, text))
            return

        if "rec_texts" in d:
            texts = d.get("rec_texts") or []
            scores = d.get("rec_scores") or []
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                out.append((
                    np.zeros(4, dtype=np.float32),
                    score,
                    str(text) if with_rec else None,
                ))

    def _parse_list_item(
        self,
        item: Any,
        out: list[tuple[np.ndarray, float, str | None]],
        with_rec: bool,
    ) -> None:
        if not isinstance(item, (list, tuple)):
            return
        if len(item) == 2:
            first, second = item
            if isinstance(second, (list, tuple)) and len(second) >= 2 and isinstance(second[0], str):
                xyxy = self._poly_to_xyxy(first)
                if xyxy is not None:
                    out.append((xyxy, float(second[1]), str(second[0]) if with_rec else None))
                return
            xyxy = self._poly_to_xyxy(first)
            if xyxy is not None:
                out.append((xyxy, 0.8, None))
                return
        xyxy = self._poly_to_xyxy(item)
        if xyxy is not None:
            out.append((xyxy, 0.8, None))
            return
        for sub in item:
            if isinstance(sub, (list, tuple, dict)):
                self._parse_raw(sub, out, with_rec)

    @staticmethod
    def _poly_to_xyxy(poly: Any) -> np.ndarray | None:
        try:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None
        if pts.shape[0] < 4:
            return None
        return np.array([
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fallback: horizontal projection profile
# ---------------------------------------------------------------------------

def _detect_lines_by_projection(
    image: np.ndarray, min_line_height: int = 10,
) -> list[np.ndarray]:
    """Split a horizontal-text image into line bboxes using row-projection."""
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    projection = np.sum(binary, axis=1).astype(np.float32)
    peak = float(np.max(projection)) if projection.size > 0 else 0.0
    threshold = max(w * 0.02, peak * 0.05) if peak > 0 else 1.0

    in_text = False
    start = 0
    boxes: list[np.ndarray] = []

    for y in range(h):
        if projection[y] > threshold:
            if not in_text:
                start = y
                in_text = True
        else:
            if in_text:
                if y - start >= min_line_height:
                    boxes.append(np.array([0, start, w, y], dtype=np.float32))
                in_text = False
    if in_text and h - start >= min_line_height:
        boxes.append(np.array([0, start, w, h], dtype=np.float32))

    return boxes


# ---------------------------------------------------------------------------
# Image preparation helpers
# ---------------------------------------------------------------------------

def _prepare_for_detection(image: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Pad + upscale a field crop for robust text detection.

    Returns ``(prepared_image, scale_factor, padding_size)``.
    """
    prepared = image.copy()
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    pad = max(8, int(round(min(prepared.shape[:2]) * 0.12)))
    prepared = cv2.copyMakeBorder(
        prepared, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE,
    )

    scale = max(1.0, 160 / max(1, prepared.shape[0]))
    if scale > 1.0:
        prepared = cv2.resize(
            prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    return prepared, scale, pad


def _map_box_to_original(
    box: np.ndarray, scale: float, pad: int, image_shape: tuple[int, ...],
) -> np.ndarray:
    """Reverse padding + scaling to get coordinates in the original crop."""
    mapped = np.array([
        float(box[0]) / scale - pad,
        float(box[1]) / scale - pad,
        float(box[2]) / scale - pad,
        float(box[3]) / scale - pad,
    ], dtype=np.float32)
    return clamp_bbox(mapped, image_shape)


def _crop_line(
    image: np.ndarray, bbox_xyxy: np.ndarray, pad_x: int = 6, pad_y: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    padded = np.array([
        max(0, float(bbox_xyxy[0]) - pad_x),
        max(0, float(bbox_xyxy[1]) - pad_y),
        min(w, float(bbox_xyxy[2]) + pad_x),
        min(h, float(bbox_xyxy[3]) + pad_y),
    ], dtype=np.float32)
    return crop_image_xyxy(image, padded), padded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_DETECTOR: PaddleTextDetector | None = None


def _get_default_detector() -> PaddleTextDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = PaddleTextDetector()
    return _DEFAULT_DETECTOR


def is_multi_line_field(field_name: str | None) -> bool:
    from src.ocr.utils import canonicalize_field_name
    return canonicalize_field_name(field_name) in MULTI_LINE_FIELDS


def detect_text_lines(
    field_crop: np.ndarray,
    field_name: str | None = None,
    *,
    detector: PaddleTextDetector | None = None,
    pad_x: int = 6,
    pad_y: int = 8,
    min_line_height: int = 10,
) -> TextLineDetectionResult:
    """Detect individual text lines in a field crop.

    For multi-line fields the image is split into per-line crops suitable for
    single-line recognition by VietOCR.  Single-line fields are returned as-is.
    """
    from src.ocr.utils import canonicalize_field_name

    canonical = canonicalize_field_name(field_name)
    warnings: list[str] = []

    # -- single-line fields: passthrough ---------------------------------- #
    if canonical not in MULTI_LINE_FIELDS:
        h, w = field_crop.shape[:2]
        return TextLineDetectionResult(
            lines=[TextLine(
                crop=field_crop,
                bbox_xyxy=np.array([0, 0, w, h], dtype=np.float32),
                line_index=0,
                detection_confidence=1.0,
                method="single_line_passthrough",
            )],
            method="single_line_passthrough",
            num_lines_raw=1,
            num_lines_filtered=1,
        )

    if field_crop is None or field_crop.size == 0:
        return TextLineDetectionResult(
            lines=[], method="empty_input",
            num_lines_raw=0, num_lines_filtered=0,
            warnings=["Empty field crop"],
        )

    # -- prepare image ---------------------------------------------------- #
    prepared, scale, pad = _prepare_for_detection(field_crop)

    # -- primary: PaddleOCR det (+rec for MRZ filter) --------------------- #
    method = "paddleocr_det"
    raw_items: list[tuple[np.ndarray, float, str | None]] = []

    try:
        det = detector or _get_default_detector()
        raw_items = det.detect(prepared, with_rec=True)
    except Exception as exc:
        LOGGER.warning("PaddleOCR det failed, trying projection: %s", exc)
        method = "projection"

    # map back to original crop coords
    mapped: list[tuple[np.ndarray, float, str | None]] = []
    for box, conf, text in raw_items:
        orig_box = _map_box_to_original(box, scale, pad, field_crop.shape)
        mapped.append((orig_box, conf, text))

    mapped = _sort_reading_order(mapped)
    num_raw = len(mapped)

    # -- filter noise + MRZ ----------------------------------------------- #
    filtered: list[tuple[np.ndarray, float, str | None]] = []
    for box, conf, text in mapped:
        if _is_noise_box(box, field_crop.shape, min_line_height):
            continue
        if text and _is_mrz_text(text):
            warnings.append(f"Filtered MRZ line: {text[:30]}")
            continue
        filtered.append((box, conf, text))

    # -- fallback: projection --------------------------------------------- #
    if not filtered:
        if method == "paddleocr_det":
            warnings.append("PaddleOCR det found no valid text lines, trying projection")
        method = "projection"
        for box in _detect_lines_by_projection(field_crop, min_line_height):
            filtered.append((box, 0.5, None))
        if filtered:
            num_raw = len(filtered)

    # -- final fallback: whole crop --------------------------------------- #
    if not filtered:
        warnings.append("No text lines detected — using whole crop (degraded)")
        h, w = field_crop.shape[:2]
        return TextLineDetectionResult(
            lines=[TextLine(
                crop=field_crop,
                bbox_xyxy=np.array([0, 0, w, h], dtype=np.float32),
                line_index=0,
                detection_confidence=0.3,
                method="whole_crop_fallback",
            )],
            method="whole_crop_fallback",
            num_lines_raw=num_raw,
            num_lines_filtered=0,
            warnings=warnings,
        )

    # -- crop each line --------------------------------------------------- #
    lines: list[TextLine] = []
    for idx, (box, conf, det_text) in enumerate(filtered):
        crop, adj_box = _crop_line(field_crop, box, pad_x, pad_y)
        if crop.size == 0:
            continue
        lines.append(TextLine(
            crop=crop,
            bbox_xyxy=adj_box,
            line_index=idx,
            detection_confidence=conf,
            method=method,
            det_text=det_text,
        ))

    if not lines:
        warnings.append("All crops empty after line splitting")
        h, w = field_crop.shape[:2]
        return TextLineDetectionResult(
            lines=[TextLine(
                crop=field_crop,
                bbox_xyxy=np.array([0, 0, w, h], dtype=np.float32),
                line_index=0,
                detection_confidence=0.3,
                method="whole_crop_fallback",
            )],
            method="whole_crop_fallback",
            num_lines_raw=num_raw,
            num_lines_filtered=len(filtered),
            warnings=warnings,
        )

    return TextLineDetectionResult(
        lines=lines,
        method=method,
        num_lines_raw=num_raw,
        num_lines_filtered=len(lines),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Label-line trimming for origin / address crops
# ---------------------------------------------------------------------------

_LABEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"p[lhi]?[lai]c[eo]?\s*[ao][fl1]", re.IGNORECASE),
    re.compile(r"place\s*[ao][fl1]", re.IGNORECASE),
    re.compile(r"piace\s", re.IGNORECASE),
    re.compile(r"p\w{0,5}c[eo]?\s+\w{0,4}f", re.IGNORECASE),
    re.compile(r"n[oơộ]i?\s*[dđ]?k?h?k?\s*th[uưừ]r?[oơờ]?ng\s*tr[uúụ]", re.IGNORECASE),
    re.compile(r"nguy[eêệ]n\s*qu[aáâ]n", re.IGNORECASE),
    re.compile(r"qu[eêế]\s*qu[aáâ]n", re.IGNORECASE),
    re.compile(r"^p[lh]?lac[eo]?\b", re.IGNORECASE),
    re.compile(r"residen", re.IGNORECASE),
    re.compile(r"origin", re.IGNORECASE),
    re.compile(r"ofonig", re.IGNORECASE),
    re.compile(r"dforig", re.IGNORECASE),
    re.compile(r"ofresid", re.IGNORECASE),
]


def _det_text_is_label(text: str | None) -> bool:
    """Check if PaddleOCR det_text looks like a field label."""
    if not text:
        return False
    import unicodedata

    folded = unicodedata.normalize("NFD", text.lower())
    folded = folded.replace("đ", "d").replace("Đ", "d")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    for pat in _LABEL_PATTERNS:
        if pat.search(folded):
            return True
    return False


def trim_label_from_crop(
    field_crop: np.ndarray,
    field_name: str | None,
    detector: "PaddleTextDetector | None" = None,
) -> np.ndarray:
    """Remove label line(s) from the top of an origin/address field crop.

    Uses PaddleOCR text detection to find lines whose ``det_text`` matches
    a known field-label pattern **and** sit in the upper 45 % of the crop.
    Everything above the bottom of the last such label line is trimmed.

    Returns the original crop unchanged when no label is found or the field
    is not origin/address.
    """
    from src.ocr.utils import canonicalize_field_name

    canonical = canonicalize_field_name(field_name)
    if canonical not in MULTI_LINE_FIELDS:
        return field_crop
    if field_crop is None or field_crop.size == 0:
        return field_crop

    h, w = field_crop.shape[:2]
    if h < 50:
        return field_crop

    det = detector or _get_default_detector()
    prepared, scale, pad = _prepare_for_detection(field_crop)
    try:
        raw_items = det.detect(prepared, with_rec=True)
    except Exception:
        return field_crop

    label_bottom_y = 0
    content_top_y = h
    for box, _conf, det_text in raw_items:
        orig = _map_box_to_original(box, scale, pad, field_crop.shape)
        center_y = float(orig[1] + orig[3]) / 2.0
        if center_y <= h * 0.45 and _det_text_is_label(det_text):
            label_bottom_y = max(label_bottom_y, int(orig[3]))
        elif not _det_text_is_label(det_text):
            content_top_y = min(content_top_y, int(orig[1]))

    if label_bottom_y <= 0:
        return field_crop

    pad_above = max(4, int(h * 0.04))
    if content_top_y < h:
        y_start = max(0, content_top_y - pad_above)
        y_start = max(y_start, label_bottom_y - pad_above)
    else:
        y_start = label_bottom_y
    y_start = min(y_start, int(h * 0.55))
    trimmed = field_crop[y_start:, :]
    if trimmed.size == 0:
        return field_crop

    LOGGER.debug(
        "Trimmed label area y=[0:%d] from %s crop (h=%d)", y_start, canonical, h,
    )
    return trimmed


__all__ = [
    "PaddleTextDetector",
    "TextLine",
    "TextLineDetectionResult",
    "detect_text_lines",
    "is_multi_line_field",
    "trim_label_from_crop",
]
