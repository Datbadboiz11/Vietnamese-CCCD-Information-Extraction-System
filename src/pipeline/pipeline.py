"""
src/pipeline — End-to-end CCCD extraction pipeline.

Flow:
    image_path → [Card Detection] → [Rectification] → [Enhancement]
               → [Field Detection] → [OCR per field] → [Parsing & Validation]
               → PipelineResult

Exports:
    CCCDPipeline    - class pipeline chính
    PipelineResult  - dataclass kết quả
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ocr.cropping import crop_image_xyxy, clamp_bbox, prepare_card_for_ocr
from src.ocr.ensemble import ensemble_recognize, select_best_ocr_result
from src.ocr.hybrid_line_pick import run_hybrid_field_ocr
from src.ocr.multiline_ocr import recognize_multiline_field
from src.ocr.paddleocr_adapter import PaddleOCRRecognizer
from src.ocr.text_detection import PaddleTextDetector, is_multi_line_field
from src.ocr.vietocr_adapter import VietOCRRecognizer
from src.parsing.validators import CCCDParser, ParsedInfo
from src.preprocessing.enhance import apply_clahe, compute_image_quality_score, enhance_field_crop
from src.preprocessing.orientation import auto_orient_for_ocr
from src.preprocessing.rectify import rectify_from_bbox

LOGGER = logging.getLogger(__name__)

# Class-index map cho field detector (theo thứ tự train của dataset Roboflow)
# classes: address, birth, card, id, name, origin, title
_FIELD_CLASS_NAMES: dict[int, str] = {
    0: "address",
    1: "birth",
    2: "card",
    3: "id",
    4: "name",
    5: "origin",
    6: "title",
}

# Class nào cần OCR (bỏ card và title)
_OCR_TARGET_CLASSES = {"address", "birth", "id", "name", "origin"}

# Map sang canonical field name (dùng bởi parser)
_CLASS_TO_FIELD = {
    "id": "id_number",
    "name": "full_name",
    "birth": "date_of_birth",
    "origin": "place_of_origin",
    "address": "place_of_residence",
}


def _ocr_sanity_score(cls_name: str, text: str) -> float:
    """Return a penalty multiplier (0.0–1.0) based on whether the OCR text
    is plausible for the detected field class.  A low score means the field
    detector likely misclassified the region."""
    if not text or not text.strip():
        return 0.0

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    if cls_name == "id":
        digits = "".join(c for c in text if c.isdigit())
        if len(digits) >= 11:
            return 1.0
        if len(digits) >= 9:
            return 0.7
        return 0.1

    if cls_name == "birth":
        import re
        if re.search(r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}", text):
            return 1.0
        digits = "".join(c for c in text if c.isdigit())
        if len(digits) >= 6:
            return 0.6
        return 0.1

    if cls_name == "name":
        if digit_ratio > 0.5:
            return 0.1
        return 1.0

    if cls_name in ("origin", "address"):
        if digit_ratio > 0.7 and len(text) > 8:
            return 0.2
        return 1.0

    return 1.0


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Kết quả đầu ra đầy đủ của một lần chạy pipeline."""

    # Detection
    card_detected: bool = False
    card_bbox: list[float] | None = None  # [x1, y1, x2, y2] trong ảnh gốc
    card_confidence: float = 0.0

    # Rectification
    rectified_image: np.ndarray | None = None
    rectification_method: str = "none"

    # Image quality
    image_quality_score: float = 0.0
    quality_metrics: dict[str, float] = field(default_factory=dict)

    # Enhancement diagnostics
    enhancement_tier: int = 0
    enhancement_tier_label: str = "none"
    enhancement_actions: list[str] = field(default_factory=list)

    # Field detection
    field_detections: list[dict[str, Any]] = field(default_factory=list)
    # {class_name, bbox_xyxy (trong card), confidence}

    # OCR results: {class_name: (text, confidence)}
    ocr_results: dict[str, tuple[str, float]] = field(default_factory=dict)
    field_debug: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Parsing
    parsed_info: ParsedInfo | None = None

    # Diagnostics
    processing_steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def _add_step(self, step: str, status: str, details: str) -> None:
        self.processing_steps.append({"step": step, "status": status, "details": details})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "card_detected": self.card_detected,
            "card_bbox": self.card_bbox,
            "card_confidence": self.card_confidence,
            "rectification_method": self.rectification_method,
            "image_quality_score": self.image_quality_score,
            "quality_metrics": self.quality_metrics,
            "enhancement_tier": self.enhancement_tier,
            "enhancement_tier_label": self.enhancement_tier_label,
            "enhancement_actions": self.enhancement_actions,
            "field_detections": self.field_detections,
            "ocr_results": {k: {"text": v[0], "confidence": v[1]} for k, v in self.ocr_results.items()},
            "field_debug": self.field_debug,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self.parsed_info is not None:
            out["parsed_info"] = self.parsed_info.to_dict()
        return out


# ---------------------------------------------------------------------------
# CCCDPipeline
# ---------------------------------------------------------------------------

class CCCDPipeline:
    """
    Pipeline end-to-end trích xuất thông tin CCCD.

    Usage::
        pipeline = CCCDPipeline(
            card_detector_path="model/card_detector/best.pt",
            field_detector_path="model/field_detector/best.pt",
        )
        result: PipelineResult = pipeline("path/to/cccd.jpg")
    """

    def __init__(
        self,
        card_detector_path: str = "model/card_detector/best.pt",
        field_detector_path: str = "model/field_detector/best.pt",
        device: str = "cpu",
        card_conf_threshold: float = 0.5,
        field_conf_threshold: float = 0.3,
        use_ensemble: bool = False,  # True = dùng cả VietOCR + PaddleOCR
        use_tta: bool = False,
        use_text_region_refinement: bool = True,
    ) -> None:
        self.card_detector_path = card_detector_path
        self.field_detector_path = field_detector_path
        self.device = device
        self.card_conf_threshold = card_conf_threshold
        self.field_conf_threshold = field_conf_threshold
        self.use_ensemble = use_ensemble
        self.use_tta = use_tta
        self.use_text_region_refinement = use_text_region_refinement
        self.place_bbox_pad_pct = 0.06

        self._card_detector: Any = None
        self._field_detector: Any = None
        self._vietocr: VietOCRRecognizer | None = None
        self._vietocr_address_origin: VietOCRRecognizer | None = None
        self._paddleocr: PaddleOCRRecognizer | None = None
        self._parser = CCCDParser()

    # ── lazy loaders ──────────────────────────────────────────────────────

    def _get_card_detector(self) -> Any:
        if self._card_detector is None:
            self._card_detector = self._load_yolo(self.card_detector_path)
        return self._card_detector

    def _get_field_detector(self) -> Any:
        if self._field_detector is None:
            self._field_detector = self._load_yolo(self.field_detector_path)
        return self._field_detector

    def _get_vietocr(self) -> VietOCRRecognizer:
        if self._vietocr is None:
            self._vietocr = VietOCRRecognizer(device=self.device)
        return self._vietocr

    def _get_vietocr_address_origin(self) -> VietOCRRecognizer:
        if self._vietocr_address_origin is None:
            self._vietocr_address_origin = VietOCRRecognizer.address_origin_reviewed(device=self.device)
        return self._vietocr_address_origin

    def _get_vietocr_for_field(self, field_name: str | None) -> VietOCRRecognizer:
        if field_name in {"place_of_origin", "place_of_residence"}:
            return self._get_vietocr_address_origin()
        return self._get_vietocr()

    def _get_paddleocr(self) -> PaddleOCRRecognizer:
        if self._paddleocr is None:
            self._paddleocr = PaddleOCRRecognizer(device=self.device)
        return self._paddleocr

    @staticmethod
    def _load_yolo(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics package is required for YOLO detection. "
                "Install it with: pip install ultralytics"
            ) from exc

    # ── image loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load_image(image_path: str | Path) -> np.ndarray | None:
        path = Path(image_path)
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except OSError:
            return None

    # ── card detection ────────────────────────────────────────────────────

    def _detect_card(
        self, image: np.ndarray, result: PipelineResult
    ) -> np.ndarray | None:
        """Detect card bbox, rectify, trả về warped card image."""
        try:
            detector = self._get_card_detector()
            predictions = detector(image, conf=self.card_conf_threshold, verbose=False)
        except Exception as exc:
            result.errors.append(f"Card detection failed: {exc}")
            result._add_step("card_detection", "error", str(exc))
            return None

        # Lấy box tốt nhất (confidence cao nhất)
        best_box = None
        best_conf = 0.0
        for pred in predictions:
            boxes = pred.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf > best_conf:
                    best_conf = conf
                    best_box = boxes.xyxy[i].cpu().numpy().tolist()

        if best_box is None or best_conf < self.card_conf_threshold:
            result.warnings.append("Không phát hiện thẻ CCCD trong ảnh")
            result._add_step("card_detection", "warning", "no_card_detected")
            return None

        result.card_detected = True
        result.card_bbox = best_box
        result.card_confidence = best_conf
        result._add_step(
            "card_detection", "success",
            f"conf={best_conf:.3f}, bbox={[round(v, 1) for v in best_box]}"
        )

        # Rectify
        try:
            warped, meta = rectify_from_bbox(image, best_box, padding_ratio=0.02)
            result.rectification_method = meta.get("rectification_method", "bbox_crop")
            result.rectified_image = warped
            result._add_step("rectification", "success", result.rectification_method)
            return warped
        except Exception as exc:
            result.warnings.append(f"Rectification failed: {exc}")
            result._add_step("rectification", "warning", str(exc))
            # Fallback: crop bbox thô
            x1, y1, x2, y2 = [int(v) for v in best_box]
            h, w = image.shape[:2]
            crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            result.rectification_method = "raw_crop"
            result.rectified_image = crop
            return crop

    # ── enhancement ───────────────────────────────────────────────────────

    def _enhance(self, card_image: np.ndarray, result: PipelineResult) -> np.ndarray:
        """Adaptive tiered enhancement based on quality metrics."""
        try:
            from src.preprocessing.enhance import adaptive_enhance
            enh = adaptive_enhance(card_image)
            result.image_quality_score = enh.quality_metrics.get("image_quality_score", 0.0)
            result.quality_metrics = enh.quality_metrics
            result.enhancement_tier = enh.tier
            result.enhancement_tier_label = enh.tier_label
            result.enhancement_actions = enh.actions_applied
            result._add_step(
                "enhancement", "success",
                f"tier={enh.tier} ({enh.tier_label}), "
                f"quality={result.image_quality_score:.2f}, "
                f"actions={enh.actions_applied}"
            )
            return enh.image
        except Exception as exc:
            result.warnings.append(f"Enhancement failed: {exc}")
            result._add_step("enhancement", "warning", str(exc))
            try:
                quality = compute_image_quality_score(card_image)
                result.image_quality_score = quality.get("image_quality_score", 0.0)
                result.quality_metrics = quality
            except Exception:
                pass
            return card_image

    # ── field detection ───────────────────────────────────────────────────

    def _detect_fields(
        self, card_image: np.ndarray, result: PipelineResult
    ) -> list[dict[str, Any]]:
        """Detect field bboxes trong card image."""
        try:
            detector = self._get_field_detector()
            predictions = detector(card_image, conf=self.field_conf_threshold, verbose=False)
        except Exception as exc:
            result.errors.append(f"Field detection failed: {exc}")
            result._add_step("field_detection", "error", str(exc))
            return []

        detections: list[dict[str, Any]] = []
        for pred in predictions:
            boxes = pred.boxes
            if boxes is None:
                continue
            names: dict[int, str] = getattr(pred, "names", _FIELD_CLASS_NAMES)
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name not in _OCR_TARGET_CLASSES:
                    continue
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                detections.append({
                    "class_name": cls_name,
                    "bbox_xyxy": bbox,
                    "confidence": conf,
                })

        result.field_detections = detections
        result._add_step(
            "field_detection", "success",
            f"{len(detections)} fields detected: {[d['class_name'] for d in detections]}"
        )

        if len(detections) < 2:
            result.warnings.append(
                f"Chỉ phát hiện {len(detections)} field — ảnh có thể không phải mặt trước CCCD"
            )
        return detections

    # ── bbox trimming ────────────────────────────────────────────────────

    @staticmethod
    def _trim_place_bbox(
        bbox: np.ndarray, cls_name: str, card_height: int,
    ) -> np.ndarray:
        """Light trim of address/origin bbox — only remove obvious bleeding.

        Label text filtering is handled downstream by multiline_ocr, so we
        only do a minimal geometric trim here to avoid cutting actual content.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox]
        field_h = y2 - y1
        line_h = card_height / 18.0

        min_h = line_h * 2.0
        if y2 - y1 < min_h:
            mid = (float(bbox[1]) + float(bbox[3])) / 2
            y1 = mid - min_h / 2
            y2 = mid + min_h / 2

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _expand_place_bbox(
        self, bbox: np.ndarray, image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Expand address/origin bbox by a percentage to capture clipped text."""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = x2 - x1
        h = y2 - y1
        pad_x = w * self.place_bbox_pad_pct
        pad_y = h * self.place_bbox_pad_pct
        expanded = np.array(
            [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
            dtype=np.float32,
        )
        return clamp_bbox(expanded, image_shape)

    # ── OCR ───────────────────────────────────────────────────────────────

    def _run_ocr(
        self,
        card_image: np.ndarray,
        detections: list[dict[str, Any]],
        result: PipelineResult,
    ) -> None:
        """Crop từng field và chạy OCR, lưu vào result.ocr_results."""
        h, _ = card_image.shape[:2]

        detector: PaddleTextDetector | None = None
        paddle_recognizer: PaddleOCRRecognizer | None = None

        for det in detections:
            cls_name: str = det["class_name"]
            bbox: list[float] = det["bbox_xyxy"]
            det_conf: float = det["confidence"]

            # Clamp bbox vào kích thước card
            clamped = clamp_bbox(bbox, card_image.shape)

            if cls_name in ("address", "origin"):
                clamped = self._trim_place_bbox(clamped, cls_name, h)
                clamped = clamp_bbox(clamped, card_image.shape)
                clamped = self._expand_place_bbox(clamped, card_image.shape)

            try:
                crop = crop_image_xyxy(card_image, clamped)
            except Exception as exc:
                result.warnings.append(f"Crop failed for {cls_name}: {exc}")
                continue

            if crop is None or crop.size == 0:
                result.warnings.append(f"Empty crop for {cls_name}")
                continue

            if cls_name in ("address", "origin"):
                crop = enhance_field_crop(crop)

            canonical_field = _CLASS_TO_FIELD.get(cls_name, cls_name)
            vietocr = self._get_vietocr_for_field(canonical_field)
            debug_info: dict[str, Any] = {
                "class_name": cls_name,
                "canonical_field": canonical_field,
                "vietocr_model": vietocr.model_label,
                "detector_confidence": float(det_conf),
                "crop_bbox_xyxy": [round(float(v), 2) for v in clamped.tolist()],
                "crop_shape": list(crop.shape),
            }

            try:
                if is_multi_line_field(canonical_field):
                    if detector is None:
                        detector = PaddleTextDetector()
                    if paddle_recognizer is None:
                        try:
                            paddle_recognizer = self._get_paddleocr()
                        except Exception:
                            pass
                    ocr_out = recognize_multiline_field(
                        crop, canonical_field, vietocr,
                        detector=detector,
                        paddleocr_recognizer=paddle_recognizer,
                    )
                    text = ocr_out.text or ""
                    conf = float(ocr_out.score)
                    debug_info["ocr_strategy"] = "multiline_field"
                    debug_info["engine"] = ocr_out.engine
                    debug_info["raw"] = ocr_out.raw or {}
                elif self.use_text_region_refinement:
                    paddle = self._get_paddleocr()
                    refined_viet, refined_paddle = run_hybrid_field_ocr(
                        image=crop,
                        field_name=canonical_field,
                        paddle_adapter=paddle,
                        viet_adapter=vietocr,
                    )
                    chosen = select_best_ocr_result(
                        canonical_field,
                        refined_viet,
                        refined_paddle,
                    )
                    text = chosen.text or ""
                    conf = float(chosen.score)
                    debug_info["ocr_strategy"] = "hybrid_text_region_refinement"
                    debug_info["engine"] = chosen.engine
                    debug_info["raw"] = chosen.raw or {}
                    debug_info["candidates"] = {
                        "vietocr": {
                            "text": refined_viet.text,
                            "confidence": float(refined_viet.score),
                        },
                        "paddleocr": {
                            "text": refined_paddle.text,
                            "confidence": float(refined_paddle.score),
                        },
                    }
                elif self.use_ensemble:
                    paddle = self._get_paddleocr()
                    ocr_out = ensemble_recognize(
                        field_name=canonical_field,
                        image=crop,
                        vietocr_recognizer=vietocr,
                        paddleocr_recognizer=paddle,
                    )
                    text = ocr_out.text or ""
                    conf = float(ocr_out.score)
                    debug_info["ocr_strategy"] = "ensemble"
                    debug_info["engine"] = ocr_out.engine
                    debug_info["raw"] = ocr_out.raw or {}
                else:
                    ocr_out = vietocr.recognize(crop, field_name=canonical_field)
                    text = ocr_out.text or ""
                    conf = float(ocr_out.score)
                    debug_info["ocr_strategy"] = "vietocr_direct"
                    debug_info["engine"] = ocr_out.engine
                    debug_info["raw"] = ocr_out.raw or {}
            except Exception as exc:
                result.warnings.append(f"OCR failed for {cls_name}: {exc}")
                text = ""
                conf = 0.0
                debug_info["ocr_strategy"] = "failed"
                debug_info["error"] = str(exc)

            # Penalize confidence when OCR text doesn't match expected field type
            sanity = _ocr_sanity_score(cls_name, text)
            debug_info["sanity_score"] = float(sanity)
            if sanity < 1.0:
                conf *= sanity
                if sanity <= 0.1:
                    result.warnings.append(
                        f"{cls_name}: OCR text doesn't match field type (sanity={sanity:.1f})"
                    )
            debug_info["final_text"] = text
            debug_info["final_confidence"] = float(conf)

            # Lưu kết quả — giữ kết quả có confidence cao hơn nếu class trùng
            existing = result.ocr_results.get(cls_name)
            if existing is None or conf > existing[1]:
                result.ocr_results[cls_name] = (text, conf)
                result.field_debug[cls_name] = debug_info

        result._add_step(
            "ocr",
            "success",
            f"{len(result.ocr_results)} fields recognized"
        )

    # ── conditional TTA retry ────────────────────────────────────────────

    def _retry_with_tta(
        self,
        card_image: np.ndarray,
        detections: list[dict[str, Any]],
        result: PipelineResult,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Retry low-confidence fields with TTA variants.

        Triggered when image_quality_score < 0.5 (all fields) or
        individual field confidence < threshold.
        """
        from src.ocr.tta import generate_ocr_tta_variants

        force_all = result.image_quality_score < 0.5
        h, w = card_image.shape[:2]

        detector: PaddleTextDetector | None = None
        paddle_recognizer = None
        tta_retried: list[str] = []

        for det in detections:
            cls_name: str = det["class_name"]
            canonical_field = _CLASS_TO_FIELD.get(cls_name, cls_name)
            vietocr = self._get_vietocr_for_field(canonical_field)

            existing = result.ocr_results.get(cls_name)
            if existing is None:
                continue
            existing_text, existing_conf = existing

            if not force_all and existing_conf >= confidence_threshold:
                continue

            bbox = det["bbox_xyxy"]
            clamped = clamp_bbox(bbox, card_image.shape)
            if cls_name in ("address", "origin"):
                clamped = self._trim_place_bbox(clamped, cls_name, h)
                clamped = clamp_bbox(clamped, card_image.shape)

            try:
                crop = crop_image_xyxy(card_image, clamped)
            except Exception:
                continue
            if crop is None or crop.size == 0:
                continue

            variants = generate_ocr_tta_variants(crop, canonical_field, enable_tta=True)
            best_text = existing_text
            best_conf = existing_conf

            for variant in variants:
                if variant.name == "base":
                    continue
                try:
                    if is_multi_line_field(canonical_field):
                        if detector is None:
                            detector = PaddleTextDetector()
                        if paddle_recognizer is None:
                            try:
                                from src.ocr.paddleocr_adapter import PaddleOCRRecognizer
                                paddle_recognizer = PaddleOCRRecognizer()
                            except Exception:
                                pass
                        ocr_out = recognize_multiline_field(
                            variant.image, canonical_field, vietocr,
                            detector=detector,
                            paddleocr_recognizer=paddle_recognizer,
                        )
                        text = ocr_out.text or ""
                        conf = float(ocr_out.score)
                    elif self.use_ensemble:
                        from src.ocr.paddleocr_adapter import PaddleOCRRecognizer
                        paddle = PaddleOCRRecognizer()
                        ocr_out = ensemble_recognize(
                            field_name=canonical_field,
                            image=variant.image,
                            vietocr_recognizer=vietocr,
                            paddleocr_recognizer=paddle,
                        )
                        text = ocr_out.text or ""
                        conf = float(ocr_out.score)
                    else:
                        ocr_out = vietocr.recognize(variant.image, field_name=canonical_field)
                        text = ocr_out.text or ""
                        conf = float(ocr_out.score)

                    sanity = _ocr_sanity_score(cls_name, text)
                    conf *= sanity

                    if conf > best_conf:
                        best_text = text
                        best_conf = conf
                except Exception as exc:
                    LOGGER.debug("TTA variant %s failed for %s: %s", variant.name, cls_name, exc)
                    continue

            if best_conf > existing_conf:
                result.ocr_results[cls_name] = (best_text, best_conf)
                tta_retried.append(cls_name)

        if tta_retried:
            result._add_step(
                "tta_retry", "success",
                f"Improved {len(tta_retried)} fields: {tta_retried}"
            )
        else:
            result._add_step("tta_retry", "skipped", "No fields improved by TTA")

    # ── template-based ID fallback ──────────────────────────────────────

    def _try_template_id(
        self, card_image: np.ndarray, result: PipelineResult,
    ) -> None:
        """Fallback: crop the known ID region on a standard CCCD and re-OCR.

        On a front-side CCCD (856×540 after rectification), the ID number
        sits at roughly y=38-50%, x=38-85% of the card.
        """
        import re as _re

        existing = result.ocr_results.get("id")
        if existing:
            digits = "".join(c for c in existing[0] if c.isdigit())
            if len(digits) >= 12:
                return

        h, w = card_image.shape[:2]
        # Template region for the 12-digit ID on a standard CCCD front
        x1, y1 = int(w * 0.38), int(h * 0.36)
        x2, y2 = int(w * 0.88), int(h * 0.52)
        crop = card_image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        vietocr = self._get_vietocr()
        ocr_out = vietocr.recognize(crop, field_name="id_number")
        text = ocr_out.text or ""
        digits = "".join(c for c in text if c.isdigit())

        if len(digits) >= 12:
            conf = float(ocr_out.score)
            existing_conf = existing[1] if existing else 0.0
            if conf > existing_conf:
                result.ocr_results["id"] = (text, conf)
                result.warnings.append("id: used template crop fallback")

    # ── parsing ───────────────────────────────────────────────────────────

    def _parse(self, result: PipelineResult) -> None:
        """Parse và validate OCR results."""
        ocr_list = [
            {"class": cls, "text": text, "confidence": conf}
            for cls, (text, conf) in result.ocr_results.items()
        ]
        result.parsed_info = self._parser.parse_batch(ocr_list)
        result._add_step(
            "parsing",
            "success" if not result.parsed_info.needs_review else "review",
            f"needs_review={result.parsed_info.needs_review}"
        )

    # ── main entry ────────────────────────────────────────────────────────

    def __call__(self, image_path: str | Path) -> PipelineResult:
        """Chạy pipeline end-to-end trên 1 ảnh."""
        result = PipelineResult()

        # 1. Load image
        image = self._load_image(image_path)
        if image is None:
            result.errors.append(f"Không thể đọc ảnh: {image_path}")
            result._add_step("load_image", "error", "imread failed")
            return result
        result._add_step("load_image", "success", f"shape={image.shape}")

        # 2. Card detection + rectification
        card_image = self._detect_card(image, result)
        if card_image is None:
            result._add_step("pipeline", "abort", "no_card_detected")
            return result

        # 3. Orientation correction (heuristic)
        try:
            orient = auto_orient_for_ocr(card_image)
            if orient.angle != 0:
                card_image = orient.image
                result.warnings.append(
                    f"Ảnh bị xoay {orient.angle}° — đã tự động xoay lại"
                )
            result._add_step(
                "orientation",
                "success",
                f"angle={orient.angle}, conf={orient.confidence:.2f}"
            )
        except Exception as exc:
            result.warnings.append(f"Orientation detection failed: {exc}")
            result._add_step("orientation", "warning", str(exc))

        # 4. Enhancement
        card_image = self._enhance(card_image, result)

        # 5. Field detection
        detections = self._detect_fields(card_image, result)
        if not detections:
            result.warnings.append("Không phát hiện field nào — bỏ qua OCR")
            result._add_step("pipeline", "partial", "no_fields_detected")
            return result

        # 6. OCR
        self._run_ocr(card_image, detections, result)

        # 6a. Conditional TTA retry for low-confidence fields
        if self.use_tta:
            self._retry_with_tta(card_image, detections, result)

        # 6b. Template fallback for ID if detection missed it
        self._try_template_id(card_image, result)

        # 7. Parsing & Validation
        self._parse(result)

        return result

    def process_image(self, image_path: str | Path) -> PipelineResult:
        """Alias cho __call__."""
        return self(image_path)
