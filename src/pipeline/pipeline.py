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
from src.ocr.ensemble import ensemble_recognize
from src.ocr.vietocr_adapter import VietOCRRecognizer
from src.parsing.validators import CCCDParser, ParsedInfo
from src.preprocessing.enhance import apply_clahe, compute_image_quality_score
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

    # Field detection
    field_detections: list[dict[str, Any]] = field(default_factory=list)
    # {class_name, bbox_xyxy (trong card), confidence}

    # OCR results: {class_name: (text, confidence)}
    ocr_results: dict[str, tuple[str, float]] = field(default_factory=dict)

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
            "field_detections": self.field_detections,
            "ocr_results": {k: {"text": v[0], "confidence": v[1]} for k, v in self.ocr_results.items()},
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
    ) -> None:
        self.card_detector_path = card_detector_path
        self.field_detector_path = field_detector_path
        self.device = device
        self.card_conf_threshold = card_conf_threshold
        self.field_conf_threshold = field_conf_threshold
        self.use_ensemble = use_ensemble
        self.use_tta = use_tta

        self._card_detector: Any = None
        self._field_detector: Any = None
        self._vietocr: VietOCRRecognizer | None = None
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
        """CLAHE enhancement + quality score.

        CLAHE chỉ áp dụng khi chất lượng ảnh thấp (quality < 0.55).
        Ảnh chất lượng cao (CCCD chip có hoa văn nền phức tạp) nếu tăng
        tương phản sẽ làm nhiễu nền nổi rõ hơn, gây mất dấu khi OCR.
        """
        try:
            quality = compute_image_quality_score(card_image)
            result.image_quality_score = quality.get("image_quality_score", 0.0)
            # Ngưỡng: chỉ enhance khi ảnh thiếu sáng / mờ / thiếu tương phản
            if result.image_quality_score < 0.55:
                enhanced = apply_clahe(card_image)
                result._add_step(
                    "enhancement", "success",
                    f"clahe_applied, quality={result.image_quality_score:.2f}"
                )
                return enhanced
            else:
                result._add_step(
                    "enhancement", "skipped",
                    f"quality={result.image_quality_score:.2f} (>= 0.55, skip CLAHE)"
                )
                return card_image
        except Exception as exc:
            result.warnings.append(f"Enhancement failed: {exc}")
            result._add_step("enhancement", "warning", str(exc))
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

    # ── OCR ───────────────────────────────────────────────────────────────

    def _run_ocr(
        self,
        card_image: np.ndarray,
        detections: list[dict[str, Any]],
        result: PipelineResult,
    ) -> None:
        """Crop từng field và chạy OCR, lưu vào result.ocr_results."""
        vietocr = self._get_vietocr()
        h, w = card_image.shape[:2]

        for det in detections:
            cls_name: str = det["class_name"]
            bbox: list[float] = det["bbox_xyxy"]
            det_conf: float = det["confidence"]

            # Clamp bbox vào kích thước card
            clamped = clamp_bbox(bbox, card_image.shape)
            try:
                crop = crop_image_xyxy(card_image, clamped)
            except Exception as exc:
                result.warnings.append(f"Crop failed for {cls_name}: {exc}")
                continue

            if crop is None or crop.size == 0:
                result.warnings.append(f"Empty crop for {cls_name}")
                continue

            canonical_field = _CLASS_TO_FIELD.get(cls_name, cls_name)

            try:
                if self.use_ensemble:
                    from src.ocr.paddleocr_adapter import PaddleOCRRecognizer
                    paddle = PaddleOCRRecognizer()
                    ocr_out = ensemble_recognize(
                        field_name=canonical_field,
                        image=crop,
                        vietocr_recognizer=vietocr,
                        paddleocr_recognizer=paddle,
                    )
                    text = ocr_out.text or ""
                    conf = float(ocr_out.score)
                else:
                    ocr_out = vietocr.recognize(crop, field_name=canonical_field)
                    text = ocr_out.text or ""
                    conf = float(ocr_out.score)
            except Exception as exc:
                result.warnings.append(f"OCR failed for {cls_name}: {exc}")
                text = ""
                conf = 0.0

            # Lưu kết quả — giữ kết quả có confidence cao hơn nếu class trùng
            existing = result.ocr_results.get(cls_name)
            if existing is None or conf > existing[1]:
                result.ocr_results[cls_name] = (text, conf)

        result._add_step(
            "ocr",
            "success",
            f"{len(result.ocr_results)} fields recognized"
        )

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

        # 7. Parsing & Validation
        self._parse(result)

        return result

    def process_image(self, image_path: str | Path) -> PipelineResult:
        """Alias cho __call__."""
        return self(image_path)
