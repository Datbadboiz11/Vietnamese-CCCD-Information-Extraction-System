from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.preprocessing.enhance import compute_image_quality_score, enhance_card_image
from src.preprocessing.rectify import rectify_from_bbox

CARD_OUTPUT_SIZE = (856, 540)


@dataclass
class PreparedCard:
    warped_card: np.ndarray
    enhanced_card: np.ndarray | None
    crop_source_image: np.ndarray
    crop_source_name: str
    rectification_method: str
    rectification_confidence: float
    rectification_used_fallback: bool
    rectification_corners: np.ndarray | None
    perspective_matrix: np.ndarray | None
    card_bbox_xyxy: np.ndarray | None
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


def bbox_xywh_to_xyxy(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(value) for value in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_xyxy_to_corners(xyxy: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def clamp_bbox(xyxy: Iterable[float], image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    clipped = np.array(
        [
            np.clip(x1, 0, max(0, width - 1)),
            np.clip(y1, 0, max(0, height - 1)),
            np.clip(x2, 0, width),
            np.clip(y2, 0, height),
        ],
        dtype=np.float32,
    )
    if clipped[2] <= clipped[0]:
        clipped[2] = min(float(width), clipped[0] + 1.0)
    if clipped[3] <= clipped[1]:
        clipped[3] = min(float(height), clipped[1] + 1.0)
    return clipped


def crop_image_xyxy(image: np.ndarray, xyxy: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = np.round(np.asarray(list(xyxy), dtype=np.float32)).astype(int).tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return image[y1:y2, x1:x2]


def compute_perspective_matrix(corners: np.ndarray, output_size: tuple[int, int] = CARD_OUTPUT_SIZE) -> np.ndarray:
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners.astype(np.float32), destination)


def project_bbox(
    xyxy: Iterable[float],
    matrix: np.ndarray | None,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    bbox = np.asarray(list(xyxy), dtype=np.float32)
    if matrix is None:
        return clamp_bbox(bbox, output_shape)

    corners = bbox_xyxy_to_corners(bbox).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    xyxy_projected = np.array(
        [
            float(np.min(projected[:, 0])),
            float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])),
            float(np.max(projected[:, 1])),
        ],
        dtype=np.float32,
    )
    return clamp_bbox(xyxy_projected, output_shape)


def _quality_from_image(image: np.ndarray) -> tuple[float, float, float, float]:
    metrics = compute_image_quality_score(image)
    return (
        float(metrics["image_quality_score"]),
        float(metrics["sharpness_score"]),
        float(metrics["brightness_score"]),
        float(metrics["contrast_score"]),
    )


def prepare_card_for_ocr(
    image: np.ndarray,
    card_bbox: Iterable[float] | None = None,
    output_size: tuple[int, int] = CARD_OUTPUT_SIZE,
    enhance: bool = True,
    rectification_padding_ratio: float = 0.02,
) -> PreparedCard:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if card_bbox is not None:
        warped_card, rect_meta = rectify_from_bbox(
            image,
            card_bbox,
            output_size=output_size,
            bbox_format="xywh",
            padding_ratio=rectification_padding_ratio,
        )
        rectification_corners = np.asarray(rect_meta["source_points"], dtype=np.float32)
        perspective_matrix = np.asarray(rect_meta["perspective_matrix"], dtype=np.float32)
        card_bbox_xyxy = bbox_xywh_to_xyxy(card_bbox)
        rectification_method = str(rect_meta["rectification_method"])
        rectification_confidence = 0.95 if rectification_method == "polygon_warp" else 0.4
        rectification_used_fallback = rect_meta["rectification_quality"] != "good"
    else:
        warped_card = image.copy()
        rectification_corners = None
        perspective_matrix = None
        card_bbox_xyxy = None
        rectification_method = "none"
        rectification_confidence = 0.0
        rectification_used_fallback = False

    enhanced_card: np.ndarray | None = None
    crop_source_image = warped_card
    crop_source_name = "warped_card" if card_bbox is not None else "source_image"
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float

    if enhance:
        enhanced_card, enhance_meta = enhance_card_image(warped_card)
        crop_source_image = enhanced_card
        crop_source_name = "enhanced_card"
        quality_score = float(enhance_meta["image_quality_score"])
        sharpness = float(enhance_meta["sharpness_score"])
        brightness = float(enhance_meta["brightness_score"])
        contrast = float(enhance_meta["contrast_score"])
    else:
        quality_score, sharpness, brightness, contrast = _quality_from_image(warped_card)

    return PreparedCard(
        warped_card=warped_card,
        enhanced_card=enhanced_card,
        crop_source_image=crop_source_image,
        crop_source_name=crop_source_name,
        rectification_method=rectification_method,
        rectification_confidence=rectification_confidence,
        rectification_used_fallback=rectification_used_fallback,
        rectification_corners=rectification_corners,
        perspective_matrix=perspective_matrix,
        card_bbox_xyxy=card_bbox_xyxy,
        quality_score=quality_score,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
    )
