from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import cv2
import numpy as np

RectificationMethod = Literal["polygon_warp", "bbox_crop"]
RectificationQuality = Literal["good", "degraded"]
PointArray = np.ndarray

CARD_OUTPUT_SIZE: tuple[int, int] = (856, 540)


def _validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")


def _coerce_points(points: Iterable[Iterable[float]]) -> PointArray:
    array = np.asarray(list(points), dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError("Expected 4 points with shape (4, 2).")
    return array


def _coerce_bbox(bbox: Sequence[float], bbox_format: str = "xyxy") -> np.ndarray:
    values = np.asarray(list(bbox), dtype=np.float32).reshape(-1)
    if values.size != 4:
        raise ValueError("Bounding box must contain exactly four values.")
    if bbox_format == "xyxy":
        return values
    if bbox_format == "xywh":
        x, y, w, h = values.tolist()
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    raise ValueError(f"Unsupported bbox format: {bbox_format}")


def order_points(points: Iterable[Iterable[float]]) -> PointArray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""

    pts = _coerce_points(points)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _destination_points(output_size: tuple[int, int]) -> PointArray:
    width, height = output_size
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def _perspective_matrix(points: Iterable[Iterable[float]], output_size: tuple[int, int]) -> np.ndarray:
    return cv2.getPerspectiveTransform(order_points(points), _destination_points(output_size))


def four_point_warp(
    image: np.ndarray,
    points: Iterable[Iterable[float]],
    output_size: tuple[int, int] = CARD_OUTPUT_SIZE,
) -> np.ndarray:
    """Warp a quadrilateral card region into the canonical CCCD size."""

    _validate_image(image)
    matrix = _perspective_matrix(points, output_size)
    width, height = output_size
    return cv2.warpPerspective(image, matrix, (width, height))


def _clamp_bbox(xyxy: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
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


def _pad_bbox(xyxy: np.ndarray, image_shape: tuple[int, ...], padding_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.tolist()
    pad_x = max(1.0, (x2 - x1) * padding_ratio)
    pad_y = max(1.0, (y2 - y1) * padding_ratio)
    padded = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)
    return _clamp_bbox(padded, image_shape)


def _bbox_to_points(xyxy: np.ndarray) -> PointArray:
    x1, y1, x2, y2 = xyxy.tolist()
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _crop_and_resize(image: np.ndarray, xyxy: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = np.round(xyxy).astype(int).tolist()
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image.copy()
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)


def rectify_from_polygon(
    image: np.ndarray,
    polygon: Iterable[Iterable[float]],
    output_size: tuple[int, int] = CARD_OUTPUT_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rectify a card region using 4 polygon corner points."""

    _validate_image(image)
    ordered_points = order_points(polygon)
    matrix = _perspective_matrix(ordered_points, output_size)
    width, height = output_size
    warped = cv2.warpPerspective(image, matrix, (width, height))
    meta: dict[str, Any] = {
        "rectification_method": "polygon_warp",
        "rectification_quality": "good",
        "source_points": ordered_points,
        "output_size": output_size,
        "perspective_matrix": matrix,
    }
    return warped, meta


def rectify_from_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    padding_ratio: float = 0.05,
    output_size: tuple[int, int] = CARD_OUTPUT_SIZE,
    bbox_format: str = "xyxy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fallback rectification using bbox crop plus padding."""

    _validate_image(image)
    xyxy = _coerce_bbox(bbox, bbox_format=bbox_format)
    padded_bbox = _pad_bbox(xyxy, image.shape, padding_ratio=padding_ratio)
    source_points = _bbox_to_points(padded_bbox)
    matrix = _perspective_matrix(source_points, output_size)
    crop = _crop_and_resize(image, padded_bbox, output_size)
    meta: dict[str, Any] = {
        "rectification_method": "bbox_crop",
        "rectification_quality": "degraded",
        "source_points": source_points,
        "output_size": output_size,
        "perspective_matrix": matrix,
        "bbox": padded_bbox,
    }
    return crop, meta


__all__ = [
    "CARD_OUTPUT_SIZE",
    "RectificationMethod",
    "RectificationQuality",
    "four_point_warp",
    "order_points",
    "rectify_from_bbox",
    "rectify_from_polygon",
]
