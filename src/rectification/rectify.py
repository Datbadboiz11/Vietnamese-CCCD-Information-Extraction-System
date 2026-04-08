from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import cv2
import numpy as np

RectificationMethod = Literal["polygon_warp", "contour_warp", "bbox_fallback"]


@dataclass
class RectificationResult:
    image: np.ndarray
    method: RectificationMethod
    confidence: float
    used_fallback: bool
    corners: np.ndarray


def order_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected 4 corner points with shape (4, 2).")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def normalize_bbox(bbox: Iterable[float], bbox_format: str = "xyxy") -> np.ndarray:
    values = np.asarray(list(bbox), dtype=np.float32).reshape(-1)
    if values.size != 4:
        raise ValueError("Bounding box must contain four numbers.")

    if bbox_format == "xywh":
        x, y, w, h = values.tolist()
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    if bbox_format != "xyxy":
        raise ValueError(f"Unsupported bbox format: {bbox_format}")
    return values.astype(np.float32)


def warp_from_corners(
    image: np.ndarray,
    corners: Iterable[Iterable[float]],
    output_size: tuple[int, int] = (856, 540),
) -> np.ndarray:
    ordered = order_points(corners)
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def _pad_bbox(xyxy: np.ndarray, image_shape: tuple[int, int, int], padding_ratio: float) -> np.ndarray:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = xyxy.tolist()
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return np.array(
        [
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(width - 1), x2 + pad_x),
            min(float(height - 1), y2 + pad_y),
        ],
        dtype=np.float32,
    )


def _bbox_to_corners(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.tolist()
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _crop_bbox(image: np.ndarray, xyxy: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = np.round(xyxy).astype(int).tolist()
    crop = image[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if crop.size == 0:
        crop = image.copy()
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)


def _quadrilateral_metrics(points: np.ndarray) -> tuple[float, float]:
    ordered = order_points(points)
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    mean_width = max((width_top + width_bottom) / 2.0, 1e-6)
    mean_height = max((height_left + height_right) / 2.0, 1e-6)
    aspect_ratio = mean_width / mean_height
    area = cv2.contourArea(ordered.astype(np.float32))
    return area, aspect_ratio


def _find_corners_from_contours(
    image: np.ndarray,
    xyxy: np.ndarray,
    padding_ratio: float,
    min_area_ratio: float,
    aspect_ratio_range: tuple[float, float],
) -> np.ndarray | None:
    padded = _pad_bbox(xyxy, image.shape, padding_ratio)
    x1, y1, x2, y2 = np.round(padded).astype(int).tolist()
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_area = float(roi.shape[0] * roi.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        candidate = approx.reshape(4, 2).astype(np.float32)
        area, aspect_ratio = _quadrilateral_metrics(candidate)
        if area < roi_area * min_area_ratio:
            continue
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        candidate[:, 0] += x1
        candidate[:, 1] += y1
        return order_points(candidate)
    return None


def rectify_card(
    image: np.ndarray,
    bbox: Iterable[float] | None = None,
    polygon: Iterable[Iterable[float]] | None = None,
    output_size: tuple[int, int] = (856, 540),
    bbox_format: str = "xyxy",
    padding_ratio: float = 0.05,
    min_area_ratio: float = 0.35,
    aspect_ratio_range: tuple[float, float] = (1.2, 2.2),
) -> RectificationResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if polygon is not None:
        corners = order_points(polygon)
        warped = warp_from_corners(image, corners, output_size=output_size)
        return RectificationResult(
            image=warped,
            method="polygon_warp",
            confidence=0.95,
            used_fallback=False,
            corners=corners,
        )

    if bbox is None:
        fallback_corners = _bbox_to_corners(np.array([0, 0, image.shape[1] - 1, image.shape[0] - 1], dtype=np.float32))
        resized = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        return RectificationResult(
            image=resized,
            method="bbox_fallback",
            confidence=0.1,
            used_fallback=True,
            corners=fallback_corners,
        )

    xyxy = normalize_bbox(bbox, bbox_format=bbox_format)
    contour_corners = _find_corners_from_contours(
        image=image,
        xyxy=xyxy,
        padding_ratio=padding_ratio,
        min_area_ratio=min_area_ratio,
        aspect_ratio_range=aspect_ratio_range,
    )
    if contour_corners is not None:
        warped = warp_from_corners(image, contour_corners, output_size=output_size)
        return RectificationResult(
            image=warped,
            method="contour_warp",
            confidence=0.8,
            used_fallback=False,
            corners=contour_corners,
        )

    padded_bbox = _pad_bbox(xyxy, image.shape, padding_ratio)
    fallback = _crop_bbox(image, padded_bbox, output_size=output_size)
    return RectificationResult(
        image=fallback,
        method="bbox_fallback",
        confidence=0.4,
        used_fallback=True,
        corners=_bbox_to_corners(padded_bbox),
    )
