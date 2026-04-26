from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OrientationResult:
    image: np.ndarray
    angle: int
    confidence: float
    method: str
    scores: dict[int, float]


def rotate_image_90n(image: np.ndarray, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return image.copy()
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported orientation angle: {angle}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _orientation_score(image: np.ndarray, angle: int) -> float:
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_profile = binary.mean(axis=1)
    col_profile = binary.mean(axis=0)
    row_variance = float(np.var(row_profile))
    col_variance = float(np.var(col_profile))
    profile_score = row_variance / max(row_variance + col_variance, 1e-6)

    edges = cv2.Canny(blurred, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    horizontal_density = float(horizontal_edges.mean() / 255.0)
    vertical_density = float(vertical_edges.mean() / 255.0)
    edge_score = horizontal_density / max(horizontal_density + vertical_density, 1e-6)

    height, width = gray.shape[:2]
    landscape_bonus = 1.0 if width >= height else 0.0
    upright_prior = 0.02 if angle == 0 else 0.0

    return 0.45 * profile_score + 0.35 * edge_score + 0.18 * landscape_bonus + upright_prior


def auto_orient_for_ocr(
    image: np.ndarray,
    min_confidence: float = 0.12,
    allow_180: bool = False,
) -> OrientationResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    candidates: dict[int, np.ndarray] = {angle: rotate_image_90n(image, angle) for angle in (0, 90, 180, 270)}
    scores = {angle: _orientation_score(candidate, angle) for angle, candidate in candidates.items()}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_angle, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(np.clip((best_score - second_score) / max(best_score, 1e-6), 0.0, 1.0))

    if best_angle in {90, 270} and confidence >= min_confidence:
        return OrientationResult(
            image=candidates[best_angle],
            angle=best_angle,
            confidence=confidence,
            method="projection_heuristic",
            scores=scores,
        )

    if best_angle == 180 and allow_180 and confidence >= max(0.35, min_confidence):
        return OrientationResult(
            image=candidates[best_angle],
            angle=best_angle,
            confidence=confidence,
            method="projection_heuristic",
            scores=scores,
        )

    return OrientationResult(
        image=candidates[0],
        angle=0,
        confidence=confidence if best_angle == 0 else 0.0,
        method="projection_heuristic",
        scores=scores,
    )


__all__ = ["OrientationResult", "auto_orient_for_ocr", "rotate_image_90n"]
