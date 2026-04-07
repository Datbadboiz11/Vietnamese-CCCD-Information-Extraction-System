"""
Module Image Enhancement
Tăng cường chất lượng ảnh thẻ CCCD sau rectification.

Kỹ thuật:
  - CLAHE: cân bằng sáng cục bộ
  - Bilateral filter / fastNlMeansDenoising: khử noise
  - Phát hiện lóa (overexposed)
  - Tính image_quality_score
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhanceResult:
    image: np.ndarray           # ảnh đã enhance (BGR)
    quality_score: float        # [0, 1] — điểm chất lượng tổng hợp
    sharpness: float            # [0, 1]
    brightness: float           # [0, 1]
    contrast: float             # [0, 1]
    has_glare: bool             # True nếu phát hiện lóa


def _calc_sharpness(gray: np.ndarray) -> float:
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(1.0, lap_var / 500.0)


def _calc_brightness(gray: np.ndarray) -> float:
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 140) / 140)


def _calc_contrast(gray: np.ndarray) -> float:
    std = float(gray.std())
    return min(1.0, std / 80.0)


def _detect_glare(gray: np.ndarray, threshold: int = 250) -> bool:
    overexposed = np.sum(gray > threshold)
    ratio = overexposed / gray.size
    return ratio > 0.05


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _denoise(image: np.ndarray) -> np.ndarray:
    """Bilateral filter: giữ cạnh, giảm noise."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance(image: np.ndarray) -> EnhanceResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính quality score trước khi enhance
    sharpness  = _calc_sharpness(gray)
    brightness = _calc_brightness(gray)
    contrast   = _calc_contrast(gray)
    has_glare  = _detect_glare(gray)

    # Enhance
    enhanced = _apply_clahe(image)
    enhanced = _denoise(enhanced)

    # Giảm lóa nếu phát hiện
    if has_glare:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = v > 250
        v[mask] = 220
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Quality score = weighted average
    quality_score = 0.5 * sharpness + 0.3 * brightness + 0.2 * contrast

    return EnhanceResult(
        image=enhanced,
        quality_score=round(quality_score, 4),
        sharpness=round(sharpness, 4),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        has_glare=has_glare,
    )
