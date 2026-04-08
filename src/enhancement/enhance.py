from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhancementResult:
    image: np.ndarray
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float


def _normalize_sharpness(value: float) -> float:
    return float(np.clip(value / 1000.0, 0.0, 1.0))


def _normalize_brightness(value: float) -> float:
    return float(np.clip(1.0 - abs(value - 140.0) / 140.0, 0.0, 1.0))


def _normalize_contrast(value: float) -> float:
    return float(np.clip(value / 64.0, 0.0, 1.0))


def compute_quality_metrics(image: np.ndarray) -> tuple[float, float, float, float]:
    if image is None or image.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_raw = float(np.mean(gray))
    contrast_raw = float(np.std(gray))

    sharpness = _normalize_sharpness(sharpness_raw)
    brightness = _normalize_brightness(brightness_raw)
    contrast = _normalize_contrast(contrast_raw)
    score = float(np.clip(0.5 * sharpness + 0.3 * brightness + 0.2 * contrast, 0.0, 1.0))
    return score, sharpness, brightness, contrast


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def enhance_for_ocr(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    denoise_strength: int = 10,
    sharpen_threshold: float = 0.35,
) -> EnhancementResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    clahe_image = _apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    gray = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY) if clahe_image.ndim == 3 else clahe_image
    denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)

    score, sharpness, brightness, contrast = compute_quality_metrics(denoised)
    if sharpness < sharpen_threshold:
        denoised = _sharpen(denoised)
        score, sharpness, brightness, contrast = compute_quality_metrics(denoised)

    enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return EnhancementResult(
        image=enhanced_bgr,
        quality_score=score,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
    )
