from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_sharpness(raw_value: float) -> float:
    return float(np.clip(raw_value / 1200.0, 0.0, 1.0))


def _normalize_brightness(raw_value: float) -> float:
    target = 145.0
    tolerance = 145.0
    return float(np.clip(1.0 - abs(raw_value - target) / tolerance, 0.0, 1.0))


def _normalize_contrast(raw_value: float) -> float:
    return float(np.clip(raw_value / 64.0, 0.0, 1.0))


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE while preserving the image channel layout."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def denoise_image(
    image: np.ndarray,
    strength: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """Denoise an image for OCR without changing its number of channels."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            strength,
            template_window_size,
            search_window_size,
        )
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        strength,
        strength,
        template_window_size,
        search_window_size,
    )


def compute_image_quality_score(image: np.ndarray) -> dict[str, float]:
    """Compute normalized sharpness, brightness, contrast, and aggregate quality."""

    if image is None or image.size == 0:
        return {
            "sharpness_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "image_quality_score": 0.0,
        }

    gray = _to_gray(image)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_raw = float(np.mean(gray))
    contrast_raw = float(np.std(gray))

    sharpness_score = _normalize_sharpness(sharpness_raw)
    brightness_score = _normalize_brightness(brightness_raw)
    contrast_score = _normalize_contrast(contrast_raw)
    image_quality_score = float(
        np.clip(0.45 * sharpness_score + 0.30 * brightness_score + 0.25 * contrast_score, 0.0, 1.0)
    )

    return {
        "sharpness_score": sharpness_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "image_quality_score": image_quality_score,
    }


def enhance_card_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    denoise_strength: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a lightweight OCR-oriented enhancement pipeline."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    clahe_image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    denoised_image = denoise_image(clahe_image, strength=denoise_strength)
    metrics = compute_image_quality_score(denoised_image)
    return denoised_image, metrics


__all__ = [
    "apply_clahe",
    "compute_image_quality_score",
    "denoise_image",
    "enhance_card_image",
]
