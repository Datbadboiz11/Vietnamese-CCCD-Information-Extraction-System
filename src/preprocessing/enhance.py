from __future__ import annotations

from dataclasses import dataclass, field
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


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction. gamma < 1.0 brightens, gamma > 1.0 darkens."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_sharpening(image: np.ndarray) -> np.ndarray:
    """Light unsharp-mask sharpening for blurry card images."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


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


@dataclass
class EnhancementResult:
    """Result of adaptive enhancement with diagnostics."""

    image: np.ndarray
    tier: int
    tier_label: str
    quality_metrics: dict[str, float]
    actions_applied: list[str] = field(default_factory=list)


def adaptive_enhance(image: np.ndarray) -> EnhancementResult:
    """Tiered enhancement based on granular quality metrics.

    Tier 1 (≥0.65): skip — high quality
    Tier 2 (0.45–0.65): light CLAHE
    Tier 3 (0.30–0.45): CLAHE + denoise
    Tier 4 (<0.30): aggressive CLAHE + denoise + gamma brighten
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    quality = compute_image_quality_score(image)
    score = quality["image_quality_score"]
    sharpness = quality["sharpness_score"]

    actions: list[str] = []
    result = image.copy()

    if score >= 0.65:
        tier, label = 1, "skip"
    elif score >= 0.45:
        tier, label = 2, "light_clahe"
        result = apply_clahe(result, clip_limit=2.0)
        actions.append("clahe")
    elif score >= 0.30:
        tier, label = 3, "clahe_denoise"
        result = apply_clahe(result, clip_limit=2.0)
        actions.append("clahe")
        result = denoise_image(result, strength=10)
        actions.append("denoise")
    else:
        tier, label = 4, "aggressive"
        result = apply_clahe(result, clip_limit=3.0)
        actions.append("aggressive_clahe")
        result = denoise_image(result, strength=12)
        actions.append("denoise")
        result = adjust_gamma(result, gamma=0.8)
        actions.append("gamma_brighten")

    if sharpness < 0.3:
        result = apply_sharpening(result)
        actions.append("sharpen")

    gray = _to_gray(result)
    raw_brightness = float(np.mean(gray))

    if raw_brightness < 80 and "gamma_brighten" not in actions:
        result = adjust_gamma(result, gamma=0.75)
        actions.append("gamma_brighten")
    elif raw_brightness > 220:
        result = adjust_gamma(result, gamma=1.4)
        actions.append("gamma_darken")

    return EnhancementResult(
        image=result,
        tier=tier,
        tier_label=label,
        quality_metrics=quality,
        actions_applied=actions,
    )


def enhance_field_crop(image: np.ndarray) -> np.ndarray:
    """Quality-adaptive enhancement for a single field crop before OCR.

    Computes quality on the crop itself (independent of card-level metrics)
    and applies targeted corrections: CLAHE for low contrast, bilateral
    filter for noise, gamma for extreme brightness.
    """
    if image is None or image.size == 0:
        return image

    quality = compute_image_quality_score(image)
    score = quality["image_quality_score"]
    sharpness = quality["sharpness_score"]
    result = image

    if score >= 0.70:
        return result

    if quality["contrast_score"] < 0.45:
        result = apply_clahe(result, clip_limit=1.5, tile_grid_size=(4, 4))

    if sharpness < 0.20:
        result = cv2.bilateralFilter(result, d=5, sigmaColor=50, sigmaSpace=50)

    gray = _to_gray(result)
    brightness = float(np.mean(gray))
    if brightness < 90:
        result = adjust_gamma(result, gamma=0.80)
    elif brightness > 210:
        result = adjust_gamma(result, gamma=1.3)

    return result


__all__ = [
    "EnhancementResult",
    "adaptive_enhance",
    "adjust_gamma",
    "apply_clahe",
    "apply_sharpening",
    "compute_image_quality_score",
    "denoise_image",
    "enhance_card_image",
    "enhance_field_crop",
]
