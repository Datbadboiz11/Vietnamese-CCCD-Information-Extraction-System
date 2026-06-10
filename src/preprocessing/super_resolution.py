"""Lightweight super-resolution for small/blurry OCR field crops.

Tries OpenCV DNN super-resolution (ESPCN x4) if available, otherwise falls
back to LANCZOS4 upscaling with unsharp-mask sharpening.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model" / "super_res"
_SR_INSTANCE = None
_SR_AVAILABLE: bool | None = None


def _try_load_dnn_sr() -> object | None:
    global _SR_AVAILABLE
    if _SR_AVAILABLE is False:
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = _MODEL_DIR / "ESPCN_x4.pb"
        if not model_path.exists():
            LOGGER.info("SR model not found at %s, using fallback upscaling", model_path)
            _SR_AVAILABLE = False
            return None
        sr.readModel(str(model_path))
        sr.setModel("espcn", 4)
        _SR_AVAILABLE = True
        LOGGER.info("Loaded ESPCN x4 super-resolution model")
        return sr
    except (AttributeError, Exception) as exc:
        LOGGER.debug("DNN super-resolution not available: %s", exc)
        _SR_AVAILABLE = False
        return None


def _get_sr() -> object | None:
    global _SR_INSTANCE
    if _SR_INSTANCE is None and _SR_AVAILABLE is not False:
        _SR_INSTANCE = _try_load_dnn_sr()
    return _SR_INSTANCE


def _enhanced_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """LANCZOS4 upscale + unsharp mask for sharper text edges."""
    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return sharpened


def super_resolve(image: np.ndarray, min_height: int = 192) -> np.ndarray:
    """Upscale a field crop using super-resolution if beneficial.

    Only applies when the image height is below *min_height*.  Returns the
    original image unchanged when upscaling is not needed.
    """
    if image is None or image.size == 0:
        return image

    h = image.shape[0]
    if h >= min_height:
        return image

    scale = min_height / max(1, h)
    if scale < 1.2:
        return image

    sr = _get_sr()
    if sr is not None:
        try:
            if image.ndim == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            result = sr.upsample(image_bgr)
            if result is not None and result.size > 0:
                target_h = int(h * 4)
                if target_h > min_height * 1.5:
                    final_h = min_height
                    final_w = int(result.shape[1] * (final_h / result.shape[0]))
                    result = cv2.resize(result, (final_w, final_h), interpolation=cv2.INTER_AREA)
                return result
        except Exception as exc:
            LOGGER.debug("DNN SR failed, using fallback: %s", exc)

    return _enhanced_upscale(image, scale)


__all__ = ["super_resolve"]
