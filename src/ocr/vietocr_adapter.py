from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.ocr.types import OCRResult
from src.ocr.utils import calibrate_ocr_confidence, cleanup_ocr_text, empty_ocr_result, estimate_text_confidence, normalize_text_for_field

LOGGER = logging.getLogger(__name__)


def _disable_broken_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key, "")
        if "127.0.0.1:9" in value or "localhost:9" in value:
            os.environ.pop(key, None)


def _infer_torch_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_torch_device(device: str | None) -> str:
    requested = (device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return _infer_torch_device()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested if ":" in requested else "cuda:0"
        except Exception:
            pass
        raise RuntimeError(f"Requested VietOCR device '{device}', but CUDA is not available.")
    return device or "cpu"


class VietOCRRecognizer:
    """Thin VietOCR adapter with lazy model loading and safe failure handling."""

    def __init__(self, config_name: str = "vgg_transformer", device: str | None = None) -> None:
        self.config_name = config_name
        self.device = _resolve_torch_device(device)
        self._predictor: Any | None = None

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor

        _disable_broken_proxy_env()

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError(
                "VietOCR is unavailable. Install VietOCR dependencies and model weights before running recognition."
            ) from exc

        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS

        config = Cfg.load_config_from_name(self.config_name)
        config["device"] = self.device
        config["predictor"]["beamsearch"] = False
        weights_value = str(config.get("weights", ""))
        weights_path = (
            Path(tempfile.gettempdir()) / Path(weights_value).name
            if weights_value.startswith("http")
            else Path(weights_value).expanduser()
        )

        try:
            self._predictor = Predictor(config)
        except RuntimeError as exc:
            message = str(exc)
            is_corrupted = "PytorchStreamReader failed reading zip archive" in message or "failed finding central directory" in message
            if not is_corrupted:
                raise
            if weights_path.exists():
                try:
                    weights_path.unlink()
                except OSError:
                    LOGGER.warning("Could not remove corrupted VietOCR cache at %s", weights_path)
            self._predictor = Predictor(config)

        return self._predictor

    def _prepare_image(self, image: np.ndarray, field_name: str | None = None) -> np.ndarray:
        prepared = image.copy()
        if prepared.ndim == 2:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

        height, width = prepared.shape[:2]
        pad_x = max(4, int(round(width * 0.06)))
        pad_y = max(4, int(round(height * 0.10)))
        prepared = cv2.copyMakeBorder(
            prepared,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            borderType=cv2.BORDER_REPLICATE,
        )

        min_height = 64 if field_name in {"id", "id_number", "birth", "date_of_birth"} else 80
        scale = max(1.0, min_height / max(1, prepared.shape[0]))
        if scale > 1.0:
            prepared = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return prepared

    def _parse_prediction(self, prediction: Any, field_name: str | None) -> tuple[str, float, dict[str, Any]]:
        if isinstance(prediction, dict):
            text = str(prediction.get("text") or prediction.get("prediction") or prediction.get("pred") or "")
            try:
                score = float(prediction.get("prob") or prediction.get("confidence") or prediction.get("score") or 0.0)
            except (TypeError, ValueError):
                score = estimate_text_confidence(text, field_name)
            return text, score, dict(prediction)

        if isinstance(prediction, tuple) and len(prediction) >= 2:
            text = str(prediction[0] or "")
            try:
                score = float(prediction[1])
            except (TypeError, ValueError):
                score = estimate_text_confidence(text, field_name)
            return text, score, {"text": text, "score": score, "prediction": repr(prediction)}

        text = str(prediction or "")
        score = estimate_text_confidence(text, field_name)
        return text, score, {"text": text, "score": score}

    def recognize(self, image: np.ndarray, field_name: str | None = None) -> OCRResult:
        if image is None or image.size == 0:
            return empty_ocr_result("vietocr")

        try:
            predictor = self._get_predictor()
        except Exception as exc:
            LOGGER.warning("VietOCR initialization failed: %s", exc)
            result = empty_ocr_result("vietocr")
            result.error_message = str(exc)
            return result

        prepared = self._prepare_image(image, field_name)
        pil_image = Image.fromarray(prepared if prepared.ndim == 2 else prepared[:, :, ::-1])

        try:
            try:
                prediction = predictor.predict(pil_image, return_prob=True)
            except TypeError:
                prediction = predictor.predict(pil_image)
        except Exception as exc:
            LOGGER.warning("VietOCR inference failed: %s", exc)
            result = empty_ocr_result("vietocr")
            result.error_message = str(exc)
            return result

        raw_text, raw_score, raw_payload = self._parse_prediction(prediction, field_name)
        cleaned_text = cleanup_ocr_text(raw_text, field_name)
        calibrated_score = calibrate_ocr_confidence(cleaned_text, raw_score, field_name)
        normalized_text = normalize_text_for_field(cleaned_text, field_name)
        return OCRResult(
            text=cleaned_text,
            score=calibrated_score,
            engine="vietocr",
            raw={**raw_payload, "text": raw_text, "score": raw_score},
            needs_review=calibrated_score < 0.5 or normalized_text == "",
            normalized_text=normalized_text,
        )

    def predict(self, image: np.ndarray, field_name: str | None = None) -> OCRResult:
        return self.recognize(image, field_name=field_name)


VietOCRAdapter = VietOCRRecognizer

__all__ = ["VietOCRAdapter", "VietOCRRecognizer"]
