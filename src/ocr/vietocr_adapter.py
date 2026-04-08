from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from src.ocr.types import OCRResult
from src.ocr.utils import estimate_text_confidence, normalize_text_for_field


def _infer_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VietOCRAdapter:
    def __init__(self, config_name: str = "vgg_transformer", device: str | None = None) -> None:
        self.config_name = config_name
        self.device = device or _infer_device()
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise ImportError(
                "VietOCR dependencies are incomplete. Install requirements.txt, including `torchvision`, before running OCR."
            ) from exc

        config = Cfg.load_config_from_name(self.config_name)
        config["device"] = self.device
        config["predictor"]["beamsearch"] = False
        weights_value = str(config.get("weights", ""))
        if weights_value.startswith("http"):
            weights_path = Path(tempfile.gettempdir()) / Path(weights_value).name
        else:
            weights_path = Path(weights_value).expanduser()

        try:
            self._predictor = Predictor(config)
        except RuntimeError as exc:
            message = str(exc)
            corrupted_checkpoint = (
                "PytorchStreamReader failed reading zip archive" in message
                or "failed finding central directory" in message
            )
            if not corrupted_checkpoint:
                raise

            if weights_path.exists():
                try:
                    weights_path.unlink()
                except OSError:
                    pass

            try:
                self._predictor = Predictor(config)
            except Exception as retry_exc:
                raise RuntimeError(
                    "VietOCR cached checkpoint appears corrupted. Delete the cached weights file "
                    f"and rerun the OCR step.\nWeights path: {weights_path}"
                ) from retry_exc
        return self._predictor

    def _parse_prediction(self, prediction: Any, field_name: str) -> tuple[str, float]:
        if isinstance(prediction, tuple) and len(prediction) >= 2:
            text = str(prediction[0] or "")
            try:
                confidence = float(prediction[1])
            except (TypeError, ValueError):
                confidence = estimate_text_confidence(text, field_name)
            return text, confidence

        text = str(prediction or "")
        return text, estimate_text_confidence(text, field_name)

    def predict(self, image: np.ndarray, field_name: str) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(
                text="",
                confidence=0.0,
                backend="vietocr",
                needs_review=True,
                raw_text="",
                normalized_text="",
            )

        predictor = self._get_predictor()
        if image.ndim == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image[:, :, ::-1])

        try:
            prediction = predictor.predict(pil_image, return_prob=True)
        except TypeError:
            prediction = predictor.predict(pil_image)

        text, confidence = self._parse_prediction(prediction, field_name)
        normalized_text = normalize_text_for_field(text, field_name)
        return OCRResult(
            text=text,
            confidence=confidence,
            backend="vietocr",
            needs_review=confidence < 0.5 or normalized_text == "",
            raw_text=text,
            normalized_text=normalized_text,
        )
