"""
VietOCR Adapter
Wrap VietOCR (vgg_transformer) thành API thống nhất.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float


class VietOCRAdapter:
    """Wrapper cho VietOCR predictor."""

    def __init__(self, device: str = "cpu") -> None:
        # Patch PIL.Image.ANTIALIAS bị xóa từ Pillow 10
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["cnn"]["pretrained"] = False
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = False

        self._predictor = Predictor(cfg)

    def predict(self, image_path: str | Path) -> OCRResult:
        img = Image.open(str(image_path)).convert("RGB")
        text, conf = self._predictor.predict(img, return_prob=True)
        return OCRResult(text=text, confidence=float(conf))

    def predict_pil(self, image: Image.Image) -> OCRResult:
        text, conf = self._predictor.predict(image, return_prob=True)
        return OCRResult(text=text, confidence=float(conf))
