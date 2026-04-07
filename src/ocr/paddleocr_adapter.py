"""
PaddleOCR Adapter
Wrap PaddleOCR thành API thống nhất.

Lưu ý: PaddleOCR >= 3.x đổi API (dùng predict() thay ocr()).
Nếu môi trường không tương thích, adapter tự fallback về VietOCR.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .vietocr_adapter import OCRResult


def _patch_pil_util() -> None:
    """Patch PIL._util.is_directory bị xóa từ Pillow 10."""
    import PIL._util

    if not hasattr(PIL._util, "is_directory"):
        PIL._util.is_directory = os.path.isdir
    if not hasattr(PIL._util, "is_path"):
        PIL._util.is_path = lambda f: isinstance(f, (str, bytes, os.PathLike))


class PaddleOCRAdapter:
    """Wrapper cho PaddleOCR."""

    def __init__(self, lang: str = "en") -> None:
        _patch_pil_util()
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=lang)
            self._available = True
        except Exception:
            self._ocr = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, image_path: str | Path) -> OCRResult:
        if not self._available:
            return OCRResult(text="", confidence=0.0)

        try:
            result = self._ocr.predict(str(image_path))
            return self._parse_result(result)
        except Exception:
            return OCRResult(text="", confidence=0.0)

    def predict_pil(self, image: Image.Image) -> OCRResult:
        if not self._available:
            return OCRResult(text="", confidence=0.0)

        import tempfile
        import cv2
        import numpy as np

        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, arr)
            return self.predict(f.name)

    def _parse_result(self, result) -> OCRResult:
        if not result:
            return OCRResult(text="", confidence=0.0)

        texts, confs = [], []
        for res in result:
            for text, score in zip(
                res.get("rec_texts", []), res.get("rec_scores", [])
            ):
                texts.append(text)
                confs.append(score)

        if not texts:
            return OCRResult(text="", confidence=0.0)

        return OCRResult(
            text=" ".join(texts),
            confidence=float(np.mean(confs)),
        )
