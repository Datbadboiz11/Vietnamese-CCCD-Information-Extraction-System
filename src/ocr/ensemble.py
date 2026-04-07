"""
OCR Ensemble
Kết hợp kết quả VietOCR + PaddleOCR theo logic trong PLAN.md:
  1. Nếu 2 model cho cùng kết quả → accept, confidence = max
  2. Nếu khác nhau:
     a. Chọn kết quả pass validation rule
     b. Nếu cả 2 đều pass / fail → chọn confidence cao hơn
     c. Nếu confidence chênh < 0.1 → flag needs_review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from .vietocr_adapter import OCRResult, VietOCRAdapter
from .paddleocr_adapter import PaddleOCRAdapter

CONF_DIFF_THRESHOLD = 0.1
PADDLE_FIELDS = {"id", "birth"}


@dataclass
class EnsembleResult:
    text: str
    confidence: float
    source: str                      # "vietocr" | "paddleocr" | "agree"
    needs_review: bool = False
    vietocr: OCRResult = field(default_factory=lambda: OCRResult("", 0.0))
    paddleocr: OCRResult = field(default_factory=lambda: OCRResult("", 0.0))


class OCREnsemble:
    """Chạy VietOCR + PaddleOCR và ensemble kết quả."""

    def __init__(self, device: str = "cpu") -> None:
        self.vietocr = VietOCRAdapter(device=device)
        self.paddleocr = PaddleOCRAdapter(lang="en")

    def predict(
        self,
        image_path: str | Path,
        field_class: str,
        validator: Callable[[str], bool] | None = None,
    ) -> EnsembleResult:
        img = Image.open(str(image_path)).convert("RGB")
        return self.predict_pil(img, field_class, validator)

    def predict_pil(
        self,
        image: Image.Image,
        field_class: str,
        validator: Callable[[str], bool] | None = None,
    ) -> EnsembleResult:
        r_viet = self.vietocr.predict_pil(image)
        r_paddle = (
            self.paddleocr.predict_pil(image)
            if self.paddleocr.available
            else OCRResult(text="", confidence=0.0)
        )

        return _ensemble(r_viet, r_paddle, field_class, validator)


def _ensemble(
    r_viet: OCRResult,
    r_paddle: OCRResult,
    field_class: str,
    validator: Callable[[str], bool] | None,
) -> EnsembleResult:
    # Nếu PaddleOCR không chạy được, dùng VietOCR
    if not r_paddle.text:
        return EnsembleResult(
            text=r_viet.text,
            confidence=r_viet.confidence,
            source="vietocr",
            vietocr=r_viet,
            paddleocr=r_paddle,
        )

    # Trường hợp 1: cùng kết quả
    if r_viet.text == r_paddle.text:
        return EnsembleResult(
            text=r_viet.text,
            confidence=max(r_viet.confidence, r_paddle.confidence),
            source="agree",
            vietocr=r_viet,
            paddleocr=r_paddle,
        )

    # Trường hợp 2: khác nhau
    v_pass = validator(r_viet.text) if validator else None
    p_pass = validator(r_paddle.text) if validator else None

    # 2a: chọn kết quả pass validation
    if v_pass is not None and p_pass is not None:
        if v_pass and not p_pass:
            return EnsembleResult(
                text=r_viet.text, confidence=r_viet.confidence,
                source="vietocr", vietocr=r_viet, paddleocr=r_paddle,
            )
        if p_pass and not v_pass:
            return EnsembleResult(
                text=r_paddle.text, confidence=r_paddle.confidence,
                source="paddleocr", vietocr=r_viet, paddleocr=r_paddle,
            )

    # 2b: cả 2 đều pass/fail → theo field_class ưu tiên
    if field_class in PADDLE_FIELDS:
        primary, secondary = r_paddle, r_viet
        primary_src = "paddleocr"
    else:
        primary, secondary = r_viet, r_paddle
        primary_src = "vietocr"

    best = primary if primary.confidence >= secondary.confidence else secondary
    best_src = primary_src if primary.confidence >= secondary.confidence else (
        "paddleocr" if primary_src == "vietocr" else "vietocr"
    )

    # 2c: confidence chênh < 0.1 → flag needs_review
    needs_review = abs(r_viet.confidence - r_paddle.confidence) < CONF_DIFF_THRESHOLD

    return EnsembleResult(
        text=best.text,
        confidence=best.confidence,
        source=best_src,
        needs_review=needs_review,
        vietocr=r_viet,
        paddleocr=r_paddle,
    )
