from __future__ import annotations

from collections.abc import Iterable
import inspect

import numpy as np

from src.ocr.types import OCRResult
from src.ocr.utils import normalize_text_for_field


def _flatten_paddle_candidates(node: object) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    if hasattr(node, "json"):
        try:
            return _flatten_paddle_candidates(node.json)
        except Exception:
            pass

    if isinstance(node, dict):
        if "rec_texts" in node:
            texts = node.get("rec_texts") or []
            scores = node.get("rec_scores") or []
            for idx, text in enumerate(texts):
                try:
                    score = float(scores[idx]) if idx < len(scores) else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                candidates.append((str(text or ""), score))
            return candidates

        if "rec_text" in node:
            text = str(node.get("rec_text") or "")
            try:
                score = float(node.get("rec_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            return [(text, score)]

    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str):
        text = node[0]
        try:
            score = float(node[1])
        except (TypeError, ValueError):
            score = 0.0
        return [(text, score)]

    if isinstance(node, Iterable) and not isinstance(node, (str, bytes)):
        for item in node:
            candidates.extend(_flatten_paddle_candidates(item))
    return candidates


class PaddleOCRAdapter:
    def __init__(self, language: str = "vi", use_angle_cls: bool = False, show_log: bool = False) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._ocr = None

    def _get_client(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "paddleocr is not installed. Install dependencies from requirements.txt before running OCR."
            ) from exc

        kwargs = {
            "lang": self.language,
            "device": "cpu",
            "enable_mkldnn": False,
            "enable_hpi": False,
        }
        try:
            signature = inspect.signature(PaddleOCR.__init__)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            kwargs["use_angle_cls"] = self.use_angle_cls
        elif "use_textline_orientation" in signature.parameters:
            kwargs["use_textline_orientation"] = self.use_angle_cls
        elif "use_angle_cls" in signature.parameters:
            kwargs["use_angle_cls"] = self.use_angle_cls

        if signature is None or "show_log" in signature.parameters:
            kwargs["show_log"] = self.show_log

        try:
            self._ocr = PaddleOCR(**kwargs)
        except ValueError as exc:
            # PaddleOCR 3.x dropped `show_log`; retry without it for compatibility.
            if "Unknown argument: show_log" not in str(exc):
                raise
            kwargs.pop("show_log", None)
            self._ocr = PaddleOCR(**kwargs)
        except ModuleNotFoundError as exc:
            if exc.name != "paddle":
                raise
            raise ImportError(
                "PaddleOCR requires `paddlepaddle`, but it is not installed in the current environment. "
                "Install `paddlepaddle` before running OCR."
            ) from exc
        return self._ocr

    def predict(self, image: np.ndarray, field_name: str) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(
                text="",
                confidence=0.0,
                backend="paddleocr",
                needs_review=True,
                raw_text="",
                normalized_text="",
            )

        client = self._get_client()
        predict_signature = inspect.signature(client.predict)
        if "use_doc_orientation_classify" in predict_signature.parameters:
            result = client.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            result = client.ocr(image, det=False, rec=True, cls=False)
        candidates = _flatten_paddle_candidates(result)
        if not candidates:
            return OCRResult(
                text="",
                confidence=0.0,
                backend="paddleocr",
                needs_review=True,
                raw_text="",
                normalized_text="",
            )

        text = " ".join(candidate[0] for candidate in candidates).strip()
        confidence = float(sum(candidate[1] for candidate in candidates) / len(candidates))
        normalized_text = normalize_text_for_field(text, field_name)
        return OCRResult(
            text=text,
            confidence=confidence,
            backend="paddleocr",
            needs_review=confidence < 0.5 or normalized_text == "",
            raw_text=text,
            normalized_text=normalized_text,
        )
