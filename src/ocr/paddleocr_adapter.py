from __future__ import annotations

from collections.abc import Iterable
import inspect
import logging
import os
from typing import Any

import cv2
import numpy as np

from src.ocr.types import OCRResult, OCRSegment
from src.ocr.utils import TEXT_LINE_PICK_FIELDS, calibrate_ocr_confidence, canonicalize_field_name, cleanup_ocr_text, collapse_whitespace, empty_ocr_result, normalize_text_for_field

LOGGER = logging.getLogger(__name__)


def _configure_headless_matplotlib_backend() -> None:
    backend = (os.environ.get("MPLBACKEND") or "").strip().lower()
    if backend.startswith("module://matplotlib_inline") or not backend:
        os.environ["MPLBACKEND"] = "Agg"


def _infer_paddle_device() -> str:
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            return paddle.get_device()
    except Exception:
        pass
    return "cpu"


def _resolve_paddle_device(device: str | None) -> str:
    requested = (device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return _infer_paddle_device()
    if requested == "cpu":
        return "cpu"
    if requested in {"gpu", "gpu:0", "cuda", "cuda:0"}:
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                return "gpu:0"
        except Exception:
            pass
        raise RuntimeError(f"Requested PaddleOCR device '{device}', but CUDA is not available.")
    return device or "cpu"


def _coerce_box(box: object) -> np.ndarray | None:
    if box is None:
        return None
    try:
        array = np.asarray(box, dtype=np.float32)
    except Exception:
        return None
    if array.size < 4:
        return None
    return array.reshape(-1, 2)


def _box_key(box: np.ndarray | None) -> tuple[float, float]:
    if box is None or box.size == 0:
        return (1e9, 1e9)
    return (float(np.mean(box[:, 1])), float(np.min(box[:, 0])))


def _append_candidate(candidates: list[tuple[str, float, np.ndarray | None, int]], text: object, score: object, *, box: object = None) -> None:
    collapsed = collapse_whitespace(str(text or ""))
    if not collapsed:
        return
    try:
        parsed_score = float(score)
    except (TypeError, ValueError):
        parsed_score = 0.0
    candidates.append((collapsed, parsed_score, _coerce_box(box), len(candidates)))


def _flatten_candidates(node: object) -> list[tuple[str, float, np.ndarray | None, int]]:
    candidates: list[tuple[str, float, np.ndarray | None, int]] = []

    def walk(item: object) -> None:
        if hasattr(item, "json"):
            try:
                walk(item.json)
                return
            except Exception:
                pass

        if isinstance(item, dict):
            if "res" in item:
                walk(item["res"])
                return
            if "rec_texts" in item:
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                boxes = None
                for key in ("rec_polys", "dt_polys", "polys", "rec_boxes", "dt_boxes", "boxes", "bboxes"):
                    if key in item:
                        boxes = item.get(key) or []
                        break
                for index, text in enumerate(texts):
                    score = scores[index] if index < len(scores) else 0.0
                    box = boxes[index] if boxes is not None and index < len(boxes) else None
                    _append_candidate(candidates, text, score, box=box)
                return
            if "rec_text" in item:
                _append_candidate(
                    candidates,
                    item.get("rec_text"),
                    item.get("rec_score", 0.0),
                    box=item.get("dt_poly") or item.get("dt_box") or item.get("box"),
                )
                return

        if isinstance(item, (list, tuple)) and len(item) == 2:
            first, second = item
            if isinstance(first, str):
                _append_candidate(candidates, first, second)
                return
            if isinstance(second, (list, tuple)) and len(second) >= 2 and isinstance(second[0], str):
                _append_candidate(candidates, second[0], second[1], box=first)
                return

        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for child in item:
                walk(child)

    walk(node)
    return candidates


def _prepare_image(image: np.ndarray, field_name: str | None = None) -> tuple[np.ndarray, float, int, int]:
    prepared = image.copy()
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)

    height, width = prepared.shape[:2]
    pad_x = max(6, int(round(width * 0.08)))
    pad_y = max(6, int(round(height * 0.12)))
    prepared = cv2.copyMakeBorder(prepared, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE)

    min_height = 96 if field_name in {"id", "id_number", "birth", "date_of_birth"} else 128
    scale = max(1.0, min_height / max(1, prepared.shape[0]))
    if scale > 1.0:
        prepared = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return prepared, scale, pad_x, pad_y


def _joined_text(candidates: list[tuple[str, float, np.ndarray | None, int]]) -> str:
    ordered = sorted(candidates, key=lambda item: (_box_key(item[2]), item[3]))
    return "\n".join(item[0] for item in ordered)


def _candidate_to_segment(
    candidate: tuple[str, float, np.ndarray | None, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    original_shape: tuple[int, ...],
) -> OCRSegment:
    text, score, box, _ = candidate
    box_points: list[list[float]] | None = None
    if box is not None and box.size > 0:
        mapped = box.astype(np.float32).copy()
        mapped /= max(scale, 1e-6)
        mapped[:, 0] -= float(pad_x)
        mapped[:, 1] -= float(pad_y)
        height, width = original_shape[:2]
        mapped[:, 0] = np.clip(mapped[:, 0], 0.0, max(0.0, float(width - 1)))
        mapped[:, 1] = np.clip(mapped[:, 1], 0.0, max(0.0, float(height - 1)))
        box_points = [[float(x), float(y)] for x, y in mapped.tolist()]
    return OCRSegment(text=text, score=float(score), box=box_points)


def _build_hypotheses(candidates: list[tuple[str, float, np.ndarray | None, int]], field_name: str | None) -> list[tuple[str, float]]:
    canonical_field = canonicalize_field_name(field_name)
    ordered = sorted(candidates, key=lambda item: (_box_key(item[2]), item[3]))
    if not ordered:
        return []

    hypotheses: dict[str, float] = {}

    def add(text: str, score: float) -> None:
        candidate_text = collapse_whitespace(text)
        if not candidate_text:
            return
        previous = hypotheses.get(candidate_text)
        if previous is None or score > previous:
            hypotheses[candidate_text] = float(score)

    for text, score, _, _ in ordered:
        add(text, score)

    if len(ordered) > 1:
        mean_score = float(sum(item[1] for item in ordered) / len(ordered))
        add(" ".join(item[0] for item in ordered), mean_score + 0.02)
        for index in range(len(ordered) - 1):
            left = ordered[index]
            right = ordered[index + 1]
            add(f"{left[0]} {right[0]}", float((left[1] + right[1]) / 2.0) + 0.02)

    if canonical_field in TEXT_LINE_PICK_FIELDS:
        longest = max(ordered, key=lambda item: (len(item[0]), item[1]))
        add(longest[0], longest[1] + 0.03)

    return list(hypotheses.items())


class PaddleOCRRecognizer:
    """Thin PaddleOCR adapter with shared output normalization."""

    def __init__(
        self,
        language: str = "vi",
        use_angle_cls: bool = False,
        show_log: bool = False,
        device: str | None = None,
    ) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self.device = _resolve_paddle_device(device)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        _configure_headless_matplotlib_backend()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is unavailable. Install paddleocr and paddlepaddle before running recognition."
            ) from exc

        kwargs: dict[str, Any] = {
            "lang": self.language,
            "device": self.device,
            "enable_mkldnn": False,
            "enable_hpi": False,
        }
        signature = None
        try:
            signature = inspect.signature(PaddleOCR.__init__)
        except (TypeError, ValueError):
            pass

        if signature is None or "show_log" in signature.parameters:
            kwargs["show_log"] = self.show_log
        if signature is None:
            kwargs["use_angle_cls"] = self.use_angle_cls
        elif "use_textline_orientation" in signature.parameters:
            kwargs["use_textline_orientation"] = self.use_angle_cls
        elif "use_angle_cls" in signature.parameters:
            kwargs["use_angle_cls"] = self.use_angle_cls

        for key, value in {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_angle_cls,
        }.items():
            if signature is not None and key in signature.parameters:
                kwargs[key] = value

        try:
            self._client = PaddleOCR(**kwargs)
        except ValueError as exc:
            if "Unknown argument: show_log" not in str(exc):
                raise
            kwargs.pop("show_log", None)
            self._client = PaddleOCR(**kwargs)
        return self._client

    def _predict_candidates(self, image: np.ndarray) -> list[tuple[str, float, np.ndarray | None, int]]:
        client = self._get_client()
        if hasattr(client, "predict"):
            kwargs: dict[str, Any] = {}
            signature = inspect.signature(client.predict)
            for key, value in {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": self.use_angle_cls,
                "return_word_box": False,
                "text_rec_score_thresh": 0.0,
            }.items():
                if key in signature.parameters:
                    kwargs[key] = value
            result = client.predict(image, **kwargs)
        else:
            result = client.ocr(image, det=True, rec=True, cls=self.use_angle_cls)
        return _flatten_candidates(result)

    def recognize(self, image: np.ndarray, field_name: str | None = None) -> OCRResult:
        if image is None or image.size == 0:
            return empty_ocr_result("paddleocr")

        try:
            prepared, scale, pad_x, pad_y = _prepare_image(image, field_name)
            candidates = self._predict_candidates(prepared)
        except Exception as exc:
            LOGGER.warning("PaddleOCR inference failed: %s", exc)
            result = empty_ocr_result("paddleocr")
            result.error_message = str(exc)
            return result

        if not candidates:
            return empty_ocr_result("paddleocr")

        segments = [_candidate_to_segment(candidate, scale, pad_x, pad_y, image.shape) for candidate in candidates]
        raw_text = _joined_text(candidates)

        best_text = ""
        best_score = 0.0
        for hypothesis_text, hypothesis_score in _build_hypotheses(candidates, field_name):
            cleaned_text = cleanup_ocr_text(hypothesis_text, field_name)
            calibrated = calibrate_ocr_confidence(cleaned_text, hypothesis_score, field_name)
            if calibrated > best_score or (calibrated == best_score and len(cleaned_text) > len(best_text)):
                best_text = cleaned_text
                best_score = calibrated

        if not best_text:
            return empty_ocr_result("paddleocr")

        normalized_text = normalize_text_for_field(best_text, field_name)
        return OCRResult(
            text=best_text,
            score=best_score,
            engine="paddleocr",
            raw={"text": raw_text, "candidates": [{"text": t, "score": s} for t, s, _, _ in candidates]},
            needs_review=best_score < 0.5 or normalized_text == "",
            normalized_text=normalized_text,
            segments=segments,
        )

    def predict(self, image: np.ndarray, field_name: str | None = None) -> OCRResult:
        return self.recognize(image, field_name=field_name)


PaddleOCRAdapter = PaddleOCRRecognizer

__all__ = ["PaddleOCRAdapter", "PaddleOCRRecognizer"]
