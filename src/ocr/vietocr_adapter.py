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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FINETUNE_V2_CONFIG = _PROJECT_ROOT / "configs" / "vietocr_finetune_v2.yml"
_FINETUNE_V2_WEIGHTS = _PROJECT_ROOT / "weights" / "vietocr_cccd_v2.pth"
_ADDRESS_V1_CONFIG = _PROJECT_ROOT / "configs" / "vietocr_address_v1.yml"
_ADDRESS_V1_WEIGHTS = _PROJECT_ROOT / "weights" / "vietocr_cccd_address_v1.pth"
_FINETUNE_CONFIG = _PROJECT_ROOT / "configs" / "vietocr_finetune.yml"
_FINETUNE_WEIGHTS = _PROJECT_ROOT / "weights" / "vietocr_cccd.pth"


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


def _patch_numpy_for_imgaug() -> None:
    if hasattr(np, "sctypes"):
        return
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }


class VietOCRRecognizer:
    """Thin VietOCR adapter with lazy model loading and safe failure handling."""

    def __init__(
        self,
        config_name: str = "vgg_transformer",
        device: str | None = None,
        finetuned: bool = True,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        model_label: str = "vietocr",
    ) -> None:
        self.config_name = config_name
        self.device = _resolve_torch_device(device)
        self.finetuned = finetuned
        self.config_path = Path(config_path) if config_path is not None else None
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.model_label = model_label
        self._predictor: Any | None = None

    @classmethod
    def address_v1(cls, device: str | None = None) -> "VietOCRRecognizer":
        return cls(
            device=device,
            config_path=_ADDRESS_V1_CONFIG,
            weights_path=_ADDRESS_V1_WEIGHTS,
            model_label="vietocr_address_v1",
        )

    def _load_predictor_from_files(self, Cfg: Any, Predictor: Any, config_path: Path, weights_path: Path) -> Any:
        LOGGER.info("Loading %s from %s", self.model_label, weights_path)
        config = Cfg.load_config_from_file(str(config_path))
        config["weights"] = str(weights_path)
        config["device"] = self.device
        config["predictor"]["beamsearch"] = True
        return Predictor(config)

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor

        _disable_broken_proxy_env()
        _patch_numpy_for_imgaug()

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError(
                "VietOCR is unavailable. Install VietOCR dependencies and model weights before running recognition."
            ) from exc

        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.LANCZOS

        if self.config_path is not None and self.weights_path is not None:
            if self.config_path.exists() and self.weights_path.exists():
                self._predictor = self._load_predictor_from_files(
                    Cfg,
                    Predictor,
                    self.config_path,
                    self.weights_path,
                )
                return self._predictor
            LOGGER.warning(
                "%s files not found: config=%s weights=%s; falling back to default VietOCR",
                self.model_label,
                self.config_path,
                self.weights_path,
            )

        if self.finetuned and _FINETUNE_V2_WEIGHTS.exists() and _FINETUNE_V2_CONFIG.exists():
            self._predictor = self._load_predictor_from_files(
                Cfg,
                Predictor,
                _FINETUNE_V2_CONFIG,
                _FINETUNE_V2_WEIGHTS,
            )
            return self._predictor

        if self.finetuned and _FINETUNE_WEIGHTS.exists() and _FINETUNE_CONFIG.exists():
            self._predictor = self._load_predictor_from_files(
                Cfg,
                Predictor,
                _FINETUNE_CONFIG,
                _FINETUNE_WEIGHTS,
            )
            return self._predictor

        config = Cfg.load_config_from_name(self.config_name)
        config["device"] = self.device
        config["predictor"]["beamsearch"] = True
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

        _is_place_field = field_name in {
            "place_of_origin", "place_of_residence", "origin", "address",
        }

        if _is_place_field:
            lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            l_ch = clahe.apply(l_ch)
            prepared = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            kernel = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5],
                               [0, -0.5, 0]], dtype=np.float32)
        else:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
        prepared = cv2.filter2D(prepared, -1, kernel)

        height, width = prepared.shape[:2]
        pad_x = max(8, int(round(width * 0.08)))
        pad_y = max(8, int(round(height * 0.12)))
        prepared = cv2.copyMakeBorder(
            prepared,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            borderType=cv2.BORDER_REPLICATE,
        )

        if field_name in {"id", "id_number", "birth", "date_of_birth"}:
            min_height = 64
        elif _is_place_field:
            min_height = 192
        else:
            min_height = 96
        scale = max(1.0, min_height / max(1, prepared.shape[0]))
        if scale >= 1.5 and _is_place_field:
            from src.preprocessing.super_resolution import super_resolve
            prepared = super_resolve(prepared, min_height=min_height)
            scale = max(1.0, min_height / max(1, prepared.shape[0]))
        if scale > 1.0:
            interp = cv2.INTER_LANCZOS4 if scale >= 2.0 else cv2.INTER_CUBIC
            prepared = cv2.resize(prepared, None, fx=scale, fy=scale, interpolation=interp)
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
            result = empty_ocr_result(self.model_label)
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
            result = empty_ocr_result(self.model_label)
            result.error_message = str(exc)
            return result

        raw_text, raw_score, raw_payload = self._parse_prediction(prediction, field_name)
        cleaned_text = cleanup_ocr_text(raw_text, field_name)
        calibrated_score = calibrate_ocr_confidence(cleaned_text, raw_score, field_name)
        normalized_text = normalize_text_for_field(cleaned_text, field_name)
        return OCRResult(
            text=cleaned_text,
            score=calibrated_score,
            engine=self.model_label,
            raw={**raw_payload, "text": raw_text, "score": raw_score},
            needs_review=calibrated_score < 0.5 or normalized_text == "",
            normalized_text=normalized_text,
        )

    def predict(self, image: np.ndarray, field_name: str | None = None) -> OCRResult:
        return self.recognize(image, field_name=field_name)


VietOCRAdapter = VietOCRRecognizer

__all__ = ["VietOCRAdapter", "VietOCRRecognizer"]
