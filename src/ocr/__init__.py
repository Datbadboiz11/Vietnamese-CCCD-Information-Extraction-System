from .vietocr_adapter import VietOCRAdapter, OCRResult
from .paddleocr_adapter import PaddleOCRAdapter
from .ensemble import OCREnsemble, EnsembleResult

__all__ = [
    "VietOCRAdapter",
    "OCRResult",
    "PaddleOCRAdapter",
    "OCREnsemble",
    "EnsembleResult",
]
