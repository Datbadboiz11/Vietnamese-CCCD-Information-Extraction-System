"""
src/parsing — Field mapping, validation, confidence routing.

Exports:
    CCCDParser      - parse + validate OCR fields → ParsedInfo
    ParsedInfo      - dataclass kết quả cuối
    FieldResult     - dataclass kết quả 1 field
    ConfidenceRouter - phân loại accept/review/reject theo ngưỡng confidence
"""
from __future__ import annotations

from .validators import CCCDParser, FieldResult, ParsedInfo

__all__ = [
    "CCCDParser",
    "ConfidenceRouter",
    "FieldResult",
    "ParsedInfo",
]


class ConfidenceRouter:
    """
    Phân loại OCR result theo ngưỡng confidence.

    Thresholds mặc định (theo PLAN.md):
        accept : score >= accept_threshold (0.8)
        review : accept_threshold > score >= review_threshold (0.5)
        reject : score < review_threshold

    Usage::
        router = ConfidenceRouter()
        bucket = router.route(0.73)  # → "review"
        bucket = router.route(0.91)  # → "accept"
    """

    def __init__(
        self,
        accept_threshold: float = 0.8,
        review_threshold: float = 0.5,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold

    def route(self, confidence: float) -> str:
        """Trả về "accept", "review", hoặc "reject"."""
        if confidence >= self.accept_threshold:
            return "accept"
        if confidence >= self.review_threshold:
            return "review"
        return "reject"

    def route_result(self, field_result: FieldResult) -> str:
        """Route dựa trên FieldResult, ưu tiên review_reason nếu có."""
        if field_result.review_reason:
            return "review"
        return self.route(field_result.confidence)

    def route_batch(self, parsed_info: ParsedInfo) -> dict[str, str]:
        """Trả về dict {field_name: bucket} cho tất cả field."""
        return {
            fname: self.route_result(fresult)
            for fname, fresult in parsed_info.field_results.items()
        }
