from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRSegment:
    """A text segment detected by an OCR engine."""

    text: str
    score: float
    box: list[list[float]] | None = None

    @property
    def confidence(self) -> float:
        return float(self.score)


@dataclass
class OCRResult:
    """Unified OCR output shared across recognizers and ensemble logic."""

    text: str
    score: float
    engine: str
    raw: dict[str, Any] | None = None
    needs_review: bool = False
    error_message: str | None = None
    normalized_text: str | None = None
    segments: list[OCRSegment] | None = None

    @property
    def confidence(self) -> float:
        return float(self.score)

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.score = float(value)

    @property
    def backend(self) -> str:
        return self.engine

    @backend.setter
    def backend(self, value: str) -> None:
        self.engine = value

    @property
    def raw_text(self) -> str:
        if not self.raw:
            return ""
        value = self.raw.get("text", "")
        return "" if value is None else str(value)

    @raw_text.setter
    def raw_text(self, value: str) -> None:
        payload = dict(self.raw or {})
        payload["text"] = value
        self.raw = payload


@dataclass
class EnsembleResult:
    """Final selection returned by OCR ensemble logic."""

    text: str
    score: float
    engine: str
    needs_review: bool
    candidates: dict[str, OCRResult] = field(default_factory=dict)
    normalized_text: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def confidence(self) -> float:
        return float(self.score)

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.score = float(value)

    @property
    def source(self) -> str:
        return self.engine

    @source.setter
    def source(self, value: str) -> None:
        self.engine = value


__all__ = ["EnsembleResult", "OCRResult", "OCRSegment"]
