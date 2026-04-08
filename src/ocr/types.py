from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OCRResult:
    text: str
    confidence: float
    backend: str
    needs_review: bool = False
    raw_text: str | None = None
    normalized_text: str | None = None


@dataclass
class EnsembleResult:
    text: str
    confidence: float
    source: str
    needs_review: bool
    candidates: dict[str, OCRResult] = field(default_factory=dict)
    normalized_text: str | None = None


__all__ = ["EnsembleResult", "OCRResult"]
