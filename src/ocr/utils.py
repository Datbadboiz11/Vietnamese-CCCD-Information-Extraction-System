from __future__ import annotations

import re
import unicodedata
from datetime import datetime

FIELD_NAME_MAP = {
    "id": "id_number",
    "name": "full_name",
    "birth": "date_of_birth",
    "origin": "place_of_origin",
    "address": "place_of_residence",
    "title": "title",
}

TARGET_FIELDS = {
    "id_number",
    "full_name",
    "date_of_birth",
    "place_of_origin",
    "place_of_residence",
}

DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d%m%Y",
)


def review_bucket(confidence: float) -> str:
    if confidence >= 0.9:
        return "accept"
    if confidence >= 0.5:
        return "review"
    return "reject"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_unicode(text)).strip()


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def normalize_date(text: str) -> str | None:
    candidate = collapse_whitespace(text)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_text_for_field(text: str, field_name: str | None) -> str:
    value = collapse_whitespace(text)
    if field_name == "id_number":
        return digits_only(value)
    if field_name == "date_of_birth":
        normalized = normalize_date(value)
        return normalized if normalized else value
    return value


def is_valid_id_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{12}", digits_only(text)))


def is_valid_date(text: str) -> bool:
    return normalize_date(text) is not None


def is_text_like(text: str) -> bool:
    if not text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(1, len(text) // 3)


def estimate_text_confidence(text: str, field_name: str | None) -> float:
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return 0.0

    confidence = 0.55
    if field_name == "id_number":
        confidence += min(len(digits_only(cleaned)) / 12.0, 1.0) * 0.25
        if is_valid_id_number(cleaned):
            confidence += 0.15
    elif field_name == "date_of_birth":
        confidence += min(len(cleaned) / 10.0, 1.0) * 0.10
        if is_valid_date(cleaned):
            confidence += 0.20
    else:
        alpha_ratio = sum(ch.isalpha() for ch in cleaned) / max(1, len(cleaned))
        confidence += min(alpha_ratio, 1.0) * 0.20
        confidence += min(len(cleaned) / 24.0, 1.0) * 0.15
        if is_text_like(cleaned):
            confidence += 0.05

    return max(0.0, min(confidence, 0.95))

__all__ = [
    "FIELD_NAME_MAP",
    "TARGET_FIELDS",
    "collapse_whitespace",
    "digits_only",
    "estimate_text_confidence",
    "is_text_like",
    "is_valid_date",
    "is_valid_id_number",
    "normalize_date",
    "normalize_text_for_field",
    "normalize_unicode",
    "review_bucket",
]
