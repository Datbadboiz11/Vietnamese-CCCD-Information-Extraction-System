from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from src.ocr.types import OCRResult

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

TEXT_LINE_PICK_FIELDS = {"full_name", "place_of_origin", "place_of_residence"}

DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d%m%Y",
)

FIELD_LABEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "full_name": (
        "ho va ten",
        "ho ten",
        "full name",
        "surname and name",
    ),
    "place_of_origin": (
        "nguyen quan",
        "que quan",
        "place of origin",
        "origin",
    ),
    "place_of_residence": (
        "noi dkhk thuong tru",
        "noi dk hk thuong tru",
        "noi thuong tru",
        "noi cu tru",
        "thuong tru",
        "dia chi",
        "place of residence",
        "residence",
        "address",
    ),
}

FIELD_LABEL_HINTS: dict[str, tuple[str, ...]] = {
    "full_name": ("full", "name", "ho", "ten"),
    "place_of_origin": ("nguyen", "que", "quan", "origin", "place"),
    "place_of_residence": ("noi", "thuong", "tru", "cu", "residen", "address", "dkhk"),
}

DATE_PATTERN = re.compile(r"(?<!\d)(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})(?!\d)")
ID_PATTERN = re.compile(r"(?:\d[\s\-./]?){9,14}\d")


def review_bucket(confidence: float) -> str:
    if confidence >= 0.9:
        return "accept"
    if confidence >= 0.5:
        return "review"
    return "reject"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def canonicalize_field_name(field_name: str | None) -> str | None:
    if field_name is None:
        return None
    return FIELD_NAME_MAP.get(field_name, field_name)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_unicode(text)).strip()


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", collapse_whitespace(text).lower())
    folded = folded.replace("đ", "d").replace("Ä‘", "d")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9/: ]+", " ", folded)


def _tokens_match(observed: str, expected: str) -> bool:
    if observed == expected:
        return True
    if len(observed) >= 3 and len(expected) >= 3:
        return observed.startswith(expected) or expected.startswith(observed)
    return False


def _starts_with_label_variant(text: str, field_name: str | None) -> tuple[bool, int]:
    field_name = canonicalize_field_name(field_name)
    variants = FIELD_LABEL_VARIANTS.get(field_name or "", ())
    tokens = _ascii_fold(text).split()
    if not tokens:
        return False, 0

    for variant in sorted(variants, key=lambda value: len(value.split()), reverse=True):
        variant_tokens = variant.split()
        if len(tokens) < len(variant_tokens):
            continue
        if all(_tokens_match(tokens[index], expected) for index, expected in enumerate(variant_tokens)):
            return True, len(variant_tokens)
    return False, 0


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
    field_name = canonicalize_field_name(field_name)
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


def _strip_label_prefix_tokens(text: str, field_name: str | None) -> str:
    matched, token_count = _starts_with_label_variant(text, field_name)
    if not matched:
        return collapse_whitespace(text)

    original_tokens = collapse_whitespace(text).split()
    if len(original_tokens) <= token_count:
        return ""
    return collapse_whitespace(" ".join(original_tokens[token_count:])).lstrip(":/-.,; ")


def strip_known_field_prefix(text: str, field_name: str | None) -> str:
    field_name = canonicalize_field_name(field_name)
    cleaned = collapse_whitespace(text)
    if not cleaned or field_name not in FIELD_LABEL_VARIANTS:
        return cleaned

    _is_place_field = field_name in ("place_of_origin", "place_of_residence")

    for _ in range(4):
        previous = cleaned
        if ":" in cleaned:
            left, right = cleaned.split(":", 1)
            # For place fields, any short prefix (≤ 6 words) before a colon is treated as a
            # label — handles garbled OCR like "Nos mutng te:" instead of "Nơi thường trú:"
            short_label = _is_place_field and 1 <= len(left.split()) <= 6
            if looks_like_label_text(left, field_name) or short_label:
                candidate = collapse_whitespace(right).lstrip(":/-.,; ")
                if candidate:
                    cleaned = candidate
                    continue
        token_stripped = _strip_label_prefix_tokens(cleaned, field_name)
        if token_stripped != cleaned:
            cleaned = token_stripped
            continue
        if cleaned == previous:
            break
    return collapse_whitespace(cleaned)


def pick_id_number_candidate(text: str) -> str | None:
    exact_match: str | None = None
    fallback_match: str | None = None
    for match in ID_PATTERN.finditer(text):
        digits = digits_only(match.group(0))
        if len(digits) == 12:
            exact_match = digits
            break
        if 9 <= len(digits) <= 14 and (fallback_match is None or len(digits) > len(fallback_match)):
            fallback_match = digits
    return exact_match or fallback_match


def pick_date_candidate(text: str) -> str | None:
    matches = [collapse_whitespace(match.group(1)) for match in DATE_PATTERN.finditer(text)]
    if not matches:
        return None
    matches.sort(key=lambda value: (normalize_date(value) is not None, len(value)), reverse=True)
    return matches[0]


def extract_value_from_label_text(text: str, field_name: str | None) -> str:
    field_name = canonicalize_field_name(field_name)
    cleaned = collapse_whitespace(text)
    if not cleaned or field_name not in TEXT_LINE_PICK_FIELDS:
        return cleaned

    if ":" in cleaned:
        left, right = cleaned.split(":", 1)
        if looks_like_label_text(left, field_name):
            candidate = collapse_whitespace(right).lstrip(":/-.,; ")
            return strip_known_field_prefix(candidate, field_name)

    stripped = strip_known_field_prefix(cleaned, field_name)
    if stripped != cleaned:
        return stripped
    return cleaned


def cleanup_ocr_text(text: str, field_name: str | None) -> str:
    field_name = canonicalize_field_name(field_name)
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return ""

    if field_name == "id_number":
        return pick_id_number_candidate(cleaned) or cleaned
    if field_name == "date_of_birth":
        return pick_date_candidate(cleaned) or cleaned

    extracted = extract_value_from_label_text(cleaned, field_name)
    if extracted == cleaned and looks_like_label_text(cleaned, field_name) and cleaned.rstrip().endswith(":"):
        return ""
    return collapse_whitespace(extracted)


def looks_like_label_text(text: str, field_name: str | None) -> bool:
    field_name = canonicalize_field_name(field_name)
    cleaned = collapse_whitespace(text)
    if not cleaned or field_name not in FIELD_LABEL_HINTS:
        return False

    folded = _ascii_fold(cleaned)
    tokens = folded.split()
    hints = FIELD_LABEL_HINTS[field_name]
    hint_hits = sum(hint in folded for hint in hints)
    variant_match, _ = _starts_with_label_variant(cleaned, field_name)

    if variant_match and len(tokens) <= 10:
        return True
    if cleaned.rstrip().endswith(":") and hint_hits >= 1:
        return True
    if hint_hits >= 3 and len(tokens) <= 10:
        return True
    if hint_hits >= 2 and len(tokens) <= 6:
        return True
    return False


def is_text_like(text: str) -> bool:
    if not text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(1, len(text) // 3)


def is_digit_heavy_text(text: str, threshold: float = 0.45) -> bool:
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return False
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    if alnum_count == 0:
        return False
    digit_ratio = sum(ch.isdigit() for ch in cleaned) / alnum_count
    return digit_ratio >= threshold


def is_repetitive_text(text: str, min_tokens: int = 6) -> bool:
    tokens = collapse_whitespace(text).lower().split()
    if len(tokens) < min_tokens:
        return False
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    repeated_bigrams = (len(tokens) - 1) != len(set(zip(tokens, tokens[1:])))
    return unique_ratio <= 0.65 or repeated_bigrams


def looks_suspicious_for_field(text: str, field_name: str | None) -> bool:
    canonical_field = canonicalize_field_name(field_name)
    cleaned = cleanup_ocr_text(text, canonical_field)
    if not cleaned:
        return True
    if canonical_field in {"place_of_origin", "place_of_residence"}:
        tokens = cleaned.split()
        if len(tokens) == 1:
            return True
        if len(tokens) <= 2 and "," not in cleaned and len(cleaned) < 12:
            return True
        if is_digit_heavy_text(cleaned, threshold=0.35):
            return True
        if is_repetitive_text(cleaned):
            return True
        if len(cleaned) >= 48 and len(cleaned.split()) <= 3:
            return True
    if canonical_field == "full_name":
        if is_digit_heavy_text(cleaned, threshold=0.20):
            return True
        if is_repetitive_text(cleaned, min_tokens=4):
            return True
    return False


def estimate_text_confidence(text: str, field_name: str | None) -> float:
    field_name = canonicalize_field_name(field_name)
    cleaned = cleanup_ocr_text(text, field_name)
    if not cleaned:
        return 0.0

    confidence = 0.45
    if field_name == "id_number":
        confidence += min(len(digits_only(cleaned)) / 12.0, 1.0) * 0.25
        if is_valid_id_number(cleaned):
            confidence += 0.25
    elif field_name == "date_of_birth":
        confidence += min(len(cleaned) / 10.0, 1.0) * 0.10
        if is_valid_date(cleaned):
            confidence += 0.25
    else:
        letters = sum(ch.isalpha() for ch in cleaned)
        digits = sum(ch.isdigit() for ch in cleaned)
        punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in cleaned)
        alpha_ratio = letters / max(1, len(cleaned))
        digit_ratio = digits / max(1, len(cleaned))
        punct_ratio = punctuation / max(1, len(cleaned))
        confidence += min(alpha_ratio, 1.0) * 0.25
        confidence += min(len(cleaned.split()) / 4.0, 1.0) * 0.10
        confidence += min(len(cleaned) / 24.0, 1.0) * 0.10
        confidence -= min(digit_ratio * 0.20, 0.10)
        confidence -= min(punct_ratio * 0.20, 0.10)
        if looks_like_label_text(cleaned, field_name):
            confidence -= 0.25
        if is_text_like(cleaned):
            confidence += 0.05
        if field_name in {"place_of_origin", "place_of_residence"}:
            if is_digit_heavy_text(cleaned, threshold=0.35):
                confidence -= 0.35
            if is_repetitive_text(cleaned):
                confidence -= 0.25
            if len(cleaned) >= 48 and len(cleaned.split()) <= 3:
                confidence -= 0.15
        elif field_name == "full_name":
            if is_digit_heavy_text(cleaned, threshold=0.20):
                confidence -= 0.25
            if is_repetitive_text(cleaned, min_tokens=4):
                confidence -= 0.20

    return max(0.0, min(confidence, 0.95))


def calibrate_ocr_confidence(text: str, raw_confidence: float, field_name: str | None) -> float:
    field_name = canonicalize_field_name(field_name)
    cleaned = cleanup_ocr_text(text, field_name)
    try:
        raw = float(raw_confidence)
    except (TypeError, ValueError):
        raw = 0.0
    raw = max(0.0, min(raw, 1.0))

    if not cleaned:
        return 0.0

    if field_name == "id_number":
        if is_valid_id_number(cleaned):
            return max(raw, 0.88)
        digit_count = len(digits_only(cleaned))
        if digit_count >= 10:
            return max(0.0, min(raw * 0.75, 0.84))
        if digit_count >= 8:
            return max(0.0, min(raw * 0.55, 0.72))
        return raw * 0.25

    if field_name == "date_of_birth":
        if is_valid_date(cleaned):
            return max(raw, 0.8)
        digit_count = len(digits_only(cleaned))
        if digit_count >= 6:
            return max(0.0, min(raw * 0.65, 0.74))
        return raw * 0.30

    quality = estimate_text_confidence(cleaned, field_name)
    if looks_like_label_text(cleaned, field_name):
        raw *= 0.5
    if field_name in {"place_of_origin", "place_of_residence"} and looks_suspicious_for_field(cleaned, field_name):
        raw *= 0.45
    if field_name == "full_name" and looks_suspicious_for_field(cleaned, field_name):
        raw *= 0.60
    return max(0.0, min(0.7 * raw + 0.3 * quality, 0.98))


def empty_ocr_result(backend: str) -> OCRResult:
    return OCRResult(
        text="",
        score=0.0,
        engine=backend,
        raw={"text": ""},
        needs_review=True,
        normalized_text="",
        segments=[],
    )


__all__ = [
    "FIELD_NAME_MAP",
    "TARGET_FIELDS",
    "TEXT_LINE_PICK_FIELDS",
    "calibrate_ocr_confidence",
    "canonicalize_field_name",
    "cleanup_ocr_text",
    "collapse_whitespace",
    "digits_only",
    "empty_ocr_result",
    "estimate_text_confidence",
    "extract_value_from_label_text",
    "is_digit_heavy_text",
    "is_repetitive_text",
    "is_text_like",
    "is_valid_date",
    "is_valid_id_number",
    "looks_like_label_text",
    "looks_suspicious_for_field",
    "normalize_date",
    "normalize_text_for_field",
    "normalize_unicode",
    "pick_date_candidate",
    "pick_id_number_candidate",
    "review_bucket",
    "strip_known_field_prefix",
]
