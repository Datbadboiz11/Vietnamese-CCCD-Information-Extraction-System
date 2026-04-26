"""
CCCD field validators, auto-correctors, and parser.

Trách nhiệm:
- Map class YOLO (id, name, birth, origin, address) → field chuẩn
- Validate từng field (regex, format date, ...)
- Auto-correct lỗi OCR phổ biến (O→0, l→1 trong id_number, v.v.)
- Trả về ParsedInfo dataclass thống nhất
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.ocr.utils import (
    canonicalize_field_name,
    collapse_whitespace,
    digits_only,
    is_valid_date,
    is_valid_id_number,
    normalize_date,
    normalize_text_for_field,
    strip_known_field_prefix,
)

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FieldResult:
    """Kết quả sau khi validate + normalize 1 field."""

    field_name: str
    raw_text: str
    value: str | None
    confidence: float
    is_valid: bool
    auto_corrected: bool
    warning: str | None
    review_reason: str | None


@dataclass
class ParsedInfo:
    """Thông tin CCCD đã parse và validate đầy đủ."""

    id_number: str | None = None
    full_name: str | None = None
    date_of_birth: str | None = None
    place_of_origin: str | None = None
    place_of_residence: str | None = None

    # Metadata per-field
    field_results: dict[str, FieldResult] = field(default_factory=dict)

    # Confidence score per field (field_name → score)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    # Validation messages
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)

    # Review flags
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id_number": self.id_number,
            "full_name": self.full_name,
            "date_of_birth": self.date_of_birth,
            "place_of_origin": self.place_of_origin,
            "place_of_residence": self.place_of_residence,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "confidence_scores": self.confidence_scores,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
        }


# ---------------------------------------------------------------------------
# OCR auto-correct lookup tables
# ---------------------------------------------------------------------------

# Ký tự OCR hay nhầm trong field số (id_number, date_of_birth)
_OCR_DIGIT_SUBSTITUTIONS: dict[str, str] = {
    "O": "0",
    "o": "0",
    "D": "0",
    "I": "1",
    "l": "1",
    "i": "1",
    "Z": "2",
    "z": "2",
    "A": "4",
    "S": "5",
    "s": "5",
    "G": "6",
    "b": "6",
    "B": "8",
    "q": "9",
    "g": "9",
}

# Tiền tố mã tỉnh/thành phố trong id_number (2 chữ số đầu)
# Dùng để soft-warn nếu prefix lạ
_VALID_PROVINCE_PREFIXES = {
    "001", "002", "004", "006", "008", "010", "011", "012", "014", "015",
    "017", "019", "020", "022", "024", "025", "026", "027", "030", "031",
    "033", "034", "035", "036", "037", "038", "040", "042", "044", "045",
    "046", "048", "049", "051", "052", "054", "056", "058", "060", "062",
    "064", "066", "067", "068", "070", "072", "074", "075", "077", "079",
    "080", "082", "083", "084", "086", "087", "089", "091", "092", "093",
    "094", "095", "096",
}


def _auto_correct_id(text: str) -> tuple[str, bool]:
    """Thay thế ký tự OCR lỗi trong id_number."""
    result = []
    changed = False
    for ch in text:
        if ch.isdigit():
            result.append(ch)
        elif ch in _OCR_DIGIT_SUBSTITUTIONS:
            result.append(_OCR_DIGIT_SUBSTITUTIONS[ch])
            changed = True
        # Bỏ ký tự không phải số (khoảng trắng, dấu gạch, ...)
        elif ch in " -./":
            pass  # skip separators
        else:
            result.append(ch)
    return "".join(result), changed


def _normalize_name(text: str) -> str:
    """Chuẩn hóa tên: NFC, strip, collapse whitespace, upper-case đầu từ."""
    value = unicodedata.normalize("NFC", collapse_whitespace(text))
    # Giữ nguyên casing gốc (có thể đã all-caps từ CCCD)
    return value


def _has_vietnamese_chars(text: str) -> bool:
    vietnamese_pattern = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    )
    return bool(vietnamese_pattern.search(text))


# ---------------------------------------------------------------------------
# Individual field validators
# ---------------------------------------------------------------------------

def _validate_id_number(raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), "id_number")
    corrected_text, auto_corrected = _auto_correct_id(text)
    value = digits_only(corrected_text)

    warning: str | None = None
    review_reason: str | None = None

    if not is_valid_id_number(value):
        # Cố gắng tìm chuỗi 12 chữ số trong text
        match = re.search(r"\d{12}", value)
        if match:
            value = match.group(0)
        else:
            value = value if value else None
            return FieldResult(
                field_name="id_number",
                raw_text=raw_text,
                value=value,
                confidence=confidence,
                is_valid=False,
                auto_corrected=auto_corrected,
                warning="id_number không đủ 12 chữ số",
                review_reason="invalid_id_format",
            )

    # Soft-check tiền tố mã tỉnh (3 số đầu)
    if value and value[:3] not in _VALID_PROVINCE_PREFIXES:
        warning = f"Tiền tố mã tỉnh '{value[:3]}' không nhận ra — có thể sai"

    if confidence < 0.5:
        review_reason = "low_confidence"

    return FieldResult(
        field_name="id_number",
        raw_text=raw_text,
        value=value,
        confidence=confidence,
        is_valid=True,
        auto_corrected=auto_corrected,
        warning=warning,
        review_reason=review_reason,
    )


def _validate_full_name(raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), "full_name")
    value = _normalize_name(text) if text else None

    warning: str | None = None
    review_reason: str | None = None
    is_valid = True

    if not value:
        return FieldResult(
            field_name="full_name",
            raw_text=raw_text,
            value=None,
            confidence=confidence,
            is_valid=False,
            auto_corrected=False,
            warning="Tên trống",
            review_reason="empty_value",
        )

    # Tên phải có ít nhất 2 từ
    words = value.split()
    if len(words) < 2:
        warning = "Tên có thể thiếu (ít hơn 2 từ)"
        review_reason = "short_name"

    # Tên không nên chứa số
    if re.search(r"\d", value):
        warning = (warning or "") + " | Tên chứa chữ số — có thể OCR lỗi"
        review_reason = review_reason or "name_contains_digits"
        is_valid = False

    if confidence < 0.5:
        review_reason = review_reason or "low_confidence"

    return FieldResult(
        field_name="full_name",
        raw_text=raw_text,
        value=value,
        confidence=confidence,
        is_valid=is_valid,
        auto_corrected=False,
        warning=warning,
        review_reason=review_reason,
    )


def _validate_date_of_birth(raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), "date_of_birth")

    # Auto-correct ký tự OCR lỗi trong phần số
    corrected, auto_corrected = _auto_correct_id(text)
    # Khôi phục dấu phân cách nếu bị mất
    digits = digits_only(corrected)

    # Nếu chỉ có 8 chữ số → ghép lại thành DD/MM/YYYY
    normalized_value: str | None = None
    if len(digits) == 8:
        normalized_value = normalize_date(f"{digits[:2]}/{digits[2:4]}/{digits[4:]}")
    
    if normalized_value is None:
        normalized_value = normalize_date(text)

    if normalized_value is None:
        return FieldResult(
            field_name="date_of_birth",
            raw_text=raw_text,
            value=text if text else None,
            confidence=confidence,
            is_valid=False,
            auto_corrected=auto_corrected,
            warning="Không parse được ngày sinh — format không nhận ra",
            review_reason="invalid_date_format",
        )

    # normalized_value is ISO YYYY-MM-DD; convert to display format DD/MM/YYYY
    # e.g. "1989-07-09" → "09/07/1989"
    display_value = (
        normalized_value[8:10] + "/" + normalized_value[5:7] + "/" + normalized_value[0:4]
    )

    # Soft-check năm hợp lý (dùng ISO để tách năm dễ)
    warning: str | None = None
    review_reason: str | None = None
    year = int(normalized_value[:4])
    if not (1900 <= year <= 2025):
        warning = f"Năm sinh {year} ngoài khoảng 1900–2025 — kiểm tra lại"
        review_reason = "suspicious_year"

    if confidence < 0.5:
        review_reason = review_reason or "low_confidence"

    return FieldResult(
        field_name="date_of_birth",
        raw_text=raw_text,
        value=display_value,
        confidence=confidence,
        is_valid=True,
        auto_corrected=auto_corrected,
        warning=warning,
        review_reason=review_reason,
    )


def _validate_place(field_name: str, raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), field_name)
    value = unicodedata.normalize("NFC", text) if text else None

    warning: str | None = None
    review_reason: str | None = None
    is_valid = True

    if not value:
        return FieldResult(
            field_name=field_name,
            raw_text=raw_text,
            value=None,
            confidence=confidence,
            is_valid=False,
            auto_corrected=False,
            warning="Địa chỉ trống",
            review_reason="empty_value",
        )

    # Địa chỉ VN thường có tiếng Việt hoặc dài > 5 ký tự
    if len(value) < 5:
        warning = "Địa chỉ ngắn bất thường — có thể OCR thiếu"
        review_reason = "short_address"

    if confidence < 0.5:
        review_reason = review_reason or "low_confidence"

    return FieldResult(
        field_name=field_name,
        raw_text=raw_text,
        value=value,
        confidence=confidence,
        is_valid=is_valid,
        auto_corrected=False,
        warning=warning,
        review_reason=review_reason,
    )


# ---------------------------------------------------------------------------
# CCCDParser
# ---------------------------------------------------------------------------

# Map từ class YOLO → tên hàm validator
_FIELD_VALIDATORS = {
    "id_number": _validate_id_number,
    "full_name": _validate_full_name,
    "date_of_birth": _validate_date_of_birth,
    "place_of_origin": lambda t, c: _validate_place("place_of_origin", t, c),
    "place_of_residence": lambda t, c: _validate_place("place_of_residence", t, c),
}


class CCCDParser:
    """
    Parse và validate các field OCR từ CCCD.

    Usage::
        parser = CCCDParser()
        ocr_list = [
            {"class": "id", "text": "079...", "confidence": 0.92},
            {"class": "name", "text": "NGUYEN VAN A", "confidence": 0.87},
            ...
        ]
        info: ParsedInfo = parser.parse_batch(ocr_list)
    """

    def parse_field(
        self,
        class_name: str,
        text: str,
        confidence: float = 1.0,
    ) -> FieldResult | None:
        """Validate 1 field đơn. Trả về None nếu class không nhận ra."""
        canonical = canonicalize_field_name(class_name)
        if canonical is None or canonical not in _FIELD_VALIDATORS:
            return None
        validator = _FIELD_VALIDATORS[canonical]
        return validator(text or "", float(confidence))

    def parse_batch(
        self,
        ocr_list: list[dict[str, Any]],
    ) -> ParsedInfo:
        """
        Parse danh sách OCR results thành ParsedInfo.

        Mỗi phần tử trong ocr_list cần có:
            - "class": str  (YOLO class name)
            - "text": str
            - "confidence": float (optional, default 1.0)
        """
        info = ParsedInfo()

        for item in ocr_list:
            class_name = str(item.get("class") or item.get("field_name") or "")
            text = str(item.get("text") or "")
            confidence = float(item.get("confidence", 1.0) or 1.0)

            result = self.parse_field(class_name, text, confidence)
            if result is None:
                continue

            info.field_results[result.field_name] = result

            # Gán vào slot tương ứng
            if result.field_name == "id_number":
                info.id_number = result.value
            elif result.field_name == "full_name":
                info.full_name = result.value
            elif result.field_name == "date_of_birth":
                info.date_of_birth = result.value
            elif result.field_name == "place_of_origin":
                info.place_of_origin = result.value
            elif result.field_name == "place_of_residence":
                info.place_of_residence = result.value

            # Confidence scores
            info.confidence_scores[result.field_name] = result.confidence

            # Validation errors / warnings
            if not result.is_valid and result.warning:
                info.validation_errors.append(f"{result.field_name}: {result.warning}")
            elif result.warning:
                info.validation_warnings.append(f"{result.field_name}: {result.warning}")

            # Review reasons
            if result.review_reason:
                info.review_reasons.append(f"{result.field_name}: {result.review_reason}")

        info.needs_review = len(info.review_reasons) > 0
        return info
