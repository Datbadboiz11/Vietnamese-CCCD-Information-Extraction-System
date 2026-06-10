"""
CCCD field validators, auto-correctors, and parser.

Trách nhiệm:
- Map class YOLO (id, name, birth, origin, address) → field chuẩn
- Validate từng field (regex, format date, ...)
- Auto-correct lỗi OCR phổ biến (O→0, l→1 trong id_number, v.v.)
- Trả về ParsedInfo dataclass thống nhất
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.ocr.utils import (
    canonicalize_field_name,
    collapse_whitespace,
    digits_only,
    is_digit_heavy_text,
    is_valid_date,
    is_valid_id_number,
    looks_suspicious_for_field,
    normalize_date,
    normalize_text_for_field,
    strip_known_field_prefix,
)


def _ascii_fold_compare(text: str) -> str:
    folded = unicodedata.normalize("NFD", collapse_whitespace(text).lower())
    folded = folded.replace("đ", "d").replace("Đ", "d")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9, ]+", "", folded).strip()


_ADMIN_PREFIX_STRIP_RE = re.compile(
    r"\b(?:xa|phuong|thi tran|huyen|quan|thanh pho|tinh|thi xa)\s+",
)


def _ascii_fold_no_admin(text: str) -> str:
    folded = _ascii_fold_compare(text)
    return _ADMIN_PREFIX_STRIP_RE.sub("", folded).strip()


def _is_mostly_diacritic_change(original: str, corrected: str) -> bool:
    """True when the correction primarily restores Vietnamese diacritics.

    Also accepts changes that add administrative prefixes (Huyện, Xã, etc.)
    since these are safe gazetteer-driven corrections.
    """
    orig_folded = _ascii_fold_compare(original)
    corr_folded = _ascii_fold_compare(corrected)
    if not orig_folded or not corr_folded:
        return False
    if orig_folded == corr_folded:
        return True

    orig_no_admin = _ascii_fold_no_admin(original)
    corr_no_admin = _ascii_fold_no_admin(corrected)
    if orig_no_admin == corr_no_admin:
        return True

    return False

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
        elif ch in " -./":
            pass
        else:
            result.append(ch)
    return "".join(result), changed


def _validate_cccd_structure(id_number: str) -> list[str]:
    """Validate the internal structure of a 12-digit CCCD number.

    Returns a list of warnings (empty if structure is valid).
    Structure: PPP G YY NNNNNN
      PPP = province code (3 digits)
      G   = gender + century (0-9)
      YY  = last 2 digits of birth year
      NNNNNN = random sequence
    """
    warnings = []
    if len(id_number) != 12:
        return warnings

    province = id_number[:3]
    gender_century = int(id_number[3])

    if province not in _VALID_PROVINCE_PREFIXES:
        warnings.append(f"province_prefix_{province}")

    if gender_century not in range(10):
        warnings.append("invalid_gender_century")

    return warnings


def _cross_validate_id_dob(id_number: str, dob_value: str | None) -> str | None:
    """Check if CCCD birth-year digits match date_of_birth.

    Returns a warning string if they disagree, None if consistent.
    """
    if not dob_value or len(id_number) != 12:
        return None

    id_birth_year_suffix = id_number[4:6]

    dob_digits = re.sub(r"\D", "", dob_value)
    dob_year: str | None = None
    if len(dob_digits) == 8:
        dob_year = dob_digits[4:8]
    else:
        m = re.search(r"(\d{4})", dob_value)
        if m:
            dob_year = m.group(1)

    if dob_year and len(dob_year) == 4:
        expected_suffix = dob_year[2:4]
        if id_birth_year_suffix != expected_suffix:
            return (
                f"CCCD birth-year digits '{id_birth_year_suffix}' "
                f"disagree with DOB year '{dob_year}'"
            )
    return None


_COMMON_VN_SURNAMES = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ",
    "đặng", "bùi", "đỗ", "hồ", "ngô", "dương", "lý", "đào", "đinh",
    "lâm", "tạ", "trịnh", "mai", "tô", "châu", "lưu", "hà", "cao",
    "từ", "la", "thái", "quách", "kiều", "tăng", "triệu", "lương",
    "nghiêm", "vương", "nông", "đoàn", "giáp", "lục", "mã", "tống",
    "phùng", "khổng", "trương", "doãn", "quản",
    "nguyen", "tran", "le", "pham", "hoang", "huynh", "phan", "vu", "vo",
    "dang", "bui", "do", "ho", "ngo", "duong", "ly", "dao", "dinh",
}


def _normalize_name(text: str) -> str:
    value = unicodedata.normalize("NFC", collapse_whitespace(text))
    return value


def _has_vietnamese_chars(text: str) -> bool:
    vietnamese_pattern = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    )
    return bool(vietnamese_pattern.search(text))


def _looks_like_vietnamese_name(text: str) -> bool:
    words = collapse_whitespace(text).lower().split()
    if len(words) < 2:
        return False
    folded_first = _ascii_fold_compare(words[0])
    return folded_first in _COMMON_VN_SURNAMES


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
        match12 = re.search(r"\d{12}", value)
        match9 = re.search(r"\d{9}", value)
        if match12:
            value = match12.group(0)
        elif match9:
            value = match9.group(0)
        else:
            value = value if value else None
            return FieldResult(
                field_name="id_number",
                raw_text=raw_text,
                value=value,
                confidence=confidence,
                is_valid=False,
                auto_corrected=auto_corrected,
                warning="ID không đủ 9 hoặc 12 chữ số",
                review_reason="invalid_id_format",
            )

    if len(value) == 12:
        structure_warnings = _validate_cccd_structure(value)
        if structure_warnings:
            warning = f"CCCD structure: {', '.join(structure_warnings)}"
            review_reason = "suspicious_cccd_structure"
    elif len(value) == 9:
        pass

    if confidence < 0.5:
        review_reason = review_reason or "low_confidence"

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

    words = value.split()
    if len(words) < 2:
        warning = "Tên có thể thiếu (ít hơn 2 từ)"
        review_reason = "short_name"

    if re.search(r"\d", value):
        warning = (warning or "") + " | Tên chứa chữ số — có thể OCR lỗi"
        review_reason = review_reason or "name_contains_digits"
        is_valid = False

    if len(words) >= 2 and not _looks_like_vietnamese_name(value):
        alpha_ratio = sum(ch.isalpha() for ch in value) / max(1, len(value))
        if alpha_ratio < 0.7:
            warning = (warning or "") + " | Tên không giống tên người Việt"
            review_reason = review_reason or "unlikely_name"
        elif not _has_vietnamese_chars(value) and confidence < 0.85:
            review_reason = review_reason or "no_vn_chars_in_name"

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


_DATE_OCR_SUBSTITUTIONS: dict[str, str] = {
    "O": "0", "o": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "L": "1", "F": "1", "f": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "q": "9", "g": "9",
}


def _auto_correct_date(text: str) -> tuple[str, bool]:
    """OCR correction tailored for date fields — preserves separators."""
    result = []
    changed = False
    for ch in text:
        if ch.isdigit():
            result.append(ch)
        elif ch in " -/.,;:|_\\":
            result.append(ch)
        elif ch in _DATE_OCR_SUBSTITUTIONS:
            result.append(_DATE_OCR_SUBSTITUTIONS[ch])
            changed = True
    return "".join(result), changed


def _validate_date_of_birth(raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), "date_of_birth")

    corrected, auto_corrected = _auto_correct_date(text)

    normalized_value: str | None = normalize_date(corrected)

    if normalized_value is None:
        normalized_value = normalize_date(text)

    if normalized_value is None:
        digits = digits_only(corrected)
        if len(digits) == 8:
            normalized_value = normalize_date(
                f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
            )

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

    display_value = (
        normalized_value[8:10] + "/" + normalized_value[5:7] + "/" + normalized_value[0:4]
    )

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


def _correct_place_with_gazetteer(text: str) -> tuple[str, bool]:
    """Apply Vietnamese administrative gazetteer correction. Returns (corrected, changed)."""
    try:
        from src.parsing.vn_places import correct_place_text

        corrected = correct_place_text(text)
        return corrected, corrected != text
    except Exception:
        return text, False


def _restore_place_diacritics(text: str) -> tuple[str, bool]:
    """Restore Vietnamese diacritics using gazetteer word lookup."""
    try:
        from src.parsing.vn_places import restore_place_diacritics

        restored = restore_place_diacritics(text)
        return restored, restored != text
    except Exception:
        return text, False


def _place_text_quality_flags(text: str) -> set[str]:
    flags: set[str] = set()
    cleaned = collapse_whitespace(text)
    if not cleaned:
        flags.add("empty")
        return flags
    if len(cleaned) < 5:
        flags.add("too_short")
    if looks_suspicious_for_field(cleaned, "place_of_residence"):
        flags.add("suspicious")
    if is_digit_heavy_text(cleaned, threshold=0.35):
        flags.add("digit_heavy")
    if len(cleaned.split()) <= 2 and "," not in cleaned:
        flags.add("understructured")
    return flags


def _place_similarity_ratio(left: str, right: str) -> float:
    left_clean = collapse_whitespace(left).lower()
    right_clean = collapse_whitespace(right).lower()
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


_COMPOUND_HYPHEN_RE = re.compile(r"\s*-\s*")

_KNOWN_COMPOUND_PROVINCES = {
    "ba ria-vung tau", "thua thien-hue",
}


def _normalize_place_output(text: str) -> str:
    if not text:
        return text
    parts = [collapse_whitespace(p.strip()) for p in text.split(",")]
    parts = [p for p in parts if p]
    for i, part in enumerate(parts):
        if not part:
            continue
        if part[0].islower():
            parts[i] = part[0].upper() + part[1:]
    if parts:
        last = parts[-1]
        last_folded = _ascii_fold_compare(last)
        for compound in _KNOWN_COMPOUND_PROVINCES:
            if compound in last_folded:
                parts[-1] = _COMPOUND_HYPHEN_RE.sub("-", last)
                break
    result = ", ".join(parts)
    result = result.rstrip(".,;: ")
    return result


_ADMIN_STOPWORDS = {
    "xa", "xã", "phuong", "phường", "thi", "thị", "tran", "trấn",
    "huyen", "huyện", "quan", "quận", "thanh", "thành", "pho", "phố",
    "tinh", "tỉnh", "tp",
}


def _place_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", collapse_whitespace(text)).lower()
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    return [token for token in tokens if len(token) >= 2 and token not in _ADMIN_STOPWORDS]


def _suffix_token_overlap(left: str, right: str) -> float:
    left_tokens = set(_place_tokens(left))
    right_tokens = set(_place_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _admin_keyword_hits(text: str) -> int:
    lowered = unicodedata.normalize("NFC", collapse_whitespace(text)).lower()
    keywords = (
        "xã", "phường", "thị trấn", "thị xã",
        "huyện", "quận", "thành phố", "tỉnh", "tp",
    )
    return sum(1 for keyword in keywords if keyword in lowered)


def _numeric_groups(text: str) -> list[str]:
    return re.findall(r"\d+", unicodedata.normalize("NFC", text))


def _preserves_numeric_content(original: str, corrected: str) -> bool:
    original_groups = _numeric_groups(original)
    if not original_groups:
        return True
    corrected_groups = set(_numeric_groups(corrected))
    return all(group in corrected_groups for group in original_groups)


def _correction_structurally_helpful(original: str, corrected: str) -> bool:
    if corrected == original:
        return False
    similarity = _place_similarity_ratio(original, corrected)
    if similarity < 0.84:
        return False
    original_hits = _admin_keyword_hits(original)
    corrected_hits = _admin_keyword_hits(corrected)
    if corrected_hits > original_hits:
        return True
    if (
        corrected_hits >= max(2, original_hits)
        and _preserves_numeric_content(original, corrected)
        and _suffix_token_overlap(original, corrected) >= 0.40
    ):
        return True
    if corrected_hits >= 2 and _suffix_token_overlap(original, corrected) >= 0.45:
        return True
    return False


def _is_safe_address_correction(original: str, corrected: str) -> bool:
    if corrected == original:
        return True

    if _is_mostly_diacritic_change(original, corrected):
        return True

    original_parts = [collapse_whitespace(part) for part in original.split(",") if collapse_whitespace(part)]
    corrected_parts = [collapse_whitespace(part) for part in corrected.split(",") if collapse_whitespace(part)]

    original_tail = ", ".join(original_parts[-3:]) if original_parts else original
    corrected_tail = ", ".join(corrected_parts[-3:]) if corrected_parts else corrected
    tail_overlap = _suffix_token_overlap(original_tail, corrected_tail)
    full_similarity = _place_similarity_ratio(original, corrected)
    original_admin_hits = _admin_keyword_hits(original_tail)
    corrected_admin_hits = _admin_keyword_hits(corrected_tail)
    preserves_numbers = _preserves_numeric_content(original, corrected)

    if (
        preserves_numbers
        and corrected_admin_hits >= original_admin_hits + 1
        and full_similarity >= 0.82
        and tail_overlap >= 0.30
    ):
        return True

    if tail_overlap < 0.5:
        if not (
            full_similarity >= 0.85
            and tail_overlap >= 0.30
            and corrected_admin_hits >= original_admin_hits
            and preserves_numbers
        ):
            return False

    if len(corrected_parts) >= 3:
        corrected_local = corrected_parts[-3]
        raw_candidates = original_parts[-3:] if len(original_parts) >= 3 else original_parts
        best_local_overlap = max(
            (_suffix_token_overlap(corrected_local, raw_candidate) for raw_candidate in raw_candidates),
            default=0.0,
        )
        if best_local_overlap < 0.5:
            if (
                len(corrected_parts) < len(original_parts)
                and full_similarity >= 0.90
                and tail_overlap >= 0.90
            ):
                best_local_overlap = 0.5
            if not (
                full_similarity >= 0.85
                and best_local_overlap >= 0.20
                and corrected_admin_hits >= original_admin_hits
                and preserves_numbers
            ):
                return False

    original_tokens = _place_tokens(original)
    corrected_tokens = _place_tokens(corrected)
    if corrected_tokens and len(corrected_tokens) < max(3, int(len(original_tokens) * 0.7)):
        return False

    return True


def _correct_address_suffix_only(text: str) -> tuple[str, bool]:
    """Correct only the administrative suffix of an address.

    For noisy long addresses, the leading free-text part (house number, alley,
    hamlet, OCR garbage) is often much less reliable than the trailing
    administrative segments. We therefore preserve the prefix and only run
    gazetteer correction on the last 2-4 comma-separated segments.
    """
    parts = [collapse_whitespace(part) for part in text.split(",")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return text, False

    best_text = text
    best_changed = False
    best_score = 0.0

    for suffix_len in range(2, min(4, len(parts)) + 1):
        prefix_parts = parts[:-suffix_len]
        suffix_parts = parts[-suffix_len:]
        suffix_raw = ", ".join(suffix_parts)
        corrected_suffix, changed = _correct_place_with_gazetteer(suffix_raw)
        if not changed:
            continue
        score = _place_similarity_ratio(suffix_raw, corrected_suffix)
        if score < 0.55:
            continue
        candidate = ", ".join(prefix_parts + [corrected_suffix]) if prefix_parts else corrected_suffix
        if score > best_score:
            best_text = candidate
            best_changed = True
            best_score = score

    return best_text, best_changed


def _correct_address_tail_without_commas(text: str) -> tuple[str, bool]:
    """Correct a trailing administrative tail even when OCR missed commas.

    Example:
        "No truong ma Van Trach Bo Trach Quang Binh"
        -> "No truong ma, Xa Van Trach, Huyen Bo Trach, Quang Binh"
    """
    cleaned = collapse_whitespace(text)
    if not cleaned or "," in cleaned:
        return text, False

    words = cleaned.split()
    if len(words) < 4:
        return text, False

    best_text = text
    best_changed = False
    best_key = (0, 0.0)

    max_tail_words = min(8, len(words))
    for tail_len in range(3, max_tail_words + 1):
        prefix_words = words[:-tail_len]
        tail_words = words[-tail_len:]
        tail_raw = " ".join(tail_words)
        corrected_tail, changed = _correct_place_with_gazetteer(tail_raw)
        if not changed:
            continue

        similarity = _place_similarity_ratio(tail_raw, corrected_tail)
        admin_hits = _admin_keyword_hits(corrected_tail)
        if similarity < 0.68 or admin_hits < 1:
            continue

        candidate = ", ".join([" ".join(prefix_words), corrected_tail]) if prefix_words else corrected_tail
        rank_key = (admin_hits, similarity)
        if rank_key > best_key:
            best_text = candidate
            best_changed = True
            best_key = rank_key

    return best_text, best_changed


def _attempt_place_correction(
    field_name: str,
    original_value: str,
) -> tuple[str, bool, str | None]:
    """Return (corrected_value, applied, warning_code_if_skipped)."""
    if field_name == "place_of_residence":
        corrected_value, auto_corrected = _correct_address_suffix_only(original_value)
        if not auto_corrected:
            corrected_value, auto_corrected = _correct_address_tail_without_commas(original_value)
        if not auto_corrected:
            corrected_value, auto_corrected = _correct_place_with_gazetteer(original_value)
    else:
        corrected_value, auto_corrected = _correct_place_with_gazetteer(original_value)

    if not auto_corrected:
        return original_value, False, None

    if _is_mostly_diacritic_change(original_value, corrected_value):
        return corrected_value, True, None

    similarity = _place_similarity_ratio(original_value, corrected_value)
    if similarity < 0.62:
        return original_value, False, "aggressive_place_correction"
    if field_name == "place_of_residence" and not _is_safe_address_correction(original_value, corrected_value):
        return original_value, False, "unsafe_address_correction"
    return corrected_value, True, None


def _validate_place(field_name: str, raw_text: str, confidence: float) -> FieldResult:
    text = strip_known_field_prefix(collapse_whitespace(raw_text), field_name)
    value = unicodedata.normalize("NFC", text) if text else None

    warning: str | None = None
    review_reason: str | None = None
    is_valid = True
    auto_corrected = False

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

    quality_flags = _place_text_quality_flags(value)
    original_value = value

    base_correction_threshold = 0.75
    if field_name == "place_of_residence":
        base_correction_threshold = 0.50
    elif field_name == "place_of_origin":
        base_correction_threshold = 0.50

    may_correct = (
        confidence >= base_correction_threshold
        and "too_short" not in quality_flags
        and "digit_heavy" not in quality_flags
        and "suspicious" not in quality_flags
    )

    if may_correct:
        corrected_value, auto_corrected, skipped_reason = _attempt_place_correction(field_name, value)
        if auto_corrected:
            value = corrected_value
        elif skipped_reason == "aggressive_place_correction":
            warning = "parser correction skipped because it changed the OCR text too aggressively"
            review_reason = "aggressive_place_correction"
        elif skipped_reason == "unsafe_address_correction":
            warning = "address correction skipped because administrative suffix drifted too far from OCR"
            review_reason = "unsafe_address_correction"
    else:
        review_reason = review_reason or (
            "suspicious_place_ocr" if "suspicious" in quality_flags else None
        )

    # Secondary rescue path for medium-confidence addresses:
    # allow suffix-only administrative correction when the OCR text is readable
    # but under-structured, as long as the correction stays close to the OCR.
    if (
        field_name == "place_of_residence"
        and not auto_corrected
        and confidence >= 0.45
        and "too_short" not in quality_flags
        and "digit_heavy" not in quality_flags
    ):
        corrected_value, secondary_corrected, skipped_reason = _attempt_place_correction(field_name, value)
        if secondary_corrected:
            value = corrected_value
            auto_corrected = True
            if review_reason in {"understructured_place", "unsafe_address_correction"}:
                review_reason = None
            if warning and (
                "administrative suffix drifted too far" in warning
                or "thiếu cấu trúc hành chính" in warning
            ):
                warning = None

    if auto_corrected:
        quality_flags = _place_text_quality_flags(value)
        if _correction_structurally_helpful(original_value, value):
            quality_flags.discard("understructured")
            quality_flags.discard("suspicious")
            if review_reason in {"understructured_place", "suspicious_place_ocr", "unsafe_address_correction"}:
                review_reason = None
            if warning and (
                "thiáº¿u cáº¥u trÃºc hÃ nh chÃ­nh" in warning
                or "khÃ´ng Ä‘Ã¡ng tin" in warning
                or "administrative suffix drifted too far" in warning
            ):
                warning = None

    restored_value, diac_restored = _restore_place_diacritics(value)
    if diac_restored and _is_mostly_diacritic_change(value, restored_value):
        value = restored_value
        auto_corrected = True

    if not auto_corrected and field_name in ("place_of_origin", "place_of_residence"):
        try:
            from src.parsing.address_corrector import correct_address
            addr_corrected = correct_address(value)
            legacy_adds_structure = _admin_keyword_hits(addr_corrected) > _admin_keyword_hits(value)
            legacy_safe = _is_mostly_diacritic_change(value, addr_corrected) or (
                legacy_adds_structure and _correction_structurally_helpful(value, addr_corrected)
            )
            if field_name == "place_of_residence":
                legacy_safe = legacy_safe and _is_safe_address_correction(value, addr_corrected)
            if (
                addr_corrected != value
                and _place_similarity_ratio(value, addr_corrected) >= 0.75
                and legacy_safe
            ):
                value = addr_corrected
                auto_corrected = True
        except Exception:
            pass

    value = _normalize_place_output(value)

    if len(value) < 5:
        warning = "Địa chỉ ngắn bất thường — có thể OCR thiếu"
        review_reason = "short_address"
    elif "suspicious" in quality_flags:
        warning = warning or "Địa chỉ OCR có dấu hiệu không đáng tin"
        review_reason = review_reason or "suspicious_place_ocr"
    elif "understructured" in quality_flags and confidence < 0.85:
        warning = warning or "Địa chỉ thiếu cấu trúc hành chính"
        review_reason = review_reason or "understructured_place"

    if confidence < 0.5:
        review_reason = review_reason or "low_confidence"

    return FieldResult(
        field_name=field_name,
        raw_text=raw_text,
        value=value,
        confidence=confidence,
        is_valid=is_valid,
        auto_corrected=auto_corrected,
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

        if info.id_number and info.date_of_birth and len(info.id_number) == 12:
            cross_warn = _cross_validate_id_dob(info.id_number, info.date_of_birth)
            if cross_warn:
                info.validation_warnings.append(f"cross_validation: {cross_warn}")
                info.review_reasons.append("id_dob_year_mismatch")

        info.needs_review = len(info.review_reasons) > 0
        return info
