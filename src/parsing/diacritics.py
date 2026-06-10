"""Vietnamese diacritics restoration for OCR output.

Builds a syllable-level dictionary from the administrative gazetteer and
common Vietnamese address vocabulary, then restores missing diacritics on
text that OCR produced without them.

Usage::

    from src.parsing.diacritics import restore_diacritics

    restore_diacritics("Hau Giang")        # -> "Hậu Giang"
    restore_diacritics("NGUYEN HUE")       # -> "NGUYỄN HUỆ"
    restore_diacritics("phuong Tan Dinh")  # -> "phường Tân Định"

Integration: called in ``src.parsing.vn_places.correct_place_text()``
BEFORE segment matching, so the fuzzy matcher receives text closer to the
canonical form.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GAZETTEER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "vn_administrative"
    / "divisions_lookup.json"
)

# ---------------------------------------------------------------------------
# Vietnamese diacritics detection
# ---------------------------------------------------------------------------

_VIET_DIAC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ASCII folding  (NFD → strip combining marks, đ→d)
# ---------------------------------------------------------------------------


def _ascii_fold(text: str) -> str:
    """Fold a Vietnamese string to plain ASCII lowercase."""
    out = unicodedata.normalize("NFD", text.lower())
    out = out.replace("đ", "d").replace("Đ", "d")  # đ / Đ
    return "".join(ch for ch in out if unicodedata.category(ch) != "Mn")


# ---------------------------------------------------------------------------
# Capitalisation helpers
# ---------------------------------------------------------------------------


def _detect_cap(word: str) -> str:
    """Detect capitalisation pattern of *word*.

    Returns one of ``"upper"``, ``"title"``, ``"lower"``.
    """
    if word.isupper():
        return "upper"
    if word[0].isupper():
        return "title"
    return "lower"


def _apply_cap(word: str, pattern: str) -> str:
    """Apply a capitalisation *pattern* to *word*."""
    if pattern == "upper":
        return word.upper()
    if pattern == "title":
        return word[0].upper() + word[1:] if len(word) > 1 else word.upper()
    return word.lower()


# ---------------------------------------------------------------------------
# Common Vietnamese address vocabulary (not always in the gazetteer)
# ---------------------------------------------------------------------------

# Tuples of (word, weight).  Higher weight makes the form win over
# alternatives that share the same ASCII-folded key.
_ADDRESS_VOCAB: list[tuple[str, int]] = [
    # Administrative prefixes — extremely common in CCCD addresses
    ("Tỉnh", 200), ("Quận", 200), ("Huyện", 200),
    ("Phường", 200), ("Xã", 200),
    ("Thành", 100), ("phố", 100),   # "Thành phố" split into syllables
    ("Thị", 100), ("xã", 100), ("trấn", 100),  # "Thị xã", "Thị trấn"
    # Street / address elements
    ("Số", 200), ("Đường", 200), ("Ngõ", 250), ("Ngách", 200),
    ("Hẻm", 200), ("Tổ", 150), ("Khu", 150), ("Phố", 150),
    ("Thôn", 200), ("Xóm", 200), ("Ấp", 200), ("Bản", 150),
    ("Buôn", 150), ("Khóm", 150), ("Liên", 50),
    # Common geographical words
    ("Sông", 50), ("Núi", 50), ("Hồ", 50), ("Biển", 50),
    ("Đảo", 50), ("Vịnh", 50), ("Đầm", 50),
    # Directional / positional
    ("Bắc", 50), ("Nam", 50), ("Đông", 50), ("Tây", 50),
    ("Trung", 50), ("Thượng", 50), ("Hạ", 50),
    ("Nội", 50), ("Ngoại", 50),
    # Common words in place names
    ("Mới", 50), ("Cũ", 50), ("Lớn", 50), ("Nhỏ", 50),
    ("Cao", 50), ("Thấp", 50),
    ("An", 50), ("Bình", 50), ("Châu", 50), ("Đức", 50),
    ("Gia", 50), ("Hà", 50), ("Hải", 50), ("Hòa", 50),
    ("Hưng", 50), ("Khánh", 50), ("Lâm", 50), ("Long", 50),
    ("Minh", 50), ("Nghĩa", 50), ("Ngọc", 50),
    ("Nhân", 50), ("Ninh", 50), ("Phú", 50), ("Phước", 50),
    ("Quang", 50), ("Sơn", 50), ("Tân", 50),
    ("Thành", 50), ("Thiện", 50), ("Thịnh", 50), ("Thuận", 50),
    ("Trường", 50), ("Vĩnh", 50), ("Xuân", 50),
    ("Định", 50), ("Đồng", 50), ("Lộc", 50), ("Phong", 50),
    ("Tài", 50), ("Thắng", 50), ("Vinh", 50),
    ("Yên", 50), ("Bảo", 50), ("Cường", 50), ("Đại", 50),
    ("Hiệp", 50), ("Hội", 50), ("Kiến", 50),
    ("Lợi", 50), ("Mỹ", 50), ("Phát", 50), ("Thọ", 50),
    ("Tiến", 50), ("Trị", 50),
    ("Tường", 50), ("Văn", 50), ("Đắk", 50), ("Đăk", 50),
    # Ethnicity-related place name words
    ("Chăm", 50), ("Khơ", 50), ("Mường", 50), ("Thái", 50),
    ("Tày", 50), ("Nùng", 50),
    # Words from "Nơi đăng ký khai sinh" / "Quê quán" / "Nơi thường trú"
    ("Nơi", 50), ("Đăng", 50), ("Thường", 50), ("Trú", 50),
]

# High-priority overrides: syllables whose correct form in CCCD / address
# context is nearly always a specific diacritical variant, even though
# the gazetteer may have other variants with higher raw count.
# These get a very large boost so they always win.
_CONTEXT_OVERRIDES: list[tuple[str, int]] = [
    # "Nguyễn" (surname, street names) — NOT "Nguyên" (source/origin)
    ("Nguyễn", 500),
    # "Huệ" as in Nguyễn Huệ — more common in addresses than "Huế" (the city)
    # Note: "Huế" still wins when it appears as-is (already accented)
    ("Huệ", 300),
    # "Điện" as in Điện Biên — NOT "Điền" (field/farm)
    ("Điện", 300),
    # "Lê" (surname, street names)
    ("Lê", 300),
    # "Trần" (surname, street names)
    ("Trần", 300),
    # "Phạm" (surname, street names)
    ("Phạm", 300),
    # "Võ" / "Vũ" (surnames) — "Vũ" is more common in the North
    ("Võ", 200),
    ("Vũ", 200),
    # "Huỳnh" (surname, street names)
    ("Huỳnh", 300),
    # "Đặng" (surname)
    ("Đặng", 300),
    # "Bùi" (surname)
    ("Bùi", 300),
    # "Ngô" (surname)
    ("Ngô", 200),
    # "Hồ" (surname, lake) — give it a boost over "hò", "hổ", etc.
    ("Hồ", 200),
    # "Tôn" as in "Tôn Đức Thắng" street
    ("Tôn", 150),
    # "Lý" (surname, street names) — NOT "lý" (reason)
    ("Lý", 200),
    # "Phan" is already ASCII-clean but ensure it's in the dict
    ("Phan", 100),
    # "Đỗ" (surname)
    ("Đỗ", 200),
    # "Đinh" (surname) — NOT "Định" (which is already boosted above)
    # Tricky: "dinh" can be Đinh (surname) or Định (place-name word)
    # In address context, Định is more common, so keep Định's higher weight

    # Province-name syllables that are frequently ambiguous.
    # These are critical because province names appear on every CCCD.
    # "Đà" as in Đà Nẵng, Lâm Đồng's Đà Lạt — NOT "đa" (multi-)
    ("Đà", 200),
    # "Nẵng" as in Đà Nẵng — NOT "năng" (energy)
    ("Nẵng", 200),
    # "Phòng" as in Hải Phòng — NOT "phong" (style/wind)
    # "phong" appears in many place names (Phong Điền, Vĩnh Phong...)
    # but "Phòng" is the more useful restoration in address context
    ("Phòng", 300),
    # "Thơ" as in Cần Thơ — NOT "thọ" (longevity)
    ("Thơ", 200),
    # "Đồng" as in Đồng Nai, Đồng Tháp — NOT "đông" (east)
    ("Đồng", 200),
    # "Nai" stays as-is (no diacritics needed for Đồng Nai)
    # "Dương" as in Hải Dương, Bình Dương — competes with "Đường" (street).
    # In CCCD addresses, "Đường" (street) is more common standalone, and
    # province names like "Bình Dương" will be fixed by gazetteer matching.
    # So we do NOT override here — let "Đường" (weight 200) win.
    # "Giang" as in Hậu Giang, Tiền Giang — already no-diacritics, fine
    # "Lạng" as in Lạng Sơn
    ("Lạng", 150),
    # "Quảng" as in Quảng Nam, Quảng Ngãi, etc.
    ("Quảng", 200),
    # "Ngãi" as in Quảng Ngãi
    ("Ngãi", 200),
    # "Thừa" as in Thừa Thiên
    ("Thừa", 150),
    # "Thiên" as in Thừa Thiên Huế
    ("Thiên", 100),
    # "Đắk" / "Đắc" as in Đắk Lắk, Đắk Nông
    ("Lắk", 150),
    # "Nông" as in Đắk Nông
    ("Nông", 150),
    # "Bến" as in Bến Tre
    ("Bến", 200),
    # "Trà" as in Trà Vinh
    ("Trà", 150),
    # "Bạc" as in Bạc Liêu
    ("Bạc", 150),
    # "Liêu" as in Bạc Liêu
    ("Liêu", 150),
    # "Cà" as in Cà Mau
    ("Cà", 150),
    # "Kiên" as in Kiên Giang
    ("Kiên", 200),
    # "Tiền" as in Tiền Giang
    ("Tiền", 200),
    # "Vĩnh" is already in address vocab
    # "Bà" as in Bà Rịa
    ("Bà", 150),
    # "Rịa" as in Bà Rịa - Vũng Tàu
    ("Rịa", 150),
    # "Vũng" as in Vũng Tàu
    ("Vũng", 150),
    # "Tàu" as in Vũng Tàu
    ("Tàu", 150),
]

# ---------------------------------------------------------------------------
# Lazy-loaded syllable dictionary
# ---------------------------------------------------------------------------

# Mapping:  ascii_folded_syllable -> list of (accented_form_lowercase, count)
# Sorted descending by count so index 0 is the most frequent form.
_syllable_dict: dict[str, list[tuple[str, int]]] | None = None


def _extract_syllables_from_name(name: str) -> list[str]:
    """Split a Vietnamese place name into individual syllables.

    Strips known administrative prefixes first so that "Phường Phúc Xá"
    yields ["Phúc", "Xá"] (the prefix "Phường" is handled separately
    via _ADDRESS_VOCAB).
    """
    # Strip leading admin prefix
    prefixes = (
        "Thành phố ", "Thị trấn ", "Thị xã ",
        "Phường ", "Quận ", "Huyện ", "Xã ", "Tỉnh ",
    )
    text = name
    for pfx in prefixes:
        if text.startswith(pfx):
            text = text[len(pfx):]
            break

    # Split on whitespace; keep only alphabetic tokens
    return [w for w in text.split() if w and re.match(r"^[A-Za-zÀ-ỹĐđ]+$", w)]


def _build_dict() -> dict[str, list[tuple[str, int]]]:
    """Build the syllable dictionary from gazetteer + address vocab."""
    counter: dict[str, Counter[str]] = {}  # ascii_key -> Counter(lowercase_accented -> count)

    def _add(syllable: str, count: int = 1) -> None:
        key = _ascii_fold(syllable)
        if not key or not key.isalpha():
            return
        low = syllable.lower()
        if key not in counter:
            counter[key] = Counter()
        counter[key][low] += count

    # --- 1. Common address vocabulary (weighted) ---
    for word, weight in _ADDRESS_VOCAB:
        _add(word, count=weight)

    # --- 2. Context overrides (surnames, common street-name words) ---
    for word, weight in _CONTEXT_OVERRIDES:
        _add(word, count=weight)

    # --- 3. Gazetteer place names ---
    if _GAZETTEER_PATH.exists():
        try:
            with open(_GAZETTEER_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            LOGGER.warning("diacritics: failed to load gazetteer: %s", exc)
            data = {}

        # Provinces  (slug -> [name, code])
        for _slug, (name, _code) in data.get("p", {}).items():
            for syl in _extract_syllables_from_name(name):
                # Provinces are high-value, give them extra weight
                _add(syl, count=10)

        # Districts  (prov_code -> {slug -> [name, code]})
        for _pcode, districts in data.get("d", {}).items():
            for _slug, (name, _code) in districts.items():
                for syl in _extract_syllables_from_name(name):
                    _add(syl, count=5)

        # Wards  (dist_code -> {slug -> name})
        for _dcode, wards in data.get("w", {}).items():
            for _slug, name in wards.items():
                for syl in _extract_syllables_from_name(name):
                    _add(syl, count=1)
    else:
        LOGGER.warning(
            "diacritics: gazetteer not found at %s; dictionary will be small",
            _GAZETTEER_PATH,
        )

    # --- 3. Convert Counter → sorted list ---
    result: dict[str, list[tuple[str, int]]] = {}
    for key, cnt in counter.items():
        # Sort by count descending, then alphabetically for stability
        forms = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
        result[key] = forms

    LOGGER.info(
        "diacritics: built dictionary with %d ASCII keys, %d total forms",
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


def _ensure_dict() -> dict[str, list[tuple[str, int]]]:
    """Lazy-load the syllable dictionary on first access."""
    global _syllable_dict
    if _syllable_dict is None:
        _syllable_dict = _build_dict()
    return _syllable_dict


# ---------------------------------------------------------------------------
# Core restoration logic
# ---------------------------------------------------------------------------

# Regex to detect tokens that are purely alphabetic (candidates for restoration)
_ALPHA_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+$")


def _has_vietnamese_diacritics(word: str) -> bool:
    """Return True if *word* already contains Vietnamese diacritics."""
    return bool(_VIET_DIAC_RE.search(word))


def _restore_word(word: str, syl_dict: dict[str, list[tuple[str, int]]]) -> str:
    """Restore diacritics for a single word, preserving capitalisation."""
    # Skip if already has diacritics
    if _has_vietnamese_diacritics(word):
        return word

    # Skip non-alphabetic tokens (numbers, punctuation, mixed)
    if not _ALPHA_RE.match(word):
        return word

    key = _ascii_fold(word)
    if not key:
        return word

    forms = syl_dict.get(key)
    if not forms:
        # No match in dictionary; return as-is
        return word

    # Pick the highest-frequency form (first in sorted list)
    best_form, _count = forms[0]

    # If the best form is the same as the ASCII-folded input (i.e. the word
    # genuinely has no diacritics in Vietnamese), return as-is
    if best_form == key:
        return word

    # Apply the original capitalisation pattern
    cap = _detect_cap(word)
    return _apply_cap(best_form, cap)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def restore_diacritics(text: str) -> str:
    """Restore Vietnamese diacritics in *text*.

    Splits text into whitespace-separated tokens.  For each token that
    lacks Vietnamese diacritics, looks up its ASCII-folded form in a
    syllable dictionary built from the administrative gazetteer and common
    address vocabulary.  Replaces with the most frequent accented form,
    preserving original capitalisation (UPPER / Title / lower).

    Tokens that already contain diacritics, numbers, or punctuation are
    passed through unchanged.

    Parameters
    ----------
    text : str
        OCR output text, possibly missing diacritics.

    Returns
    -------
    str
        Text with diacritics restored where possible.

    Examples
    --------
    >>> restore_diacritics("Hau Giang")
    'Hậu Giang'
    >>> restore_diacritics("NGUYEN HUE")
    'NGUYỄN HUỆ'
    """
    if not text or not text.strip():
        return text

    syl_dict = _ensure_dict()

    # Tokenise preserving whitespace structure.
    # We split on whitespace boundaries but keep the whitespace intact.
    parts = re.split(r"(\s+)", text)
    result: list[str] = []

    for part in parts:
        if not part:
            continue
        # Whitespace chunk → keep as-is
        if part.isspace():
            result.append(part)
            continue
        # Non-whitespace chunk → may contain punctuation attached to word
        # e.g. "Giang," or "(Hue)" — separate leading/trailing punct
        m = re.match(r"^([^A-Za-zÀ-ỹĐđ]*)([A-Za-zÀ-ỹĐđ]+)([^A-Za-zÀ-ỹĐđ]*)$", part)
        if m:
            prefix_punct, core, suffix_punct = m.groups()
            restored = _restore_word(core, syl_dict)
            result.append(prefix_punct + restored + suffix_punct)
        else:
            # Complex token (multiple punct-word transitions) → keep as-is
            result.append(part)

    return "".join(result)


def get_dictionary_stats() -> dict[str, int]:
    """Return statistics about the loaded syllable dictionary.

    Useful for diagnostics / testing.
    """
    syl_dict = _ensure_dict()
    total_forms = sum(len(v) for v in syl_dict.values())
    ambiguous = sum(1 for v in syl_dict.values() if len(v) > 1)
    return {
        "ascii_keys": len(syl_dict),
        "total_forms": total_forms,
        "ambiguous_keys": ambiguous,
    }


__all__ = [
    "restore_diacritics",
    "get_dictionary_stats",
]
