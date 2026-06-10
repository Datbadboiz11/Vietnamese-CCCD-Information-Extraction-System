"""
Post-processing corrector for Vietnamese address/origin fields.

Uses the national administrative divisions database to fuzzy-match
and correct OCR output for province, district, and ward names.

Strategy:
  1. Split OCR text by commas into parts
  2. Match the last part against known province names
  3. Match second-to-last against districts of that province
  4. Match earlier parts against wards of that district
  5. Reconstruct with corrected names, preserving non-matched prefixes
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

LOGGER = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "vn_administrative"
_LOOKUP_PATH = _DATA_DIR / "divisions_lookup.json"

_PREFIX_RE = re.compile(
    r"^(?:Thành phố|Tỉnh|Thị xã|Thị trấn|Quận|Huyện|Phường|Xã|Thành Phố|TP\.?\s*|TH\.?\s*|TT\.?\s*|TX\.?\s*|T\.T\.?\s*|T\.t\.?\s*|H\.|P\.|Q\.)\s*",
    re.IGNORECASE,
)

_LOADED: dict | None = None


def _load_data() -> dict:
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not _LOOKUP_PATH.exists():
        LOGGER.warning("Administrative data not found at %s", _LOOKUP_PATH)
        _LOADED = {"provinces": {}, "districts": {}, "wards": {}, "province_names": {}}
        return _LOADED

    raw = json.loads(_LOOKUP_PATH.read_text(encoding="utf-8"))

    provinces: dict[str, str] = {}
    province_code_by_name: dict[str, str] = {}
    for slug, (name, code) in raw["p"].items():
        provinces[code] = name
        province_code_by_name[_normalize(name)] = code
        province_code_by_name[_normalize(_strip_prefix(name))] = code

    districts: dict[str, dict[str, str]] = {}
    district_code_map: dict[str, str] = {}
    for pcode, dists in raw["d"].items():
        districts[pcode] = {}
        for slug, (full_name, dcode) in dists.items():
            short = _strip_prefix(full_name)
            districts[pcode][_normalize(short)] = short
            district_code_map[dcode] = pcode

    wards: dict[str, dict[str, str]] = {}
    for dcode, ward_dict in raw["w"].items():
        pcode = district_code_map.get(dcode, "")
        wards[dcode] = {}
        for slug, full_name in ward_dict.items():
            short = _strip_prefix(full_name)
            wards[dcode][_normalize(short)] = short

    _LOADED = {
        "provinces": provinces,
        "province_names": province_code_by_name,
        "districts": districts,
        "district_code_map": district_code_map,
        "wards": wards,
        "raw": raw,
    }
    return _LOADED


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())




def _ascii_fold(text: str) -> str:
    text = _normalize(text)
    text = text.replace("đ", "d").replace("Đ", "d")
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _strip_prefix(name: str) -> str:
    return _PREFIX_RE.sub("", name).strip()


def _extract_prefix(text: str) -> tuple[str, str]:
    m = _PREFIX_RE.match(text.strip())
    if m:
        prefix = m.group(0)
        rest = text.strip()[len(prefix):].strip()
        return prefix, rest
    return "", text.strip()


def _edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[len(a)]


class _Match(NamedTuple):
    name: str
    score: float
    key: str


def _fuzzy_find(query: str, candidates: dict[str, str], max_dist_ratio: float = 0.35) -> _Match | None:
    if not query or not candidates:
        return None

    q_norm = _normalize(query)
    q_ascii = _ascii_fold(query)

    if q_norm in candidates:
        return _Match(candidates[q_norm], 1.0, q_norm)

    best: _Match | None = None
    best_dist = float("inf")

    for cand_key, cand_name in candidates.items():
        c_ascii = _ascii_fold(cand_name)

        dist = _edit_distance(q_ascii, c_ascii)
        max_len = max(len(q_ascii), len(c_ascii), 1)
        ratio = dist / max_len

        if ratio <= max_dist_ratio and dist < best_dist:
            score = 1.0 - ratio
            best_dist = dist
            best = _Match(cand_name, score, cand_key)

    return best


def _would_change(original: str, corrected: str) -> bool:
    return _normalize(original.strip()) != _normalize(corrected.strip())


def _find_province(part: str, data: dict) -> tuple[str | None, str | None]:
    part_clean = _strip_prefix(part.strip())
    part_norm = _normalize(part_clean)

    pcode_by_name = data["province_names"]
    if part_norm in pcode_by_name:
        code = pcode_by_name[part_norm]
        return data["provinces"][code], code

    provinces_lookup = {_normalize(name): name for code, name in data["provinces"].items()}
    match = _fuzzy_find(part_clean, provinces_lookup, max_dist_ratio=0.30)
    if match:
        name_norm = _normalize(match.name)
        code = pcode_by_name.get(name_norm)
        return match.name, code

    return None, None


def _find_district(part: str, province_code: str, data: dict) -> tuple[str | None, str | None]:
    part_clean = _strip_prefix(part.strip())
    districts = data["districts"].get(province_code, {})
    if not districts:
        return None, None

    match = _fuzzy_find(part_clean, districts, max_dist_ratio=0.35)
    if not match:
        return None, None

    raw_dists = data["raw"]["d"].get(province_code, {})
    for slug, (full_name, dcode) in raw_dists.items():
        if _normalize(_strip_prefix(full_name)) == match.key:
            return match.name, dcode

    return match.name, None


def _find_ward(part: str, district_code: str, data: dict) -> str | None:
    part_clean = _strip_prefix(part.strip())
    wards = data["wards"].get(district_code, {})
    if not wards:
        return None

    match = _fuzzy_find(part_clean, wards, max_dist_ratio=0.35)
    return match.name if match else None


def _has_number_prefix(text: str) -> bool:
    return bool(re.match(r"^\d+[/\-\.]?\d*\s", text.strip()))


def _only_digits_differ(a: str, b: str) -> bool:
    a_no_d = re.sub(r"\d+", "", a)
    b_no_d = re.sub(r"\d+", "", b)
    return a_no_d == b_no_d and a != b


def _replace_name_only(original: str, corrected_short: str) -> str:
    orig_stripped = _strip_prefix(original)
    if not orig_stripped:
        return original
    if _normalize(orig_stripped) == _normalize(corrected_short):
        return original
    if _ascii_fold(orig_stripped) == _ascii_fold(corrected_short):
        return original
    if _only_digits_differ(_normalize(orig_stripped), _normalize(corrected_short)):
        return original
    idx = original.find(orig_stripped)
    if idx < 0:
        return original
    return original[:idx] + corrected_short + original[idx + len(orig_stripped):]


def correct_address(text: str) -> str:
    if not text or not text.strip():
        return text

    data = _load_data()
    if not data["provinces"]:
        return text

    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return text

    province_name, province_code = _find_province(parts[-1], data)
    if not province_name or not province_code:
        return text

    result_parts = list(parts)
    result_parts[-1] = _replace_name_only(parts[-1], province_name)

    if len(parts) >= 2:
        district_name, district_code = _find_district(parts[-2], province_code, data)
        if district_name:
            result_parts[-2] = _replace_name_only(parts[-2], district_name)

            if len(parts) >= 3 and district_code:
                ward_part = parts[-3]
                if not _has_number_prefix(ward_part):
                    ward_name = _find_ward(ward_part, district_code, data)
                    if ward_name:
                        result_parts[-3] = _replace_name_only(ward_part, ward_name)

    return ", ".join(result_parts)


__all__ = ["correct_address"]
