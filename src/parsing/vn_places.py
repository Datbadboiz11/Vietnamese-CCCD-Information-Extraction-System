"""Vietnamese place name post-correction using administrative division gazetteer.

Architecture:
    1. Split input by comma/period into segments
    2. Fix common OCR prefix errors (Phương→Phường, Quân→Quận, etc.)
    3. Classify segments by administrative prefix (province/district/ward/detail)
    4. Match hierarchically: province → district (in province) → ward (in district)
    5. Keep unmatched segments (street, house number, thôn, xóm) as-is
    6. Assemble corrected output

Data source: github.com/madnh/hanhchinhvn (Tổng Cục Thống Kê)
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "vn_administrative"
    / "divisions_lookup.json"
)

_provinces: dict[str, list[str]] = {}
_districts_by_prov: dict[str, dict[str, list[str]]] = {}
_wards_by_dist: dict[str, dict[str, str]] = {}
_loaded = False


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _ascii_fold(text: str) -> str:
    out = unicodedata.normalize("NFD", text.lower())
    out = out.replace("đ", "d").replace("Đ", "d")
    return "".join(ch for ch in out if unicodedata.category(ch) != "Mn")


def _to_slug(text: str) -> str:
    folded = _ascii_fold(text.strip())
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-")


# ---------------------------------------------------------------------------
# OCR prefix corrections  (applied per-segment before matching)
# ---------------------------------------------------------------------------

_OCR_PREFIX_FIXES: list[tuple[re.Pattern[str], str]] = [
    # Phường (only before digits to avoid false positives like "Phương Liệt")
    (re.compile(r"^Phương(?=\s+\d)"), "Phường"),
    (re.compile(r"^Phướng(?=\s)"), "Phường"),
    (re.compile(r"^Phưởng(?=\s)"), "Phường"),       # OCR: ở vs ờ
    (re.compile(r"^Phurng(?=\s)", re.IGNORECASE), "Phường"),   # OCR: ườ→ur
    (re.compile(r"^Phurong(?=\s)", re.IGNORECASE), "Phường"),  # OCR: ườ→uro
    (re.compile(r"^Phuờng(?=\s)"), "Phường"),        # OCR: mixed
    (re.compile(r"^Phưng(?=\s)", re.IGNORECASE), "Phường"),    # OCR: missing ờ
    (re.compile(r"^Phuong(?=\s+\d)", re.IGNORECASE), "Phường"),
    (re.compile(r"^P\.\s*(?=\d)"), "Phường "),
    (re.compile(r"^P\.\s+(?=[A-ZĐÀ-Ỹ])"), "Phường "),
    (re.compile(r"^P(?=\d)"), "Phường "),
    # Quận
    (re.compile(r"^Quân(?=\s)"), "Quận"),
    (re.compile(r"^Quan(?=\s+\d)", re.IGNORECASE), "Quận"),
    (re.compile(r"^Quàn(?=\s)"), "Quận"),
    (re.compile(r"^Q\.\s*(?=\d)"), "Quận "),
    (re.compile(r"^Q\.\s+(?=[A-ZĐÀ-Ỹ])"), "Quận "),
    (re.compile(r"^Q(?=\d)"), "Quận "),
    # Huyện
    (re.compile(r"^Huyên(?=\s)"), "Huyện"),
    (re.compile(r"^Huyen(?=\s)", re.IGNORECASE), "Huyện"),
    (re.compile(r"^H\.\s*(?=\S)"), "Huyện "),
    # Xã
    (re.compile(r"^Xà(?=\s)"), "Xã"),
    # Thành phố / Thị xã / Thị trấn
    (re.compile(r"^Thành\s+phổ(?=\s)", re.IGNORECASE), "Thành phố"),  # OCR: ổ vs ố
    (re.compile(r"^T\.?P\.?\s+", re.IGNORECASE), "Thành phố "),
    (re.compile(r"^TH\.?\s+", re.IGNORECASE), "Thành phố "),
    (re.compile(r"^TX\.?\s+", re.IGNORECASE), "Thị xã "),
    (re.compile(r"^TT\.?\s+", re.IGNORECASE), "Thị trấn "),
    # Tỉnh
    (re.compile(r"^Tính(?=\s)"), "Tỉnh"),            # OCR: í vs ỉ
    # Đường
    (re.compile(r"^Đườn(?=\s)"), "Đường"),
    (re.compile(r"^Duong(?=\s)", re.IGNORECASE), "Đường"),
    # Thôn (common OCR confusions)
    (re.compile(r"^Trôn(?=\s)"), "Thôn"),
    (re.compile(r"^Thón(?=\s)"), "Thôn"),
    (re.compile(r"^Thon(?=\s)", re.IGNORECASE), "Thôn"),
    (re.compile(r"^Thồn(?=\s)"), "Thôn"),
    # Ấp
    (re.compile(r"^[ẪÂẨ]p(?=\s)"), "Ấp"),
    (re.compile(r"^Ap(?=\s+\d)", re.IGNORECASE), "Ấp"),
    # Khóm
    (re.compile(r"^Khom(?=\s)", re.IGNORECASE), "Khóm"),
    # Tổ
    (re.compile(r"^To(?=\s+\d)", re.IGNORECASE), "Tổ"),
]

_LABEL_BLEED_RE = re.compile(
    r"^(?:N[oơ][itl]?\s+(?:[DdĐđ]?K?H?[IiKk]?[Kk]?\s*)?th[uư]?r?[oơờ]?ng\s*tr[uú]?|"
    r"[NnVv][oơà][itl]?\s+[DdĐđ]K?H?[IiKk]?K|"
    r"N[gq]?uy[eê]?n\s*qu[aá]n|"
    r"[Qq]u[eê]?\s*qu[aáâ]n|"
    r"[FfE]?'?\s*[Pp]\w{0,5}c[eo]?\s+\w{0,4}f?\s*\w{0,12}|"
    r"Place\s+[ao][fl]\s+\w+|"
    r"\borigin\b|"
    r"\borgin\b|"
    r"\bof\s*orgin\b|"
    r"\bresidence?\b|"
    r"N[oơ]i?\s+[DdĐđ]ương\s+N\w*|"
    r"NGƯỜI\s+[ĐDdđ]\w*\s+[Tt]r[uư]?\w*|"
    r"Sinh\s+ng[aà]y\s*[\d/\-\.]+|"
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"\s*[,:./ I]*",
    re.IGNORECASE,
)

_TRAILING_LABEL_RE = re.compile(
    r"\s*[,.]?\s*(?:N[oơ]i?\s*(?:D?K?H?K?\s*)?th[uư]?r?[oơờ]?ng\s*tr[uú]?|"
    r"N?g?[qu]+y[eê]?n\s*qu[aá]n|"
    r"[Qq]u[eê]?\s*qu[aáâ]n|"
    r"[FfE]?'?\s*[Pp]\w{0,5}c[eo]?\s+\w{0,4}f?\s*\w*|"
    r"Place\s*[ao][fl]\s*\w*|"
    r"Pila\w*|"
    r"NGƯỜI\s+\w+|"
    r"N[oơ]i?\s+thu\w*)\s*[/:\\-]?\s*$",
    re.IGNORECASE,
)

_MID_TEXT_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bPhurng\b", re.IGNORECASE), "Phường"),
    (re.compile(r"\bPhurong\b", re.IGNORECASE), "Phường"),
    (re.compile(r"\bPhuong(?=\s+\d)", re.IGNORECASE), "Phường"),
    (re.compile(r"\bQuan(?=\s+\d)", re.IGNORECASE), "Quận"),
    (re.compile(r"\bHuyen\b", re.IGNORECASE), "Huyện"),
    (re.compile(r"\bThanh\s+pho\b", re.IGNORECASE), "Thành phố"),
    (re.compile(r"\bThi\s+xa\b", re.IGNORECASE), "Thị xã"),
    (re.compile(r"\bThi\s+tran\b", re.IGNORECASE), "Thị trấn"),
    (re.compile(r"\bTron(?=\s)", re.IGNORECASE), "Thôn"),
    (re.compile(r"\bThon(?=\s)", re.IGNORECASE), "Thôn"),
    (re.compile(r"\bTH\.\s+", re.IGNORECASE), "TP. "),
    (re.compile(r"\bNg[\-\s](?=[A-ZĐÀ-Ỹ])"), "Nguyễn "),
    (re.compile(r"\bTr[\-\s](?=[A-ZĐÀ-Ỹ])"), "Trần "),
]


_VIET_DIAC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

_ENGLISH_LABEL_KW = re.compile(
    r"\b(?:place|resid\w*|origin\w*|orgin\w*|birth|surname|full\s*name|nationality|sex)\b",
    re.IGNORECASE,
)

_GARBLED_PLACE_KW = re.compile(
    r"(?:p[ilh]?[aei]c[eo]?|[ao][fl]\s*(?:resid|orig|orgin)|ofonig|dforig|ofresid|oforgin|of\s*orgin)",
    re.IGNORECASE,
)


def _is_english_label_segment(text: str) -> bool:
    """Detect text that is English label text (possibly OCR-garbled).

    Heuristic: no Vietnamese diacritics + contains English label keywords.
    Safe because real Vietnamese place names either have diacritics or
    are administrative names (Phường/Quận/...) that don't contain English words.
    """
    if _VIET_DIAC_RE.search(text):
        return False
    if _ENGLISH_LABEL_KW.search(text):
        return True
    if _GARBLED_PLACE_KW.search(text):
        return True
    return False


_SEGMENT_TEXT_ALIASES: dict[str, str] = {
    "bo-trach": "Bố Trạch",
    "dong-van": "Đồng Văn",
    "ha-duong": "Hải Dương",
    "hai-duong": "Hải Dương",
    "ho-chinh": "Hồ Chí Minh",
    "ho-ching": "Hồ Chí Minh",
    "mang-thi": "Mang Thít",
    "mang-thit": "Mang Thít",
    "quang-binh": "Quảng Bình",
    "van-trach": "Vạn Trạch",
    "vinh-long": "Vĩnh Long",
}


def _fix_ocr_prefix(text: str) -> str:
    for _ in range(3):
        new = _LABEL_BLEED_RE.sub("", text).strip()
        if new == text:
            break
        text = new
    text = _TRAILING_LABEL_RE.sub("", text).strip()
    if not text:
        return ""
    if _is_english_label_segment(text):
        return ""
    for pat, repl in _OCR_PREFIX_FIXES:
        text = pat.sub(repl, text, count=1)
    for pat, repl in _MID_TEXT_FIXES:
        text = pat.sub(repl, text)
    return text.strip()


def _apply_segment_alias(text: str) -> str:
    slug = _to_slug(text)
    alias = _SEGMENT_TEXT_ALIASES.get(slug)
    if alias:
        return alias
    return text


# ---------------------------------------------------------------------------
# Administrative prefix detection → level classification
# ---------------------------------------------------------------------------

_ADMIN_PREFIX_RE = re.compile(
    r"^(Thành\s+phố|Tỉnh|Quận|Huyện|Thị\s+xã|Phường|Xã|Thị\s+trấn)\s+",
    re.IGNORECASE,
)
_PROVINCE_PREFIXES = {"thành phố", "tỉnh"}
_DISTRICT_PREFIXES = {"quận", "huyện", "thị xã", "thành phố"}
_WARD_PREFIXES = {"phường", "xã", "thị trấn"}
_DETAIL_PREFIX_RE = re.compile(
    r"^(?:Đường|Thôn|Xóm|Ấp|Khu\s+phố|Tổ|Số|Ngõ|Ngách|Hẻm)\b",
    re.IGNORECASE,
)


def _classify_segment(text: str) -> tuple[str, str, str]:
    """Classify segment → (level_hint, name_part, prefix_text).

    level_hint is one of 'province', 'district', 'ward', 'detail', 'unknown'.
    name_part strips the administrative prefix so matching works on the name.
    prefix_text is the original prefix (e.g. "Quận", "Phường") or "" if none.
    """
    if _DETAIL_PREFIX_RE.match(text):
        return "detail", text, ""
    m = _ADMIN_PREFIX_RE.match(text)
    if m:
        prefix_lower = re.sub(r"\s+", " ", m.group(1).lower())
        prefix_text = re.sub(r"\s+", " ", m.group(1))
        name_part = text[m.end():].strip()
        if not name_part:
            return "unknown", text, ""
        if prefix_lower in _PROVINCE_PREFIXES:
            return "province", name_part, prefix_text
        if prefix_lower in _DISTRICT_PREFIXES:
            return "district", name_part, prefix_text
        if prefix_lower in _WARD_PREFIXES:
            return "ward", name_part, prefix_text
    return "unknown", text, ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_loaded() -> bool:
    global _provinces, _districts_by_prov, _wards_by_dist, _loaded
    if _loaded:
        return True
    if not _DATA_PATH.exists():
        LOGGER.warning("VN places lookup not found at %s", _DATA_PATH)
        return False
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _provinces = data.get("p", {})
        _districts_by_prov = data.get("d", {})
        _wards_by_dist = data.get("w", {})
        _loaded = True
        LOGGER.info(
            "Loaded VN gazetteer: %d provinces, %d districts, %d wards",
            len(_provinces),
            sum(len(v) for v in _districts_by_prov.values()),
            sum(len(v) for v in _wards_by_dist.values()),
        )
        return True
    except Exception as exc:
        LOGGER.warning("Failed to load VN gazetteer: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Fuzzy matching with confidence & ambiguity check
# ---------------------------------------------------------------------------

_ADMIN_SLUG_PREFIX_RE = re.compile(
    r"^(xa|phuong|thi-?tran|huyen|quan|thanh-?pho|tinh|tp)-",
)


def _strip_admin_slug(slug: str) -> str:
    return _ADMIN_SLUG_PREFIX_RE.sub("", slug)


def _max_edit(slug: str) -> int:
    n = len(slug)
    if n <= 2:
        return 0
    if n <= 4:
        return 1
    if n <= 7:
        return 2
    return 3


def _fuzzy_match(
    query_slug: str,
    candidates: dict[str, object],
    max_dist: int | None = None,
) -> tuple[str, object, float] | None:
    """Fuzzy match with confidence scoring and ambiguity rejection.

    Tries both the raw query and admin-prefix-stripped variants of both
    the query and each candidate.

    Returns ``(matched_slug, value, confidence)`` or ``None`` when:
    - no candidate is close enough, or
    - the two best candidates are equally close (ambiguous).
    """
    if not query_slug or not candidates:
        return None

    queries = [query_slug]
    q_stripped = _strip_admin_slug(query_slug)
    if q_stripped != query_slug:
        queries.append(q_stripped)

    # Exact match (including prefix-stripped forms)
    for q in queries:
        if q in candidates:
            return q, candidates[q], 1.0
    for cand_slug, cval in candidates.items():
        cs = _strip_admin_slug(cand_slug)
        if cs == cand_slug:
            continue
        for q in queries:
            if cs == q:
                return cand_slug, cval, 1.0

    if max_dist is None:
        max_dist = _max_edit(query_slug)

    matches: list[tuple[str, object, int]] = []
    for cand_slug, cval in candidates.items():
        cand_forms = [cand_slug]
        cs = _strip_admin_slug(cand_slug)
        if cs != cand_slug:
            cand_forms.append(cs)
        best_d = max_dist + 1
        for q in queries:
            for cf in cand_forms:
                if abs(len(q) - len(cf)) > max_dist:
                    continue
                d = _levenshtein(q, cf)
                if d < best_d:
                    best_d = d
        if best_d <= max_dist:
            matches.append((cand_slug, cval, best_d))

    if not matches:
        return None

    matches.sort(key=lambda x: x[2])
    best_slug, best_val, best_d = matches[0]

    # Ambiguity: two equally close matches → don't correct
    if len(matches) >= 2 and matches[1][2] == best_d:
        return None

    confidence = 1.0 - best_d / max(len(query_slug), len(best_slug), 1)
    if confidence < 0.7:
        return None

    return best_slug, best_val, confidence


def _try_match(
    segment_text: str,
    name_part: str,
    candidates: dict[str, object],
) -> tuple[str, object, float] | None:
    """Try matching both the name_part and the full segment against candidates."""
    slug = _to_slug(name_part)
    result = _fuzzy_match(slug, candidates)
    if result:
        return result
    full_slug = _to_slug(segment_text)
    if full_slug != slug:
        result = _fuzzy_match(full_slug, candidates)
        if result:
            return result
    if slug.isdigit():
        stripped = slug.lstrip("0") or "0"
        if stripped != slug:
            result = _fuzzy_match(stripped, candidates)
            if result:
                return result
        padded = slug.zfill(2)
        if padded != slug:
            result = _fuzzy_match(padded, candidates)
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Province aliases  (abbreviations / common variants not in gazetteer)
# ---------------------------------------------------------------------------

_PROVINCE_ALIASES: dict[str, str] = {
    # Hồ Chí Minh — abbreviations
    "tp-hcm": "ho-chi-minh",
    "tphcm": "ho-chi-minh",
    "hcm": "ho-chi-minh",
    "sai-gon": "ho-chi-minh",
    "sg": "ho-chi-minh",
    # Hồ Chí Minh — common OCR garbles
    "ho-chinh": "ho-chi-minh",       # OCR: "Hồ Chính" → merged tokens
    "ho-chi": "ho-chi-minh",         # OCR truncation: "Hồ Chí"
    "ho-chi-mihn": "ho-chi-minh",    # OCR transposition
    "ho-chi-mmh": "ho-chi-minh",     # OCR: i→m ligature
    # Hà Nội — abbreviations & OCR garbles
    "hn": "ha-noi",
    "ha-n0i": "ha-noi",              # OCR: o→0
    "ha-nol": "ha-noi",              # OCR: i→l
    "ha-n6i": "ha-noi",              # OCR: o→6
    "ha-nam": "ha-nam",
    # Đà Nẵng — OCR garbles (after ASCII folding)
    "da-nong": "da-nang",            # OCR: ẵ misread as o
    "da-naing": "da-nang",           # OCR: extra i inserted
    # Cần Thơ — OCR garbles (after ASCII folding)
    "can-the": "can-tho",            # OCR: ơ→e
    "con-tho": "can-tho",            # OCR: ầ→o
    # Bắc Ninh
    "bac-nihn": "bac-ninh",          # OCR: transposition
    "bac-nm": "bac-ninh",            # OCR: truncation
    "bac-nmh": "bac-ninh",           # OCR: truncation
    # Hải Dương
    "hai-duơng": "hai-duong",        # OCR: mixed diacritics
    # Hồ Chí Minh — OCR reads "Hồ" as "Hà"
    "ha-chi-minh": "ho-chi-minh",
    "ha-chi-winh": "ho-chi-minh",
    "h-chi-minh": "ho-chi-minh",     # OCR: truncated "Hồ"
    # Common "TP. X" OCR garbles
    "tp-ho-chi-minh": "ho-chi-minh",
    "tp-ha-noi": "ha-noi",
    "tp-da-nang": "da-nang",
    "tp-can-tho": "can-tho",
    "tp-hai-phong": "hai-phong",
}

_MIN_PROVINCE_FUZZY_CONF = 0.88


def _match_province_slug(slug: str) -> tuple[str, str, float] | None:
    """Match slug against provinces (aliases + fuzzy).

    Returns ``(name, code, confidence)`` or ``None``.
    """
    alias_target = _PROVINCE_ALIASES.get(slug)
    if not alias_target and slug in {"ho-ching", "tp-ho-ching"}:
        alias_target = "ho-chi-minh"
    if alias_target and alias_target in _provinces:
        name, code = _provinces[alias_target]
        return name, code, 1.0
    result = _fuzzy_match(slug, _provinces)
    if result:
        _, (name, code), conf = result
        if conf < _MIN_PROVINCE_FUZZY_CONF:
            return None
        return name, code, conf
    return None


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

_UPPER_VIET = (
    r"[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    r"ÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
)
_PERIOD_SEP = re.compile(rf"\.\s+(?={_UPPER_VIET})")


_BARE_TP_RE = re.compile(r"^(?:T\.?P\.?|TH\.?)$", re.IGNORECASE)
_BARE_Q_RE = re.compile(r"^(?:Q\.?|0)$", re.IGNORECASE)


def _split_segments(text: str) -> list[str]:
    cleaned = re.sub(r"[;]", ",", text)
    cleaned = _PERIOD_SEP.sub(", ", cleaned)
    raw = [s.strip() for s in cleaned.split(",") if s.strip()]
    merged: list[str] = []
    i = 0
    while i < len(raw):
        if _BARE_TP_RE.match(raw[i]) and i + 1 < len(raw):
            merged.append(f"Thành phố {raw[i + 1]}")
            i += 2
        elif _BARE_Q_RE.match(raw[i]) and i + 1 < len(raw) and raw[i + 1].strip().isdigit():
            merged.append(f"Quận {raw[i + 1].strip()}")
            i += 2
        else:
            merged.append(raw[i])
            i += 1
    return merged


# ---------------------------------------------------------------------------
# District scanning inside a long segment (missing commas)
# ---------------------------------------------------------------------------

def _scan_district_in_segment(
    seg_slug: str,
    districts: dict[str, list[str]],
) -> tuple[str, str, str] | None:
    """Find a district name embedded in *seg_slug*.

    Returns ``(district_name, district_code, remaining_ward_slug)`` or ``None``.
    """
    words = seg_slug.split("-")
    if len(words) < 2:
        return None
    for dlen in range(min(4, len(words) - 1), 0, -1):
        for start in range(len(words) - dlen, -1, -1):
            candidate = "-".join(words[start : start + dlen])
            dr = _fuzzy_match(candidate, districts, max_dist=min(2, _max_edit(candidate)))
            if dr:
                _, (dist_name, dist_code), _ = dr
                before = "-".join(words[:start])
                after = "-".join(words[start + dlen :])
                remaining = "-".join(p for p in (before, after) if p)
                return dist_name, dist_code, remaining
    return None


def _scan_province_in_segment(
    segment_text: str,
) -> tuple[str, str, str] | None:
    """Find a province at the end of a multi-word segment (missing comma).

    Returns ``(province_name, province_code, remaining_text)`` or ``None``.
    Uses strict matching (confidence >= 0.8) to avoid false positives.
    """
    words = segment_text.split()
    if len(words) < 2:
        return None
    for plen in range(min(4, len(words) - 1), 0, -1):
        candidate = " ".join(words[-plen:])
        slug = _to_slug(candidate)
        pr = _match_province_slug(slug)
        if pr:
            prov_name, prov_code, conf = pr
            if conf < 0.8:
                continue
            remaining = " ".join(words[:-plen])
            return prov_name, prov_code, remaining
    return None


# ---------------------------------------------------------------------------
# Hierarchical matching  (province → district → ward)
# ---------------------------------------------------------------------------

def _with_prefix(prefix_text: str, matched_name: str) -> str:
    """Reconstruct 'Prefix Name' if input had a prefix, else just name."""
    if _ADMIN_PREFIX_RE.match(matched_name):
        return matched_name
    if prefix_text:
        return f"{prefix_text} {matched_name}"
    return matched_name


def _match_segments(segments: list[str]) -> list[str]:
    """Match segments right-to-left against the gazetteer hierarchy.

    Unmatched segments (street address, house number, thôn/xóm) are kept as-is.
    """
    n = len(segments)
    corrected = list(segments)
    hints = [_classify_segment(s) for s in segments]
    matched_idx: set[int] = set()
    prov_code: str | None = None
    dist_code: str | None = None

    # --- Province (rightmost non-detail segment) ---
    for i in range(n - 1, -1, -1):
        level, name_part, prefix_text = hints[i]
        if level == "detail" or level == "ward":
            continue
        if level in ("province", "unknown"):
            slug = _to_slug(name_part)
            pr = _match_province_slug(slug)
            if not pr:
                full_slug = _to_slug(segments[i])
                if full_slug != slug:
                    pr = _match_province_slug(full_slug)
            if pr:
                prov_name, pc, _ = pr
                corrected[i] = _with_prefix(prefix_text, prov_name)
                prov_code = pc
                matched_idx.add(i)
                break
        if i < n - 2:
            continue
        scan = _scan_province_in_segment(segments[i])
        if scan:
            prov_name, pc, remaining = scan
            prov_code = pc
            matched_idx.add(i)
            if remaining:
                dists = _districts_by_prov.get(pc, {})
                rem_slug = _to_slug(remaining)
                dr = _fuzzy_match(rem_slug, dists) if dists else None
                if dr:
                    _, (dname, dc), _ = dr
                    _, _, rem_prefix = _classify_segment(remaining)
                    corrected[i] = f"{_with_prefix(rem_prefix, dname)}, {prov_name}"
                    dist_code = dc
                else:
                    corrected[i] = f"{remaining}, {prov_name}"
            else:
                corrected[i] = prov_name
            break

    # --- District (within matched province) ---
    if prov_code:
        dists = _districts_by_prov.get(prov_code, {})
        if dists:
            for i in range(n - 1, -1, -1):
                if i in matched_idx:
                    continue
                level, name_part, prefix_text = hints[i]
                if level == "detail":
                    continue
                if level in ("district", "unknown", "province"):
                    dr = _try_match(segments[i], name_part, dists)
                    if dr:
                        _, (dist_name, dc), _ = dr
                        corrected[i] = _with_prefix(prefix_text, dist_name)
                        dist_code = dc
                        matched_idx.add(i)
                        break
                    full_slug = _to_slug(segments[i])
                    found = _scan_district_in_segment(full_slug, dists)
                    if found:
                        dist_name, dc, ward_slug = found
                        dist_code = dc
                        matched_idx.add(i)
                        wards = _wards_by_dist.get(dc, {})
                        ward_name = None
                        if wards and ward_slug:
                            wr = _fuzzy_match(ward_slug, wards)
                            if not wr:
                                wr = _fuzzy_match(_strip_admin_slug(ward_slug), wards)
                            if wr:
                                _, ward_name, _ = wr
                        corrected[i] = f"{ward_name}, {dist_name}" if ward_name else dist_name
                        break

    # --- Ward (within matched district) ---
    if dist_code:
        wards = _wards_by_dist.get(dist_code, {})
        if wards:
            for i in range(n):
                if i in matched_idx:
                    continue
                level, name_part, prefix_text = hints[i]
                if level == "detail":
                    continue
                wr = _try_match(segments[i], name_part, wards)
                if wr:
                    _, ward_name, ward_conf = wr
                    if not prefix_text and ward_conf < 0.88:
                        continue
                    corrected[i] = _with_prefix(prefix_text, ward_name)
                    matched_idx.add(i)

    return corrected


# ---------------------------------------------------------------------------
# No-comma fallback: scan full text for province → district → ward
# ---------------------------------------------------------------------------

def _try_no_comma(text: str) -> str | None:
    slug = _to_slug(text)
    words = slug.split("-")
    if len(words) < 2:
        return None

    original_words = re.split(r"[\s,.\-]+", text.strip())

    for plen in range(min(4, len(words)), 0, -1):
        prov_slug = "-".join(words[-plen:])
        pr = _match_province_slug(prov_slug)
        if not pr:
            continue
        prov_name, prov_code, _ = pr
        rest_w = words[:-plen]
        rest_orig = original_words[: len(original_words) - plen]

        dists = _districts_by_prov.get(prov_code, {})
        if not dists or not rest_w:
            if rest_orig:
                return f"{' '.join(rest_orig)}, {prov_name}"
            return prov_name

        for dlen in range(min(4, len(rest_w)), 0, -1):
            dist_slug = "-".join(rest_w[-dlen:])
            dr = _fuzzy_match(dist_slug, dists)
            if not dr:
                continue
            _, (dist_name, dist_code), _ = dr
            ward_w = rest_w[:-dlen]
            ward_orig = rest_orig[: len(rest_orig) - dlen]

            wards = _wards_by_dist.get(dist_code, {})
            if wards and ward_w:
                ward_slug = "-".join(ward_w)
                wr = _fuzzy_match(ward_slug, wards)
                if wr:
                    _, ward_name, _ = wr
                    return f"{ward_name}, {dist_name}, {prov_name}"
                ws = _strip_admin_slug(ward_slug)
                if ws != ward_slug:
                    wr2 = _fuzzy_match(ws, wards)
                    if wr2:
                        _, ward_name, _ = wr2
                        return f"{ward_name}, {dist_name}, {prov_name}"

            if ward_orig:
                return f"{' '.join(ward_orig)}, {dist_name}, {prov_name}"
            return f"{dist_name}, {prov_name}"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_place_text(text: str) -> str:
    """Correct OCR errors in a Vietnamese place name string.

    Splits text into segments, fixes OCR prefix errors, then matches
    hierarchically against the administrative gazetteer.  Unmatched
    segments (street address, house number, thôn, xóm) are kept as-is.
    """
    if not text or not text.strip():
        return text
    if not _ensure_loaded():
        return text

    segments = _split_segments(text)
    if not segments:
        return text

    segments = [_apply_segment_alias(_fix_ocr_prefix(s)) for s in segments]
    segments = [s for s in segments if s and len(s) >= 2]

    if not segments:
        return ""

    if len(segments) <= 1 and "," not in text:
        fixed = segments[0]
        result = _try_no_comma(fixed)
        if result:
            return result
        return fixed

    corrected = _match_segments(segments)
    return ", ".join(corrected)


def match_province(text: str) -> tuple[str, str] | None:
    """Match text against known provinces. Returns ``(name, code)`` or ``None``."""
    if not _ensure_loaded():
        return None
    slug = _to_slug(text)
    result = _match_province_slug(slug)
    if result:
        name, code, _ = result
        return name, code
    return None


def match_district(text: str, province_code: str) -> tuple[str, str] | None:
    """Match text against districts of a province. Returns ``(name, code)`` or ``None``."""
    if not _ensure_loaded():
        return None
    dists = _districts_by_prov.get(province_code, {})
    slug = _to_slug(text)
    result = _fuzzy_match(slug, dists)
    if result:
        _, (name, code), _ = result
        return name, code
    return None


def match_ward(text: str, district_code: str) -> str | None:
    """Match text against wards of a district. Returns corrected name or ``None``."""
    if not _ensure_loaded():
        return None
    wards = _wards_by_dist.get(district_code, {})
    slug = _to_slug(text)
    result = _fuzzy_match(slug, wards)
    if result:
        _, name, _ = result
        return name
    return None


def score_place_text(text: str) -> tuple[int, int, float]:
    """Score how well *text* matches known Vietnamese administrative names.

    Returns ``(matched_segments, total_segments, avg_confidence)``.
    A higher matched_segments means the text contains more recognizable
    place names from the gazetteer.
    """
    if not text or not text.strip():
        return (0, 0, 0.0)
    if not _ensure_loaded():
        return (0, 0, 0.0)

    segments = _split_segments(text)
    segments = [_apply_segment_alias(_fix_ocr_prefix(s)) for s in segments]
    segments = [s for s in segments if s and len(s) >= 2]

    if not segments:
        return (0, 0, 0.0)

    total = len(segments)
    matched = 0
    confidences: list[float] = []

    for seg in segments:
        slug = _to_slug(seg)
        if not slug:
            continue

        pr = _match_province_slug(slug)
        if pr:
            matched += 1
            confidences.append(pr[2])
            continue

        found = False
        for dists in _districts_by_prov.values():
            dr = _fuzzy_match(slug, dists)
            if dr:
                matched += 1
                confidences.append(dr[2])
                found = True
                break
            stripped = _strip_admin_slug(slug)
            if stripped != slug:
                dr = _fuzzy_match(stripped, dists)
                if dr:
                    matched += 1
                    confidences.append(dr[2])
                    found = True
                    break
        if found:
            continue

        for wards in _wards_by_dist.values():
            wr = _fuzzy_match(slug, wards)
            if wr:
                matched += 1
                confidences.append(wr[2])
                found = True
                break
            stripped = _strip_admin_slug(slug)
            if stripped != slug:
                wr = _fuzzy_match(stripped, wards)
                if wr:
                    matched += 1
                    confidences.append(wr[2])
                    found = True
                    break

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return (matched, total, avg_conf)


def restore_place_diacritics(text: str) -> str:
    """Restore Vietnamese diacritics per comma-separated segment.

    For each segment, attempts a gazetteer match.  If the matched name's
    ASCII-folded form is identical to the segment's, replaces with the
    diacritically-correct version.  This preserves commas and structure.
    """
    if not text or not text.strip():
        return text
    if not _ensure_loaded():
        return text

    segments = _split_segments(text)
    if not segments:
        return text

    corrected = list(segments)
    for i, seg in enumerate(segments):
        seg_stripped = seg.strip()
        if not seg_stripped or len(seg_stripped) < 2:
            continue
        seg_slug = _to_slug(seg_stripped)
        if not seg_slug:
            continue

        best_name: str | None = None

        pr = _match_province_slug(seg_slug)
        if pr and pr[2] >= 0.9:
            best_name = pr[0]
        if not best_name:
            for dists in _districts_by_prov.values():
                dr = _fuzzy_match(seg_slug, dists, max_dist=1)
                if dr and dr[2] >= 0.9:
                    _, (name, _), _ = dr
                    best_name = name
                    break
        if not best_name:
            stripped = _strip_admin_slug(seg_slug)
            if stripped != seg_slug:
                for dists in _districts_by_prov.values():
                    dr = _fuzzy_match(stripped, dists, max_dist=1)
                    if dr and dr[2] >= 0.9:
                        _, (name, _), _ = dr
                        best_name = name
                        break
        if not best_name:
            for wards in _wards_by_dist.values():
                wr = _fuzzy_match(seg_slug, wards, max_dist=1)
                if wr and wr[2] >= 0.9:
                    _, best_name, _ = wr
                    break

        if best_name:
            name_folded = _ascii_fold(best_name)
            seg_folded = _ascii_fold(seg_stripped)
            if name_folded == seg_folded:
                corrected[i] = best_name

    return ", ".join(corrected)


__all__ = [
    "correct_place_text",
    "match_district",
    "match_province",
    "match_ward",
    "restore_place_diacritics",
    "score_place_text",
]
