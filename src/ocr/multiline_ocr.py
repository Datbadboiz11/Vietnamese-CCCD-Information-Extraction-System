"""
Multi-line OCR orchestrator for CCCD address/origin fields.

Combines text line detection (PaddleOCR det) with single-line recognition
(VietOCR) to handle multi-line fields that VietOCR alone hallucinates on.

Flow:
    1. detect_text_lines()  — split crop into per-line crops
    2. strip label          — remove "Nơi thường trú:", "Nguyên quán:", etc.
    3. VietOCR per line     — recognize each line separately
    4. join text            — combine lines with ", "
    5. quality check        — fallback to whole-crop if per-line result is worse
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from src.ocr.text_detection import (
    PaddleTextDetector,
    TextLineDetectionResult,
    _det_text_is_label,
    detect_text_lines,
    is_multi_line_field,
    trim_label_from_crop,
)
from src.ocr.types import OCRResult
from src.ocr.utils import (
    calibrate_ocr_confidence,
    canonicalize_field_name,
    cleanup_ocr_text,
    collapse_whitespace,
    looks_like_label_text,
    looks_suspicious_for_field,
    normalize_text_for_field,
    strip_known_field_prefix,
)

LOGGER = logging.getLogger(__name__)

_FINETUNE_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "vietocr_cccd.pth"
LINE_SEPARATOR = ", "
_ADMIN_KEYWORDS = (
    "xã", "phường", "thị xã", "thị trấn", "huyện", "quận", "tỉnh", "thành phố",
)


def _strip_label_from_line(text: str, field_name: str | None) -> str:
    """Remove field label prefix if the line looks like 'Label: value'."""
    stripped = strip_known_field_prefix(text, field_name)
    if not stripped or looks_like_label_text(stripped, field_name):
        return ""
    return stripped


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.lower())
    folded = folded.replace("đ", "d").replace("đ", "d")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded


_CROSS_FIELD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sinh\s*n?g?[aà]?y?", re.IGNORECASE),
    re.compile(r"date\s*of\s*birth", re.IGNORECASE),
    re.compile(r"d[o0]b", re.IGNORECASE),
    re.compile(r"^s[o6ô][\s:]*$", re.IGNORECASE),
    re.compile(r"^n[o0][\s.:]*$", re.IGNORECASE),
    re.compile(r"^gi[o0o]i\s*tinh", re.IGNORECASE),
    re.compile(r"^sex[\s:]*$", re.IGNORECASE),
    re.compile(r"^qu[o6ô]c\s*t[i1]ch", re.IGNORECASE),
    re.compile(r"^nationality", re.IGNORECASE),
    re.compile(r"^h[o0ô]\s*v[aà]\s*t[eê]n", re.IGNORECASE),
    re.compile(r"p[lh]?lac[eo]?\s+[ao][fl1]\b", re.IGNORECASE),
    re.compile(r"place\s+[ao][fl1]\b", re.IGNORECASE),
    re.compile(r"n[oơ]i\s*th[uư][oơ]?ng\s*tr[uú]", re.IGNORECASE),
    re.compile(r"n[oơ]i\s*[dđ]?k?h?k?\s*th[uư]", re.IGNORECASE),
    re.compile(r"nguy[eê]n\s*qu[aá]n", re.IGNORECASE),
    re.compile(r"qu[aâeê]\s*qu[aâ]n", re.IGNORECASE),
    re.compile(r"full\s*name", re.IGNORECASE),
    re.compile(r"surname", re.IGNORECASE),
    re.compile(r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*$"),
    re.compile(r"^\d{10,12}\s*$"),
]

_VIET_DIACRITIC = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)

_DATE_LIKE = re.compile(r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$")


def _count_diacritics(text: str) -> int:
    return len(_VIET_DIACRITIC.findall(text))


def _pick_line_result(
    viet_text: str,
    viet_score: float,
    paddle_text: str,
    paddle_score: float,
) -> tuple[str, float]:
    """Pick the better OCR result for a single line based on Vietnamese diacritics."""
    viet_d = _count_diacritics(viet_text)
    paddle_d = _count_diacritics(paddle_text)

    if paddle_d >= viet_d + 2:
        return paddle_text, max(paddle_score, viet_score)
    if viet_d >= paddle_d + 2:
        return viet_text, max(viet_score, paddle_score)

    viet_len = len(viet_text.strip())
    paddle_len = len(paddle_text.strip())

    if paddle_len > viet_len + 3 and paddle_d >= viet_d:
        return paddle_text, max(paddle_score, viet_score)
    if viet_len > paddle_len + 3 and viet_d >= paddle_d:
        return viet_text, max(viet_score, paddle_score)

    if viet_score >= paddle_score:
        return viet_text, viet_score
    return paddle_text, paddle_score


_LABEL_STRIP_RE: list[re.Pattern[str]] = [
    re.compile(r"p[lh]?lac[eo]?\s+[ao][fl1]\s+\w+", re.IGNORECASE),
    re.compile(r"place\s+[ao][fl1]\s+\w+", re.IGNORECASE),
    re.compile(r"qu[eê]?\s*qu[aâ]n", re.IGNORECASE),
    re.compile(r"nguy[eê]n\s*qu[aâ]n", re.IGNORECASE),
    re.compile(r"n[oơ]i?\s*d?k?h?k?\s*th[uư]?r?[oơờ]?ng\s*tr[uư]", re.IGNORECASE),
    re.compile(r"residen\w*", re.IGNORECASE),
    re.compile(r"origin\w*", re.IGNORECASE),
    re.compile(r"orgin\w*", re.IGNORECASE),
    re.compile(r"of\s+orgin\w*", re.IGNORECASE),
]


def _strip_label_from_det_text(det_text: str) -> str:
    """Strip label prefix from PaddleOCR det_text, return content after it."""
    folded = _ascii_fold(det_text)
    best_end = 0
    for pat in _LABEL_STRIP_RE:
        m = pat.search(folded)
        if m and m.start() < len(folded) * 0.65:
            best_end = max(best_end, m.end())
    if best_end > 0 and best_end <= len(det_text):
        remainder = det_text[best_end:].lstrip(" :/,.-")
        return remainder
    return ""


def _is_pure_label(det_text: str) -> bool:
    """True when det_text is *only* a field label with no address data after it."""
    if not _VIET_DIACRITIC.search(det_text):
        if _ENGLISH_LABEL_RE.search(det_text) or _GARBLED_LABEL_RE.search(det_text):
            stripped = _strip_label_from_det_text(det_text)
            if not stripped or not _VIET_DIACRITIC.search(stripped):
                return True
    stripped = _strip_label_from_det_text(det_text)
    return len(stripped) < 4


def _is_cross_field_text(text: str) -> bool:
    """Text that belongs to a neighboring field (birth label, date value, etc.)."""
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return False
    folded = _ascii_fold(cleaned)
    for pat in _CROSS_FIELD_PATTERNS:
        if pat.search(folded):
            return True
    if _DATE_LIKE.match(cleaned.strip()):
        return True
    return False


_ENGLISH_LABEL_RE = re.compile(
    r"\b(?:place|resid\w*|origin\w*|orgin\w*|birth|surname|full\s*name|nationality|sex)\b",
    re.IGNORECASE,
)
_GARBLED_LABEL_RE = re.compile(
    r"(?:p[ilh]?[aei]c[eo]?|ofonig|dforig|ofresid|oforgin|of\s*orgin)",
    re.IGNORECASE,
)

_EMBEDDED_LABEL_RE = re.compile(
    r"[\(\[\{]?\s*(?:"
    r"[Pp][ilh]?[aei]c[eo]?\w{0,20}|"
    r"[Nn][oơ]i?\s*[DdĐđ]?K?H?[Kk]?\s*th[uư]\w*\s*tr[uú]\w*|"
    r"[Nn]guy[eê]n\s*qu[aá]n\w*|"
    r"[Pp]lace\s+[oO][fF]\s*\w+|"
    r"of\s*orgin\w*|of\s*origin\w*|"
    r"residen\w*|orgin\w*"
    r")\s*[\)\]\}:/ ]*",
    re.IGNORECASE,
)


def _strip_embedded_label(text: str) -> str:
    """Remove label fragments embedded in the middle of recognized text."""
    result = _EMBEDDED_LABEL_RE.sub(" ", text)
    result = re.sub(r"\s+", " ", result).strip(" ,./;:-")
    return result


def _has_embedded_label(text: str) -> bool:
    """Check if text contains embedded field label fragments."""
    return bool(_EMBEDDED_LABEL_RE.search(text))


def _is_line_garbage(text: str, field_name: str | None) -> bool:
    """Check if a recognized line is likely OCR garbage."""
    if not text or not text.strip():
        return True
    cleaned = collapse_whitespace(text)
    if len(cleaned) <= 1:
        return True
    digit_count = sum(ch.isdigit() for ch in cleaned)
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    if alnum_count > 0 and digit_count / alnum_count > 0.7 and digit_count >= 6:
        return True
    if _is_cross_field_text(cleaned):
        return True
    words = cleaned.split()
    if len(words) == 1 and len(cleaned) > 10 and not _VIET_DIACRITIC.search(cleaned):
        return True

    # General English label detection: no Vietnamese diacritics + English label keywords
    if not _VIET_DIACRITIC.search(cleaned):
        if _ENGLISH_LABEL_RE.search(cleaned) or _GARBLED_LABEL_RE.search(cleaned):
            return True

    stripped_num = cleaned.lstrip("0123456789 ")
    folded_stripped = _ascii_fold(stripped_num)
    if folded_stripped.startswith(("place", "plac", "phace", "phlace", "placo", "piace", "piac")):
        return True
    if re.match(r"^p[ilh]?[ai]c[eo]?\s", folded_stripped):
        return True
    if re.search(r"[ao][fl1]\s*(resid|orig|birth)", folded_stripped):
        return True
    if re.search(r"d?f?orig", folded_stripped):
        return True
    if re.search(r"of\s*orgin|^orgin", folded_stripped):
        return True
    return False


def _is_marginal_line(line: Any, crop_h: int, field_name: str | None) -> bool:
    """Check if a text line sits in the top/bottom margin of the crop.

    Lines at the very top or bottom are likely bleeding from an adjacent
    field on the card. We apply extra scrutiny: short lines, date-like
    text, or label-only text in the margins are filtered out.
    """
    if crop_h <= 0:
        return False
    bbox = line.bbox_xyxy
    center_y = float(bbox[1] + bbox[3]) / 2.0
    rel_y = center_y / crop_h

    det = line.det_text or ""

    if rel_y < 0.15:
        if _is_cross_field_text(det):
            return True
    if rel_y > 0.85 and field_name in ("place_of_origin", "origin"):
        if _det_text_is_label(det):
            return True
        if _is_cross_field_text(det):
            return True
    return False


def _recognize_lines(
    det_result: TextLineDetectionResult,
    field_name: str | None,
    vietocr: Any,
    crop_h: int = 0,
) -> list[tuple[str, float]]:
    """Run VietOCR on each detected line, optionally improve with PaddleOCR det_text."""
    line_results: list[tuple[str, float]] = []

    for i, line in enumerate(det_result.lines):
        if line.crop is None or line.crop.size == 0:
            continue

        if crop_h > 0 and _is_marginal_line(line, crop_h, field_name):
            LOGGER.debug("Skipping marginal line %d: %s", i, line.det_text)
            continue

        if line.det_text and _det_text_is_label(line.det_text):
            if _is_pure_label(line.det_text):
                LOGGER.debug("Skipping pure-label line %d: %s", i, line.det_text)
                continue
            content = _strip_label_from_det_text(line.det_text)
            if content and len(content) >= 3:
                LOGGER.debug("Label+content line %d: stripped to '%s'", i, content)
                line_results.append((collapse_whitespace(content), line.detection_confidence))
                continue

        ocr_out: OCRResult = vietocr.recognize(line.crop, field_name=field_name)
        text = cleanup_ocr_text(ocr_out.text or "", field_name)
        text = _strip_label_from_line(text, field_name)
        text = _strip_embedded_label(text)

        if _is_line_garbage(text, field_name):
            continue

        score = float(ocr_out.score)

        if line.det_text:
            paddle_text = cleanup_ocr_text(line.det_text, field_name)
            paddle_text = _strip_label_from_line(paddle_text, field_name)
            if paddle_text and not _is_line_garbage(paddle_text, field_name):
                pd = _count_diacritics(paddle_text)
                vd = _count_diacritics(text)
                if pd > vd + 1:
                    text = paddle_text
                    score = max(score, line.detection_confidence)
                elif (len(paddle_text) > len(text) + 3 and pd >= vd):
                    text = paddle_text
                    score = max(score, line.detection_confidence)

        line_results.append((collapse_whitespace(text), score))

    return line_results


def _whole_crop_recognize(
    field_crop: np.ndarray,
    field_name: str | None,
    vietocr: Any,
) -> OCRResult:
    """Fallback: recognize the entire crop as-is (old behavior)."""
    return vietocr.recognize(field_crop, field_name=field_name)


def _structure_score(text: str) -> tuple[int, int, int, int]:
    cleaned = collapse_whitespace(text).lower()
    keyword_hits = sum(1 for keyword in _ADMIN_KEYWORDS if keyword in cleaned)
    comma_count = cleaned.count(",")
    segments = [seg.strip() for seg in cleaned.split(",") if seg.strip()]
    usable_segments = sum(1 for seg in segments if len(seg) >= 2)
    alpha_count = sum(ch.isalpha() for ch in cleaned)
    return (keyword_hits, min(comma_count, 4), usable_segments, alpha_count)


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.lower())
    folded = folded.replace("đ", "d").replace("Đ", "d")
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return collapse_whitespace(folded)


def _label_contamination(text: str) -> int:
    """Return penalty score for candidates containing embedded label text."""
    if _has_embedded_label(text):
        return 1
    return 0


def _garbage_penalty(text: str) -> tuple[int, int]:
    cleaned = collapse_whitespace(text)
    if not cleaned:
        return (99, 99)

    raw_tokens = re.findall(r"\w+", cleaned, flags=re.UNICODE)
    if not raw_tokens:
        return (99, 99)

    short_or_noisy = 0
    admin_like = 0
    for token in raw_tokens:
        folded = _ascii_fold(token)
        if len(folded) <= 1 or sum(ch.isalpha() for ch in folded) <= 1:
            short_or_noisy += 1
        if any(keyword in folded for keyword in ("xa", "phuong", "huyen", "quan", "tinh", "thanh", "pho", "thi", "tran", "tp")):
            admin_like += 1
    return (short_or_noisy, -admin_like)


def _normalized_structure_bonus(text: str, field_name: str | None) -> tuple[int, int]:
    canonical = canonicalize_field_name(field_name)
    if canonical not in {"place_of_origin", "place_of_residence"}:
        return (0, 0)
    try:
        from src.parsing.vn_places import correct_place_text

        corrected = correct_place_text(text)
    except Exception:
        return (0, 0)

    if not corrected or corrected == text:
        return (0, 0)

    before = _structure_score(text)
    after = _structure_score(corrected)
    structure_gain = max(0, after[0] - before[0]) + max(0, after[1] - before[1])
    similarity = SequenceMatcher(None, collapse_whitespace(text).lower(), collapse_whitespace(corrected).lower()).ratio()
    if structure_gain <= 0 or similarity < 0.72:
        return (0, 0)
    return (structure_gain, int(round(similarity * 100)))


def _gazetteer_match_score(text: str) -> tuple[int, float]:
    """Return (matched_segments, avg_confidence) from gazetteer matching.

    Cached per text to avoid repeated expensive gazetteer lookups during sorting.
    """
    try:
        from src.parsing.vn_places import score_place_text

        matched, total, avg_conf = score_place_text(text)
        return (matched, avg_conf)
    except Exception:
        return (0, 0.0)


def _method_bias(method: str) -> int:
    if method == "paddleocr_full_trimmed":
        return 5
    if method == "paddleocr_full":
        return 4
    if method.startswith("multiline_tail"):
        return 3
    if method.startswith("multiline_"):
        return 2
    if method == "whole_crop_trimmed":
        return 1
    return 0


def _build_line_candidates(
    line_results: list[tuple[str, float]],
    detection_method: str,
) -> list[tuple[str, float, str]]:
    if not line_results:
        return [("", 0.0, f"multiline_{detection_method}")]

    candidates: list[tuple[str, float, str]] = []
    full_text = LINE_SEPARATOR.join(text for text, _ in line_results)
    full_score = float(np.mean([score for _, score in line_results]))
    candidates.append((full_text, full_score, f"multiline_{detection_method}"))

    for tail_len in (2, 3):
        if len(line_results) < tail_len:
            continue
        tail = line_results[-tail_len:]
        tail_text = LINE_SEPARATOR.join(text for text, _ in tail)
        tail_score = float(np.mean([score for _, score in tail]))
        candidates.append((tail_text, tail_score, f"multiline_tail{tail_len}_{detection_method}"))

    return candidates


def _pick_best_candidate(
    candidates: list[tuple[str, float, str]],
    field_name: str | None,
) -> tuple[str, float, str]:
    """Pick the best candidate balancing structure reliability and Vietnamese diacritics.

    PaddleOCR full is structurally most reliable (no hallucination on multi-line)
    and generally better at Vietnamese diacritics. Per-line VietOCR only wins when
    it has significantly more diacritics AND longer text.
    """
    valid = [
        (t, s, m) for t, s, m in candidates
        if t and not looks_suspicious_for_field(t, field_name)
    ]
    if not valid:
        valid = [(t, s, m) for t, s, m in candidates if t]
    if not valid:
        return "", 0.0, "all_empty"

    canonical = canonicalize_field_name(field_name)
    use_gazetteer = canonical in {"place_of_origin", "place_of_residence"}

    gaz_cache: dict[str, tuple[int, float]] = {}

    def _cached_gaz(text: str) -> tuple[int, float]:
        if text not in gaz_cache:
            gaz_cache[text] = _gazetteer_match_score(text) if use_gazetteer else (0, 0.0)
        return gaz_cache[text]

    valid.sort(
        key=lambda c: (
            -_label_contamination(c[0]),
            _cached_gaz(c[0]),
            _structure_score(c[0]),
            _normalized_structure_bonus(c[0], field_name),
            tuple(-value for value in _garbage_penalty(c[0])),
            _count_diacritics(c[0]),
            _method_bias(c[2]),
            c[1],
        ),
        reverse=True,
    )
    best = valid[0]
    for candidate in valid[1:]:
        cand_label = _label_contamination(candidate[0])
        best_label = _label_contamination(best[0])
        if cand_label > best_label:
            continue
        if cand_label < best_label:
            best = candidate
            continue
        cand_gaz = _cached_gaz(candidate[0])
        best_gaz = _cached_gaz(best[0])
        if cand_gaz[0] > best_gaz[0]:
            best = candidate
            continue
        if cand_gaz[0] < best_gaz[0]:
            continue
        if _structure_score(candidate[0]) < _structure_score(best[0]):
            continue
        if _normalized_structure_bonus(candidate[0], field_name) > _normalized_structure_bonus(best[0], field_name):
            best = candidate
            continue
        if _count_diacritics(candidate[0]) >= _count_diacritics(best[0]) + 3:
            best = candidate
            break
    return best[0], best[1], best[2]


def recognize_multiline_field(
    field_crop: np.ndarray,
    field_name: str | None,
    vietocr: Any,
    detector: PaddleTextDetector | None = None,
    paddleocr_recognizer: Any | None = None,
) -> OCRResult:
    """
    Full multi-line OCR pipeline for address/origin fields.

    Compares three candidates and picks the one with the most Vietnamese diacritics:
      1. Per-line (PaddleOCR det + VietOCR rec per line)
      2. Whole-crop VietOCR
      3. PaddleOCR full pipeline (det+rec, sees full context)
    """
    canonical = canonicalize_field_name(field_name)

    if not is_multi_line_field(canonical):
        return vietocr.recognize(field_crop, field_name=canonical)

    # VietOCR v2 is a single-line model — always use the full multiline
    # pipeline (PaddleOCR line detection + per-line VietOCR) for address/origin.

    if field_crop is None or field_crop.size == 0:
        return OCRResult(
            text="", score=0.0, engine="multiline",
            needs_review=True, normalized_text="",
        )

    # Fast path: current fine-tuned VietOCR/PP-OCRv5 models are trained on
    # whole field crops. Prefer direct whole-crop recognition when it yields a
    # non-suspicious place candidate; line detection is slower and can pull in
    # neighboring label/noise for address/origin crops.
    whole_result = _whole_crop_recognize(field_crop, canonical, vietocr)
    whole_text = cleanup_ocr_text(whole_result.text or "", canonical)
    whole_text = strip_known_field_prefix(whole_text, canonical)
    whole_text = _strip_embedded_label(whole_text)
    whole_score = float(whole_result.score)

    paddle_full_text = ""
    paddle_full_score = 0.0
    if paddleocr_recognizer is not None:
        try:
            paddle_out = paddleocr_recognizer.recognize(field_crop, field_name=canonical)
            paddle_full_text = cleanup_ocr_text(paddle_out.text or "", canonical)
            paddle_full_text = strip_known_field_prefix(paddle_full_text, canonical)
            paddle_full_score = float(paddle_out.score)
        except Exception as exc:
            LOGGER.warning("PaddleOCR full recognition failed: %s", exc)

    direct_candidates: list[tuple[str, float, str]] = [
        (whole_text, whole_score, "whole_crop"),
        (paddle_full_text, paddle_full_score, "paddleocr_full"),
    ]
    direct_valid = [
        candidate for candidate in direct_candidates
        if candidate[0] and not looks_suspicious_for_field(candidate[0], canonical)
    ]
    if direct_valid:
        chosen_text, chosen_score, method = _pick_best_candidate(direct_candidates, canonical)
        calibrated = calibrate_ocr_confidence(chosen_text, chosen_score, canonical)
        normalized = normalize_text_for_field(chosen_text, canonical)
        return OCRResult(
            text=chosen_text,
            score=calibrated,
            engine="multiline",
            raw={
                "method": method,
                "detection_method": "skipped_whole_crop_fast_path",
                "num_lines": 0,
                "num_valid_lines": 0,
                "per_line_texts": [],
                "per_line_scores": [],
                "line_candidates": [],
                "whole_crop_text": whole_text,
                "whole_crop_score": whole_score,
                "paddle_full_text": paddle_full_text,
                "paddle_full_score": paddle_full_score,
                "warnings": [],
            },
            needs_review=calibrated < 0.5 or not normalized,
            normalized_text=normalized,
        )

    # --- Candidate 1: per-line (PaddleOCR det + VietOCR rec) ---
    det_result = detect_text_lines(field_crop, canonical, detector=detector)
    crop_h = field_crop.shape[0] if field_crop is not None else 0
    line_results = _recognize_lines(det_result, canonical, vietocr, crop_h=crop_h)
    line_candidates = _build_line_candidates(line_results, det_result.method)

    # --- Candidate 2: whole-crop VietOCR ---

    # --- Candidate 3: PaddleOCR full pipeline ---
    if paddleocr_recognizer is not None:
        if len(paddle_full_text) < 5 and paddleocr_recognizer is not None:
            try:
                paddle_out_default = paddleocr_recognizer.recognize(field_crop, field_name=None)
                default_text = cleanup_ocr_text(paddle_out_default.text or "", canonical)
                default_text = strip_known_field_prefix(default_text, canonical)
                if len(default_text) > len(paddle_full_text):
                    paddle_full_text = default_text
                    paddle_full_score = float(paddle_out_default.score)
            except Exception:
                pass

    # --- Candidate 4/5: OCR on crop with label area trimmed ---
    trimmed_crop = trim_label_from_crop(field_crop, canonical, detector=detector)
    trimmed_whole_text = ""
    trimmed_whole_score = 0.0
    trimmed_paddle_text = ""
    trimmed_paddle_score = 0.0
    if trimmed_crop is not None and trimmed_crop.size > 0 and trimmed_crop.shape != field_crop.shape:
        try:
            trimmed_whole = _whole_crop_recognize(trimmed_crop, canonical, vietocr)
            trimmed_whole_text = cleanup_ocr_text(trimmed_whole.text or "", canonical)
            trimmed_whole_text = strip_known_field_prefix(trimmed_whole_text, canonical)
            trimmed_whole_score = float(trimmed_whole.score)
        except Exception as exc:
            LOGGER.warning("Whole-crop OCR on trimmed crop failed: %s", exc)
        if paddleocr_recognizer is not None:
            try:
                trimmed_paddle = paddleocr_recognizer.recognize(trimmed_crop, field_name=canonical)
                trimmed_paddle_text = cleanup_ocr_text(trimmed_paddle.text or "", canonical)
                trimmed_paddle_text = strip_known_field_prefix(trimmed_paddle_text, canonical)
                trimmed_paddle_score = float(trimmed_paddle.score)
            except Exception as exc:
                LOGGER.warning("PaddleOCR full recognition on trimmed crop failed: %s", exc)

    # Prefer whole-crop recognition for place fields. Line detection often
    # captures neighboring labels/noise and can produce long, plausible-looking
    # garbage that beats cleaner whole-crop candidates by structure score.
    direct_candidates: list[tuple[str, float, str]] = [
        (whole_text, whole_score, "whole_crop"),
        (paddle_full_text, paddle_full_score, "paddleocr_full"),
        (trimmed_whole_text, trimmed_whole_score, "whole_crop_trimmed"),
        (trimmed_paddle_text, trimmed_paddle_score, "paddleocr_full_trimmed"),
    ]
    direct_valid = [
        candidate for candidate in direct_candidates
        if candidate[0] and not looks_suspicious_for_field(candidate[0], canonical)
    ]
    if direct_valid:
        candidates = direct_candidates
    else:
        candidates = [
            *line_candidates,
            *direct_candidates,
        ]
    chosen_text, chosen_score, method = _pick_best_candidate(candidates, canonical)

    calibrated = calibrate_ocr_confidence(chosen_text, chosen_score, canonical)
    normalized = normalize_text_for_field(chosen_text, canonical)

    return OCRResult(
        text=chosen_text,
        score=calibrated,
        engine="multiline",
        raw={
            "method": method,
            "detection_method": det_result.method,
            "num_lines": len(det_result.lines),
            "num_valid_lines": len(line_results),
            "per_line_texts": [t for t, _ in line_results],
            "per_line_scores": [s for _, s in line_results],
            "line_candidates": [
                {"text": text, "score": score, "method": method_name}
                for text, score, method_name in line_candidates
            ],
            "whole_crop_text": whole_text,
            "whole_crop_score": whole_score,
            "paddle_full_text": paddle_full_text,
            "paddle_full_score": paddle_full_score,
            "warnings": det_result.warnings,
        },
        needs_review=calibrated < 0.5 or not normalized,
        normalized_text=normalized,
    )


__all__ = ["recognize_multiline_field"]
