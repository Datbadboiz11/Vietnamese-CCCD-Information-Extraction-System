from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ocr.cropping import clamp_bbox, crop_image_xyxy
from src.ocr.types import OCRResult, OCRSegment
from src.ocr.utils import (
    TEXT_LINE_PICK_FIELDS,
    calibrate_ocr_confidence,
    canonicalize_field_name,
    cleanup_ocr_text,
    collapse_whitespace,
    empty_ocr_result,
    estimate_text_confidence,
    extract_value_from_label_text,
    looks_like_label_text,
    normalize_text_for_field,
)


@dataclass(frozen=True)
class LineSegment:
    text: str
    confidence: float
    box_xyxy: np.ndarray
    source: OCRSegment
    index: int


@dataclass(frozen=True)
class TextRegionCandidate:
    kind: str
    text: str
    raw_text: str
    score: float
    box_xyxy: np.ndarray | None
    segments: list[OCRSegment]


def _segment_box_xyxy(segment: OCRSegment) -> np.ndarray | None:
    if not segment.box:
        return None
    try:
        points = np.asarray(segment.box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if points.size < 4:
        return None
    return np.array(
        [
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        ],
        dtype=np.float32,
    )


def _ordered_segments(segments: list[OCRSegment] | None) -> list[LineSegment]:
    ordered: list[LineSegment] = []
    for index, segment in enumerate(segments or []):
        text = collapse_whitespace(segment.text)
        box_xyxy = _segment_box_xyxy(segment)
        if not text or box_xyxy is None:
            continue
        ordered.append(
            LineSegment(
                text=text,
                confidence=float(segment.confidence or 0.0),
                box_xyxy=box_xyxy,
                source=segment,
                index=index,
            )
        )
    ordered.sort(key=lambda item: (float((item.box_xyxy[1] + item.box_xyxy[3]) / 2.0), float(item.box_xyxy[0])))
    return ordered


def _box_width(box_xyxy: np.ndarray) -> float:
    return float(max(1.0, box_xyxy[2] - box_xyxy[0]))


def _box_height(box_xyxy: np.ndarray) -> float:
    return float(max(1.0, box_xyxy[3] - box_xyxy[1]))


def _vertical_gap(upper: np.ndarray, lower: np.ndarray) -> float:
    return float(lower[1] - upper[3])


def _horizontal_overlap_ratio(left: np.ndarray, right: np.ndarray) -> float:
    overlap = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    return overlap / max(1.0, min(_box_width(left), _box_width(right)))


def _boxes_can_merge(current: np.ndarray, nxt: np.ndarray) -> bool:
    gap = _vertical_gap(current, nxt)
    if gap > max(18.0, 1.2 * max(_box_height(current), _box_height(nxt))):
        return False
    overlap_ratio = _horizontal_overlap_ratio(current, nxt)
    left_delta = abs(float(current[0]) - float(nxt[0]))
    return overlap_ratio >= 0.10 or left_delta <= max(_box_width(current), _box_width(nxt)) * 0.40


def _union_boxes(boxes: list[np.ndarray]) -> np.ndarray | None:
    if not boxes:
        return None
    stacked = np.stack(boxes, axis=0)
    return np.array(
        [
            float(np.min(stacked[:, 0])),
            float(np.min(stacked[:, 1])),
            float(np.max(stacked[:, 2])),
            float(np.max(stacked[:, 3])),
        ],
        dtype=np.float32,
    )


def _estimate_inline_box(box_xyxy: np.ndarray, raw_text: str, value_text: str) -> np.ndarray:
    start_ratio = 0.45
    raw_text = collapse_whitespace(raw_text)
    value_text = collapse_whitespace(value_text)
    if ":" in raw_text:
        prefix = raw_text.split(":", 1)[0]
        start_ratio = max(0.18, min(0.92, len(prefix) / max(1, len(raw_text)) - 0.03))
    else:
        raw_words = raw_text.split()
        value_words = value_text.split()
        prefix_words = max(0, len(raw_words) - len(value_words))
        start_ratio = max(0.18, min(0.92, prefix_words / max(1, len(raw_words)) - 0.02))

    width = _box_width(box_xyxy)
    x1 = min(float(box_xyxy[2]) - 4.0, float(box_xyxy[0]) + width * start_ratio)
    return np.array([x1, float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])], dtype=np.float32)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / max(1, len(text))


def _score_candidate(
    text: str,
    raw_text: str,
    avg_confidence: float,
    field_name: str,
    kind: str,
    segment_count: int,
) -> float:
    cleaned_text = cleanup_ocr_text(text, field_name)
    if not cleaned_text:
        return 0.0

    score = calibrate_ocr_confidence(cleaned_text, avg_confidence, field_name)
    score = 0.55 * score + 0.45 * estimate_text_confidence(cleaned_text, field_name)

    if looks_like_label_text(raw_text, field_name):
        score -= 0.25
    if len(cleaned_text.split()) >= 2:
        score += 0.03
    if field_name in {"place_of_origin", "place_of_residence"} and segment_count > 1:
        score += 0.04
    if field_name == "full_name" and segment_count > 1:
        score -= 0.04

    digit_ratio = _digit_ratio(cleaned_text)
    if digit_ratio > 0.30:
        score -= min(0.18, digit_ratio * 0.30)

    if kind == "same-line-after-colon":
        score += 0.06
    elif kind == "next-lines-below-label":
        score += 0.04
    elif kind == "longest-non-label-block":
        score += 0.02

    return max(0.0, min(score, 0.99))


def _make_candidate(
    *,
    kind: str,
    raw_text: str,
    field_name: str,
    box_xyxy: np.ndarray | None,
    segments: list[OCRSegment],
) -> TextRegionCandidate | None:
    cleaned_text = cleanup_ocr_text(raw_text, field_name)
    if not cleaned_text:
        return None
    avg_confidence = (
        sum(float(segment.confidence or 0.0) for segment in segments) / len(segments)
        if segments
        else 0.0
    )
    score = _score_candidate(cleaned_text, raw_text, avg_confidence, field_name, kind, len(segments))
    return TextRegionCandidate(
        kind=kind,
        text=cleaned_text,
        raw_text=collapse_whitespace(raw_text),
        score=score,
        box_xyxy=box_xyxy,
        segments=segments,
    )


def _build_same_line_candidates(lines: list[LineSegment], field_name: str) -> list[TextRegionCandidate]:
    candidates: list[TextRegionCandidate] = []
    for line in lines:
        value_text = extract_value_from_label_text(line.text, field_name)
        if not value_text or value_text == line.text:
            continue
        if ":" not in line.text and not looks_like_label_text(line.text, field_name):
            continue
        inline_box = _estimate_inline_box(line.box_xyxy, line.text, value_text)
        candidate = _make_candidate(
            kind="same-line-after-colon",
            raw_text=value_text,
            field_name=field_name,
            box_xyxy=inline_box,
            segments=[line.source],
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _build_below_label_candidates(lines: list[LineSegment], field_name: str) -> list[TextRegionCandidate]:
    candidates: list[TextRegionCandidate] = []
    for index, line in enumerate(lines):
        if not looks_like_label_text(line.text, field_name):
            continue

        used_segments: list[OCRSegment] = []
        text_parts: list[str] = []
        boxes: list[np.ndarray] = []

        inline_value = extract_value_from_label_text(line.text, field_name)
        if inline_value and inline_value != line.text:
            text_parts.append(inline_value)
            boxes.append(_estimate_inline_box(line.box_xyxy, line.text, inline_value))
            used_segments.append(line.source)

        previous_box = line.box_xyxy
        for next_index in range(index + 1, min(index + 3, len(lines))):
            nxt = lines[next_index]
            if looks_like_label_text(nxt.text, field_name):
                break
            if not _boxes_can_merge(previous_box, nxt.box_xyxy):
                break
            text_parts.append(nxt.text)
            boxes.append(nxt.box_xyxy)
            used_segments.append(nxt.source)
            previous_box = nxt.box_xyxy

        if not text_parts:
            continue

        candidate = _make_candidate(
            kind="next-lines-below-label",
            raw_text=" ".join(text_parts),
            field_name=field_name,
            box_xyxy=_union_boxes(boxes),
            segments=used_segments,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _build_non_label_block_candidates(lines: list[LineSegment], field_name: str) -> list[TextRegionCandidate]:
    candidates: list[TextRegionCandidate] = []
    blocks: list[list[LineSegment]] = []
    current_block: list[LineSegment] = []

    for line in lines:
        if looks_like_label_text(line.text, field_name):
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        if not current_block:
            current_block = [line]
            continue
        previous = current_block[-1]
        if _boxes_can_merge(previous.box_xyxy, line.box_xyxy):
            current_block.append(line)
        else:
            blocks.append(current_block)
            current_block = [line]

    if current_block:
        blocks.append(current_block)

    for block in blocks:
        raw_text = " ".join(segment.text for segment in block)
        candidate = _make_candidate(
            kind="longest-non-label-block",
            raw_text=raw_text,
            field_name=field_name,
            box_xyxy=_union_boxes([segment.box_xyxy for segment in block]),
            segments=[segment.source for segment in block],
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _dedupe_candidates(candidates: list[TextRegionCandidate]) -> list[TextRegionCandidate]:
    best_by_text: dict[str, TextRegionCandidate] = {}
    for candidate in candidates:
        previous = best_by_text.get(candidate.text)
        if previous is None or candidate.score > previous.score:
            best_by_text[candidate.text] = candidate
    return sorted(best_by_text.values(), key=lambda item: (item.score, len(item.text)), reverse=True)


def _has_label_heavy_raw_text(paddle_result: OCRResult, field_name: str) -> bool:
    raw_text = paddle_result.raw_text or ""
    return any(looks_like_label_text(line, field_name) for line in raw_text.splitlines() if line.strip())


def select_best_text_candidate(field_name: str, paddle_result: OCRResult) -> TextRegionCandidate | None:
    canonical_field = canonicalize_field_name(field_name) or field_name
    if canonical_field not in TEXT_LINE_PICK_FIELDS:
        return None

    lines = _ordered_segments(paddle_result.segments)
    if not lines:
        return None

    candidates = _dedupe_candidates(
        _build_same_line_candidates(lines, canonical_field)
        + _build_below_label_candidates(lines, canonical_field)
        + _build_non_label_block_candidates(lines, canonical_field)
    )
    if not candidates:
        return None

    best_candidate = candidates[0]
    base_text = collapse_whitespace(paddle_result.text)
    base_confidence = float(paddle_result.confidence or 0.0)
    raw_label_heavy = _has_label_heavy_raw_text(paddle_result, canonical_field)

    if not base_text:
        return best_candidate
    if looks_like_label_text(base_text, canonical_field) and not looks_like_label_text(best_candidate.text, canonical_field):
        return best_candidate
    if raw_label_heavy and best_candidate.score >= base_confidence - 0.02:
        return best_candidate
    if best_candidate.score >= base_confidence + 0.03:
        return best_candidate
    if len(best_candidate.text) > len(base_text) + 6 and best_candidate.score >= base_confidence - 0.02:
        return best_candidate
    if base_confidence < 0.55 and best_candidate.score >= base_confidence - 0.05:
        return best_candidate
    return None


def _expand_candidate_box(box_xyxy: np.ndarray, image_shape: tuple[int, ...], kind: str) -> np.ndarray:
    width = _box_width(box_xyxy)
    height = _box_height(box_xyxy)
    pad_x = max(4.0, width * (0.08 if kind == "same-line-after-colon" else 0.05))
    pad_y = max(4.0, height * 0.30)
    expanded = np.array(
        [
            float(box_xyxy[0]) - pad_x,
            float(box_xyxy[1]) - pad_y,
            float(box_xyxy[2]) + pad_x,
            float(box_xyxy[3]) + pad_y,
        ],
        dtype=np.float32,
    )
    return clamp_bbox(expanded, image_shape)


def build_refined_paddle_result(field_name: str, paddle_result: OCRResult) -> tuple[OCRResult, TextRegionCandidate | None]:
    candidate = select_best_text_candidate(field_name, paddle_result)
    if candidate is None:
        return paddle_result, None

    normalized_text = normalize_text_for_field(candidate.text, field_name)
    refined = OCRResult(
        text=candidate.text,
        score=float(candidate.score),
        engine=paddle_result.backend,
        raw={"text": paddle_result.raw_text},
        needs_review=candidate.score < 0.5 or normalized_text == "",
        normalized_text=normalized_text,
        segments=paddle_result.segments,
    )
    return refined, candidate


def _pick_better_viet_result(
    field_name: str,
    whole_result: OCRResult,
    subcrop_result: OCRResult,
    candidate_text: str,
) -> OCRResult:
    if not collapse_whitespace(subcrop_result.text):
        return whole_result
    if not collapse_whitespace(whole_result.text):
        return subcrop_result

    whole_text = collapse_whitespace(whole_result.text)
    sub_text = collapse_whitespace(subcrop_result.text)
    whole_label = looks_like_label_text(whole_text, field_name)
    sub_label = looks_like_label_text(sub_text, field_name)

    if whole_label and not sub_label:
        return subcrop_result
    if sub_label and not whole_label:
        return whole_result

    candidate_normalized = normalize_text_for_field(candidate_text, field_name)
    whole_normalized = normalize_text_for_field(whole_text, field_name)
    sub_normalized = normalize_text_for_field(sub_text, field_name)
    if candidate_normalized and sub_normalized == candidate_normalized and whole_normalized != candidate_normalized:
        return subcrop_result

    if subcrop_result.confidence > whole_result.confidence + 0.05:
        return subcrop_result
    if len(sub_text) > len(whole_text) + 6 and subcrop_result.confidence >= whole_result.confidence - 0.03:
        return subcrop_result
    return whole_result if whole_result.confidence >= subcrop_result.confidence else subcrop_result


def _safe_predict(adapter, image: np.ndarray, field_name: str, backend: str) -> OCRResult:
    try:
        return adapter.predict(image, field_name)
    except Exception:
        return empty_ocr_result(backend)


def run_hybrid_field_ocr(
    image: np.ndarray,
    field_name: str,
    paddle_adapter,
    viet_adapter,
) -> tuple[OCRResult, OCRResult]:
    canonical_field = canonicalize_field_name(field_name) or field_name
    paddle_result = _safe_predict(paddle_adapter, image, field_name, "paddleocr")
    if canonical_field not in TEXT_LINE_PICK_FIELDS:
        return _safe_predict(viet_adapter, image, field_name, "vietocr"), paddle_result

    refined_paddle, candidate = build_refined_paddle_result(canonical_field, paddle_result)
    whole_viet = _safe_predict(viet_adapter, image, field_name, "vietocr")
    if candidate is None or candidate.box_xyxy is None:
        return whole_viet, refined_paddle

    crop_box = _expand_candidate_box(candidate.box_xyxy, image.shape, candidate.kind)
    if _box_width(crop_box) < 24 or _box_height(crop_box) < 12:
        return whole_viet, refined_paddle

    subcrop = crop_image_xyxy(image, crop_box)
    if subcrop.size == 0:
        return whole_viet, refined_paddle

    subcrop_viet = _safe_predict(viet_adapter, subcrop, field_name, "vietocr")
    chosen_viet = _pick_better_viet_result(field_name, whole_viet, subcrop_viet, refined_paddle.text)
    return chosen_viet, refined_paddle
