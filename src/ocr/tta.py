from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TEXT_FIELDS = {"full_name", "place_of_origin", "place_of_residence"}


@dataclass(frozen=True)
class TTAVariant:
    name: str
    image: np.ndarray


def _vertical_band(field_name: str) -> tuple[float, float] | None:
    if field_name == "full_name":
        return (0.16, 0.86)
    if field_name == "place_of_origin":
        return (0.16, 0.84)
    if field_name == "place_of_residence":
        return (0.18, 0.90)
    if field_name == "date_of_birth":
        return (0.10, 0.90)
    if field_name == "id_number":
        return (0.10, 0.92)
    return None


def _append_variant(
    variants: list[TTAVariant],
    seen_keys: set[tuple[str, tuple[int, ...]]],
    name: str,
    image: np.ndarray,
) -> None:
    if image is None or image.size == 0:
        return
    key = (name, tuple(int(dim) for dim in image.shape))
    if key in seen_keys:
        return
    seen_keys.add(key)
    variants.append(TTAVariant(name=name, image=image))


def _crop_by_ratios(
    image: np.ndarray,
    *,
    top_ratio: float = 0.0,
    bottom_ratio: float = 1.0,
    left_ratio: float = 0.0,
    right_ratio: float = 1.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    y1 = max(0, min(height - 1, int(round(height * top_ratio))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom_ratio))))
    x1 = max(0, min(width - 1, int(round(width * left_ratio))))
    x2 = max(x1 + 1, min(width, int(round(width * right_ratio))))
    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size != 0 else image


def generate_ocr_tta_variants(
    image: np.ndarray,
    field_name: str,
    enable_tta: bool = False,
) -> list[TTAVariant]:
    if image is None or image.size == 0:
        return []

    variants: list[TTAVariant] = []
    seen_keys: set[tuple[str, tuple[int, ...]]] = set()
    _append_variant(variants, seen_keys, "base", image.copy())
    if not enable_tta:
        return variants

    center_band = _vertical_band(field_name)
    if center_band is not None:
        top_ratio, bottom_ratio = center_band
        cropped = _crop_by_ratios(image, top_ratio=top_ratio, bottom_ratio=bottom_ratio)
        _append_variant(variants, seen_keys, "center_band", cropped)

    if field_name in TEXT_FIELDS and image.shape[1] >= image.shape[0] * 2.2:
        cropped = _crop_by_ratios(image, left_ratio=0.08, right_ratio=0.98)
        _append_variant(variants, seen_keys, "left_trim", cropped)

    return variants


__all__ = ["TTAVariant", "generate_ocr_tta_variants"]
