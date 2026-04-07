"""
Module Rectification
Warp thẻ CCCD về kích thước chuẩn 856×540px (tỷ lệ 85.6mm × 54mm).

Chiến lược tìm 4 góc (theo thứ tự ưu tiên):
  1. Contour + approxPolyDP  — primary
  2. Hough Line Detection    — fallback
  3. Bbox crop + padding 5%  — last resort
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

CARD_W = 856
CARD_H = 540


@dataclass
class RectifyResult:
    image: np.ndarray      # ảnh đã warp (856×540)
    method: str            # "contour" | "hough" | "bbox_crop"
    corners: np.ndarray    # 4 góc tìm được, shape (4, 2), dtype float32
    failed: bool = False   # True nếu phải dùng last resort


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm: top-left, top-right, bottom-right, bottom-left."""
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def _warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([
        [0,          0         ],
        [CARD_W - 1, 0         ],
        [CARD_W - 1, CARD_H - 1],
        [0,          CARD_H - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (CARD_W, CARD_H))


def _try_contour(image: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    roi     = image[ry1:ry2, rx1:rx2]
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 50, 150)
    edged   = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners[:, 0] += rx1
            corners[:, 1] += ry1
            return corners
    return None


def _try_hough(image: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    roi     = image[y1:y2, x1:x2]
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=60,
                             minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None

    h_lines, v_lines = [], []
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        angle = abs(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
        if angle < 20 or angle > 160:
            h_lines.append(line[0])
        elif 70 < angle < 110:
            v_lines.append(line[0])

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    v_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bottom = h_lines[0], h_lines[-1]
    left, right = v_lines[0], v_lines[-1]

    def intersect(l1, l2):
        x1_, y1_, x2_, y2_ = l1
        x3,  y3,  x4,  y4  = l2
        denom = (x1_ - x2_) * (y3 - y4) - (y1_ - y2_) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1_ - x3) * (y3 - y4) - (y1_ - y3) * (x3 - x4)) / denom
        return [x1_ + t * (x2_ - x1_) + x1, y1_ + t * (y2_ - y1_) + y1]

    pts = [intersect(top, left), intersect(top, right),
           intersect(bottom, right), intersect(bottom, left)]
    if any(p is None for p in pts):
        return None
    return np.array(pts, dtype=np.float32)


def _bbox_crop(image: np.ndarray, bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    corners = np.array([[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]], dtype=np.float32)
    warped  = cv2.resize(image[ry1:ry2, rx1:rx2], (CARD_W, CARD_H))
    return warped, corners


def rectify(image: np.ndarray, bbox: tuple) -> RectifyResult:
    corners = _try_contour(image, bbox)
    if corners is not None:
        ordered = _order_corners(corners)
        return RectifyResult(image=_warp(image, ordered), method="contour", corners=ordered)

    corners = _try_hough(image, bbox)
    if corners is not None:
        ordered = _order_corners(corners)
        return RectifyResult(image=_warp(image, ordered), method="hough", corners=ordered)

    warped, corners = _bbox_crop(image, bbox)
    return RectifyResult(image=warped, method="bbox_crop", corners=corners, failed=True)
