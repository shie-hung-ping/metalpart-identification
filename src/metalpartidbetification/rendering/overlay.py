"""結果圖繪製：全部零件標註，或只突顯單一零件（純 OpenCV，不依賴 GUI）。"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import DetectedPart, Roi

ROI_COLOR = (255, 160, 0)        # ROI 框：藍色 (BGR)
SUSPECT_COLOR = (0, 215, 255)    # 可疑相黏：黃色
SELECT_COLOR = (0, 0, 255)       # 單選突顯：紅色


def _palette(i: int) -> tuple[int, int, int]:
    """依編號給每個零件不同但固定的顏色（HSV 色相環平均分配）。"""
    hsv = np.uint8([[[(i * 47) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _thickness(image: np.ndarray) -> int:
    """線條粗細隨影像尺寸調整。"""
    return max(2, round(max(image.shape[:2]) / 600))


def _put_label(img: np.ndarray, text: str, center: tuple[float, float], color, scale: float) -> None:
    """在重心處寫編號（加黑邊方便在任何背景上閱讀）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = max(2, int(scale * 2))
    (w, h), _ = cv2.getTextSize(text, font, scale, t)
    org = (int(center[0] - w / 2), int(center[1] + h / 2))
    cv2.putText(img, text, org, font, scale, (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, t, cv2.LINE_AA)


def draw_all(image: np.ndarray, parts: list[DetectedPart], roi: Roi | None = None) -> np.ndarray:
    """畫出所有零件輪廓與編號；有疑慮的零件用黃色（無法判定加「?」，面積推估多片加「xN」）。"""
    out = image.copy()
    th = _thickness(out)
    scale = th * 0.6
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), ROI_COLOR, th)
    for p in parts:
        color = SUSPECT_COLOR if p.suspicious else _palette(p.id)
        cv2.drawContours(out, [p.contour], -1, color, th + 1, cv2.LINE_AA)
        if p.source == "unresolved":
            label = f"{p.id}?"
        elif p.estimated_count > 1:
            label = f"{p.id} x{p.estimated_count}"
        else:
            label = str(p.id)
        _put_label(out, label, p.centroid, color, scale)
    return out


def draw_single(image: np.ndarray, parts: list[DetectedPart], part_id: int, roi: Roi | None = None) -> np.ndarray:
    """只突顯指定零件：其他區域變暗，選中零件保持原亮度並加上輪廓與外框。"""
    part = next((p for p in parts if p.id == part_id), None)
    if part is None:
        return draw_all(image, parts, roi)
    th = _thickness(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [part.contour], -1, 255, -1)
    out = (image.astype(np.float32) * 0.3).astype(np.uint8)   # 整張變暗
    out[mask > 0] = image[mask > 0]                             # 選中零件恢復原亮度
    cv2.drawContours(out, [part.contour], -1, SELECT_COLOR, th + 1, cv2.LINE_AA)
    x, y, w, h = part.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), SELECT_COLOR, max(1, th - 1))
    _put_label(out, str(part.id), part.centroid, SELECT_COLOR, th * 0.6)
    return out
