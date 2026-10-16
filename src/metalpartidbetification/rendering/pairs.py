"""配對比較圖：原圖（左）與鏡像圖（右）並排，依逐片判定著色：

  綠 P3   高信心，配對正確
  黃 P3?  無法確定是否配對成功
  紅 P3!  疑似可配對到多個零件（畫線連到每個候選）
  灰 #8x  無法對應
另外：藍色虛線框 = ROI（鏡像圖上的 ROI 跟著鏡像、不跟著位移）；ROI! = 位移可能過大；LOST = 可能遺失。
（OpenCV 不支援中文，圖上只用顏色與英數標記，中文說明放在 GUI 圖例與報告。）
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import (
    LEVEL_GREEN, LEVEL_RED, LEVEL_YELLOW, ROI_LOST, ROI_ORIGINAL, ROI_PARTIAL, DetectedPart, MatchResult, Roi,
)
from .overlay import _put_label, _thickness

LEVEL_COLORS = {LEVEL_GREEN: (60, 190, 60), LEVEL_YELLOW: (0, 215, 255), LEVEL_RED: (50, 50, 235)}
GRAY = (170, 170, 170)
ROI_COLOR = (255, 160, 0)
TAG_COLORS = {ROI_PARTIAL: (0, 140, 255), ROI_LOST: (180, 0, 255), ROI_ORIGINAL: (200, 200, 200)}
TAG_TEXT = {ROI_PARTIAL: "ROI!", ROI_LOST: "LOST", ROI_ORIGINAL: "ROI0"}
GAP = 20


def draw_pairs(img_a: np.ndarray, parts_a: list[DetectedPart], img_b: np.ndarray, parts_b: list[DetectedPart],
               match: MatchResult, selected: int | None = None, roi_a: Roi | None = None,
               legend: bool = False) -> np.ndarray:
    """selected = 配對編號：只突顯該組（以及紅色時的所有候選），其他區域變暗。"""
    a, b = img_a.copy(), img_b.copy()
    by_a = {p.id: p for p in parts_a}
    by_b = {p.id: p for p in parts_b}
    th = _thickness(a)
    scale = th * 0.6
    sel = next((p for p in match.pairs if p.pair_id == selected), None) if selected is not None else None
    if sel is not None:
        a = (a.astype(np.float32) * 0.3).astype(np.uint8)
        b = (b.astype(np.float32) * 0.3).astype(np.uint8)
        keep_a = {sel.a_id} | {aid for aid, _ in sel.rival_a}
        keep_b = {sel.b_id} | {bid for bid, _ in sel.candidates}
        for canvas, src, ids, lookup in ((a, img_a, keep_a, by_a), (b, img_b, keep_b, by_b)):
            for pid in ids:
                if pid in lookup:
                    m = np.zeros(canvas.shape[:2], np.uint8)
                    cv2.drawContours(m, [lookup[pid].contour], -1, 255, -1)
                    canvas[m > 0] = src[m > 0]

    # ROI 虛線框
    if roi_a is not None:
        _dashed_rect(a, roi_a, ROI_COLOR, th)
    if match.roi_b is not None:
        _dashed_rect(b, match.roi_b, ROI_COLOR, th)

    # 配對（綠／黃／紅）
    lines = []
    for pr in match.pairs:
        if sel is not None and pr.pair_id != sel.pair_id:
            continue
        color = LEVEL_COLORS[pr.level]
        mark = {LEVEL_GREEN: "", LEVEL_YELLOW: "?", LEVEL_RED: "!"}[pr.level]
        for canvas, part in ((a, by_a.get(pr.a_id)), (b, by_b.get(pr.b_id))):
            if part is not None:
                cv2.drawContours(canvas, [part.contour], -1, color, th + 1, cv2.LINE_AA)
                _put_label(canvas, f"P{pr.pair_id}{mark}", part.centroid, color, scale)
        if pr.level == LEVEL_RED:                         # 紅色：連到每個候選
            pa = by_a.get(pr.a_id)
            for bid, _ in pr.candidates:
                if pa is not None and bid in by_b:
                    if bid != pr.b_id:
                        cv2.drawContours(b, [by_b[bid].contour], -1, color, th, cv2.LINE_AA)
                    lines.append((pa.centroid, by_b[bid].centroid, color))
            pb = by_b.get(pr.b_id)
            for aid, _ in pr.rival_a:
                if pb is not None and aid in by_a:
                    cv2.drawContours(a, [by_a[aid].contour], -1, color, th, cv2.LINE_AA)
                    lines.append((by_a[aid].centroid, pb.centroid, color))

    # 無法對應（灰）
    if sel is None:
        for u in match.unmatched:
            canvas, lookup = (a, by_a) if u.side == "A" else (b, by_b)
            part = lookup.get(u.part_id)
            if part is not None:
                cv2.drawContours(canvas, [part.contour], -1, GRAY, th + 1, cv2.LINE_AA)
                _put_label(canvas, f"#{u.part_id}x", part.centroid, GRAY, scale)

    # ROI 警告標記（畫在零件外框右上角）
    for wn in match.roi_warnings:
        canvas, lookup = (a, by_a) if wn.side == "A" else (b, by_b)
        part = lookup.get(wn.part_id)
        if part is None or (sel is not None and part.id not in ({sel.a_id} if wn.side == "A" else {sel.b_id})):
            continue
        x, y, w, _ = part.bbox
        _put_label(canvas, TAG_TEXT[wn.kind], (x + w, max(y, 20)), TAG_COLORS[wn.kind], scale * 0.8)

    h = max(a.shape[0], b.shape[0])
    out = np.hstack([_pad_h(a, h), np.full((h, GAP, 3), 255, np.uint8), _pad_h(b, h)])
    off = a.shape[1] + GAP
    for (x0, y0), (x1, y1), color in lines:              # 跨越兩張圖的候選連線
        cv2.line(out, (int(x0), int(y0)), (int(x1) + off, int(y1)), color, max(2, th - 1), cv2.LINE_AA)
    if legend:
        _draw_legend(out, th)
    return out


def draw_roi_box(image: np.ndarray, roi: Roi | None) -> np.ndarray:
    """預覽用：只在影像上畫 ROI 框（沒有 ROI 時回傳原圖副本），作為檢視圖片時的基準。"""
    out = image.copy()
    if roi is not None:
        _dashed_rect(out, roi, ROI_COLOR, _thickness(out))
    return out


def _dashed_rect(img: np.ndarray, roi: Roi, color, th: int, dash: int = 18) -> None:
    x, y, w, h = roi
    pts = [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)), ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]
    for (x0, y0), (x1, y1) in pts:
        length = int(max(abs(x1 - x0), abs(y1 - y0)))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            p0 = (int(x0 + (x1 - x0) * s / max(length, 1)), int(y0 + (y1 - y0) * s / max(length, 1)))
            p1 = (int(x0 + (x1 - x0) * e / max(length, 1)), int(y0 + (y1 - y0) * e / max(length, 1)))
            cv2.line(img, p0, p1, color, th, cv2.LINE_AA)


def _draw_legend(img: np.ndarray, th: int) -> None:
    """匯出用的英文圖例（中文說明在報告中）。"""
    items = [("P3  high confidence", LEVEL_COLORS[LEVEL_GREEN]), ("P3? uncertain", LEVEL_COLORS[LEVEL_YELLOW]),
             ("P3! multiple candidates", LEVEL_COLORS[LEVEL_RED]), ("#8x unmatched", GRAY),
             ("ROI! partly outside ROI", TAG_COLORS[ROI_PARTIAL]), ("LOST possibly lost", TAG_COLORS[ROI_LOST])]
    s = th * 0.45
    lh = int(28 * s / 0.9) + 6
    w = int(330 * s / 0.9)
    cv2.rectangle(img, (8, 8), (8 + w, 8 + lh * len(items) + 8), (30, 30, 30), -1)
    for k, (text, color) in enumerate(items):
        y = 8 + lh * (k + 1)
        cv2.rectangle(img, (16, y - lh + 10), (16 + lh - 12, y - 2), color, -1)
        cv2.putText(img, text, (16 + lh, y - 4), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def _pad_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
