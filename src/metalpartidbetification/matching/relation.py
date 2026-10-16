"""① 推估翻面關係：已知 B 是 A 翻面拍攝（由圖片編號得知），但不知道鏡像方向與位移量。

只使用兩邊的「零件遮罩」（不使用灰階影像）：實際翻面拍到的是零件背面，表面紋理不同，
只有外形輪廓兩面共有。已知是同一張圖 → 所有零件共用同一個「鏡像＋平移」關係。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import FLIP_AXES, NO_RELATION, DetectedPart, FlipRelation
from .geometry import union_mask

_FLIP = {"h": lambda m: m[:, ::-1], "v": lambda m: m[::-1, :], "hv": lambda m: m[::-1, ::-1]}
COARSE_SCALE = 0.25      # 粗搜尋時縮小比例
REFINE_RADIUS = 4        # 原尺寸微調範圍 ±4 px


def estimate_relation(parts_a: list[DetectedPart], parts_b: list[DetectedPart],
                      size: tuple[int, int], max_shift: int = 0) -> FlipRelation:
    """三種鏡像方向各自在 ±max_shift 內找最佳位移，取「翻轉＋位移後與 B 遮罩重疊度（IoU）」最高者。

    max_shift = 0 時只比較三種方向（不估算位移）。
    """
    if not parts_a or not parts_b:              # 任一邊沒有零件 → 無從比較
        return FlipRelation(axis=NO_RELATION, score=0.0)
    w, h = size
    best: FlipRelation | None = None
    scores: dict[str, float] = {}

    # 粗搜尋：縮小影像，三種方向 × 位移
    ma = union_mask([p.contour for p in parts_a], size, COARSE_SCALE).astype(np.float32)
    mb = union_mask([p.contour for p in parts_b], size, COARSE_SCALE).astype(np.float32)
    r = int(np.ceil(max_shift * COARSE_SCALE))
    for axis in FLIP_AXES:
        fa = np.ascontiguousarray(_FLIP[axis](ma))
        dx_s, dy_s, iou = _best_shift(fa, mb, r)
        scores[axis] = iou
        if best is None or iou > best.score:
            best = FlipRelation(axis, int(round(dx_s / COARSE_SCALE)), int(round(dy_s / COARSE_SCALE)), iou)

    # 微調：在原尺寸、以粗搜尋結果為中心 ±REFINE_RADIUS 內再找一次（只在零件所在範圍內計算，加速）
    if max_shift > 0:
        fa_full = np.ascontiguousarray(_FLIP[best.axis](union_mask([p.contour for p in parts_a], size)))
        fa_full = _shift(fa_full, best.dx, best.dy).astype(np.float32)
        mb_full = union_mask([p.contour for p in parts_b], size).astype(np.float32)
        ddx, ddy, iou = _best_shift(fa_full, mb_full, REFINE_RADIUS, crop=True)
        dx = int(np.clip(best.dx + ddx, -max_shift, max_shift))
        dy = int(np.clip(best.dy + ddy, -max_shift, max_shift))
        best = FlipRelation(best.axis, dx, dy, iou)
    best.candidates = scores
    return best


def _best_shift(fa: np.ndarray, mb: np.ndarray, r: int, crop: bool = False) -> tuple[int, int, float]:
    """把 fa 在 ±r 內平移，找與 mb 重疊度（IoU）最高的位移 (dx, dy)。"""
    sa, sb = float(fa.sum()), float(mb.sum())
    if r <= 0:
        inter = float((fa * mb).sum())
        return 0, 0, inter / max(sa + sb - inter, 1.0)
    if crop:                                    # 只取兩個遮罩聯集的外框（加上 r）計算
        ys, xs = np.nonzero((fa + mb) > 0)
        y0, y1 = max(ys.min() - r, 0), min(ys.max() + r + 1, fa.shape[0])
        x0, x1 = max(xs.min() - r, 0), min(xs.max() + r + 1, fa.shape[1])
        fa, mb = fa[y0:y1, x0:x1], mb[y0:y1, x0:x1]
    padded = cv2.copyMakeBorder(mb, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    inter = cv2.matchTemplate(padded, fa, cv2.TM_CCORR)        # (2r+1)×(2r+1)：每種位移的交集面積
    iou = inter / np.maximum(sa + sb - inter, 1.0)
    _, mx, _, (lx, ly) = cv2.minMaxLoc(iou)
    return lx - r, ly - r, float(mx)


def _shift(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """整數平移遮罩（移出邊界的部分捨棄、空出的部分補 0）。"""
    h, w = m.shape
    out = np.zeros_like(m)
    xs0, xd0 = (0, dx) if dx >= 0 else (-dx, 0)
    ys0, yd0 = (0, dy) if dy >= 0 else (-dy, 0)
    cw, ch = w - abs(dx), h - abs(dy)
    if cw > 0 and ch > 0:
        out[yd0:yd0 + ch, xd0:xd0 + cw] = m[ys0:ys0 + ch, xs0:xs0 + cw]
    return out


def flip_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    return _FLIP[axis](mask)
