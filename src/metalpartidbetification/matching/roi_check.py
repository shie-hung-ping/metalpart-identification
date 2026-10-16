"""ROI 警告：檢查零件是否超出使用者框選的 ROI（零件應待著的範圍，例如托盤或治具）。

- B 圖上的 ROI ＝ 原圖 ROI「依推估的鏡像方向」鏡像後的位置，不加位移（ROI 代表固定的托盤範圍）
- ROI!（partial）：輪廓有任何一點在 ROI 外 → 位移可能過大
- LOST（lost）  ：整片在 ROI 外；或原圖零件沒配對到且推估位置在 ROI 外／畫面外 → 可能遺失
- 原圖（original）：原圖零件一開始就超出 ROI，和位移造成的分開標示
"""
from __future__ import annotations

import numpy as np

from ..core import (
    NO_RELATION, ROI_LOST, ROI_ORIGINAL, ROI_PARTIAL, DetectedPart, FlipRelation, MatchResult, Roi, RoiWarning,
)
from .geometry import centroid, mirror_contour


def mirror_roi(roi: Roi, axis: str, size: tuple[int, int]) -> Roi:
    """ROI 依鏡像方向翻轉（不位移）。"""
    w, h = size
    x, y, rw, rh = roi
    if axis in ("h", "hv"):
        x = w - (x + rw)
    if axis in ("v", "hv"):
        y = h - (y + rh)
    return int(x), int(y), int(rw), int(rh)


def _outside(points: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, w, h = roi
    p = points.reshape(-1, 2)
    return (p[:, 0] < x) | (p[:, 0] > x + w - 1) | (p[:, 1] < y) | (p[:, 1] > y + h - 1)


def check_roi(parts_a: list[DetectedPart], parts_b: list[DetectedPart], match: MatchResult,
              roi: Roi | None, size: tuple[int, int]) -> None:
    """就地填入 match.roi_warnings 與 match.roi_b。沒有 ROI 或無法推估關係時不檢查。"""
    rel: FlipRelation = match.relation
    if roi is None or rel.axis == NO_RELATION:
        return
    roi_b = mirror_roi(roi, rel.axis, size)
    match.roi_b = roi_b
    by_a = {p.id: p for p in parts_a}
    warns: list[RoiWarning] = []

    orig_out: set[int] = set()
    for p in parts_a:                                   # 原圖一開始就超出 ROI
        out = _outside(p.contour, roi)
        if out.any():
            orig_out.add(p.id)
            warns.append(RoiWarning("A", p.id, ROI_ORIGINAL, f"原圖中已有 {out.mean():.0%} 輪廓在 ROI 外（非位移造成）"))

    a_of_b = {pr.b_id: pr.a_id for pr in match.pairs}
    for p in parts_b:                                   # 鏡像（位移後）圖
        out = _outside(p.contour, roi_b)
        if not out.any():
            continue
        if a_of_b.get(p.id) in orig_out:
            continue                                    # 原圖就已超出，已另外標示
        if out.all():
            warns.append(RoiWarning("B", p.id, ROI_LOST, "可能遺失：整片在 ROI 外"))
        else:
            warns.append(RoiWarning("B", p.id, ROI_PARTIAL, f"位移可能過大：輪廓有 {out.mean():.0%} 在 ROI 外"))

    w, h = size
    for aid in match.unmatched_a:                       # 原圖有、鏡像沒配到 → 以推估關係換算位置
        p = by_a[aid]
        moved = mirror_contour(p.contour, rel.axis, size, rel.dx, rel.dy)
        cx, cy = centroid(moved)
        off_frame = not (0 <= cx < w and 0 <= cy < h)
        if off_frame or _outside(np.array([[cx, cy]]), roi_b).all():
            where = "畫面外" if off_frame else "ROI 外"
            warns.append(RoiWarning("A", aid, ROI_LOST, f"可能遺失：鏡像圖中找不到，推估位置在{where}"))
    match.roi_warnings = warns
