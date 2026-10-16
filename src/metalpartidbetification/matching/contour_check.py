"""③ 輪廓細節驗證（以中心點為基準）。

把 A 零件的輪廓以自身中心點為原點鏡像，疊到 B 零件的輪廓上（也以中心點對齊），
在小範圍內微調位置與角度，計算兩條輪廓的平均距離（倒角距離）→ 貼合度。
同時計算「不鏡像」的貼合度作為對照：鏡像後明顯更貼合，才代表輪廓細節支持「同一片翻面」。
只使用輪廓（外形），不使用表面紋理（實際翻面時兩面表面不同）。
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import centroid, mirror_local, resample

SHIFT_RANGE = 5          # 中心點微調範圍 ±5 px（吸收兩邊辨識輪廓的誤差）
ANGLE_RANGE = 3          # 角度微調範圍 ±3°
DIST_SCALE = 10.0        # 平均距離 10 px → 貼合度 0


def contour_fit(contour_a: np.ndarray, contour_b: np.ndarray, axis: str | None) -> float:
    """回傳 0~1 的貼合度；axis=None 表示不鏡像（對照組）。"""
    pa = resample(contour_a) - centroid(contour_a)
    pb = resample(contour_b) - centroid(contour_b)
    if axis is not None:
        pa = mirror_local(pa, axis)

    # 把 B 的輪廓畫在畫布上，算距離轉換：每個像素到 B 輪廓的距離
    r = int(np.ceil(max(np.abs(pa).max(), np.abs(pb).max()))) + SHIFT_RANGE + 4
    size = 2 * r + 1
    canvas = np.full((size, size), 255, np.uint8)
    cv2.polylines(canvas, [(pb + r).round().astype(np.int32).reshape(-1, 1, 2)], True, 0, 1)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)

    offsets = np.array([(dx, dy) for dy in range(-SHIFT_RANGE, SHIFT_RANGE + 1)
                        for dx in range(-SHIFT_RANGE, SHIFT_RANGE + 1)], np.float64)
    best = np.inf
    for deg in range(-ANGLE_RANGE, ANGLE_RANGE + 1):
        t = np.deg2rad(deg)
        rot = pa @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        pts = (rot[None, :, :] + offsets[:, None, :] + r).round().astype(np.int64)   # (位移數, 點數, 2)
        pts = np.clip(pts, 0, size - 1)
        mean_d = dist[pts[..., 1], pts[..., 0]].mean(axis=1)                          # 每種位移的平均距離
        best = min(best, float(mean_d.min()))
    return max(0.0, 1.0 - best / DIST_SCALE)
