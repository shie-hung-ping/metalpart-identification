"""配對演算法用的幾何工具：輪廓鏡像換算、輪廓轉遮罩、輪廓重新取樣。"""
from __future__ import annotations

import cv2
import numpy as np


def mirror_contour(contour: np.ndarray, axis: str, size: tuple[int, int], dx: int = 0, dy: int = 0) -> np.ndarray:
    """把 A 圖上的輪廓換算到 B 圖：依 axis 鏡像整張圖的座標，再平移 (dx, dy)。size = (寬, 高)。"""
    w, h = size
    pts = contour.reshape(-1, 2).astype(np.float64)
    if axis in ("h", "hv"):
        pts[:, 0] = (w - 1) - pts[:, 0]
    if axis in ("v", "hv"):
        pts[:, 1] = (h - 1) - pts[:, 1]
    pts += (dx, dy)
    return pts.round().astype(np.int32).reshape(-1, 1, 2)


def mirror_local(points: np.ndarray, axis: str) -> np.ndarray:
    """以原點（中心點）為準鏡像一組相對座標（N×2）。"""
    p = points.astype(np.float64).copy()
    if axis in ("h", "hv"):
        p[:, 0] = -p[:, 0]
    if axis in ("v", "hv"):
        p[:, 1] = -p[:, 1]
    return p


def union_mask(contours: list[np.ndarray], size: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """把多個輪廓畫成一張填滿的遮罩（可縮小以加速）。"""
    w, h = size
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    m = np.zeros((sh, sw), np.uint8)
    if contours:
        cv2.drawContours(m, [(c.astype(np.float64) * scale).round().astype(np.int32) for c in contours], -1, 1, -1)
    return m


def centroid(contour: np.ndarray) -> np.ndarray:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return contour.reshape(-1, 2).mean(0)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def resample(contour: np.ndarray, step: float = 2.0) -> np.ndarray:
    """沿輪廓周長每 step 像素取一點（CHAIN_APPROX_SIMPLE 的輪廓直線段只有端點，需補點）。"""
    pts = contour.reshape(-1, 2).astype(np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.sqrt((seg ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return pts
    t = np.arange(0, total, step)
    idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1)
    frac = (t - cum[idx]) / np.maximum(seg_len[idx], 1e-9)
    return closed[idx] + seg[idx] * frac[:, None]
