"""② 位置配對：輪廓重疊度表 + 匈牙利演算法（一對一、整體最佳）。"""
from __future__ import annotations

import cv2
import numpy as np


def iou_matrix(contours_a: list[np.ndarray], contours_b: list[np.ndarray]) -> np.ndarray:
    """計算每一對輪廓的重疊度（IoU）；外框不相交的直接為 0，只在兩者外框聯集範圍內畫圖計算。"""
    na, nb = len(contours_a), len(contours_b)
    out = np.zeros((na, nb))
    boxes_a = [cv2.boundingRect(c) for c in contours_a]
    boxes_b = [cv2.boundingRect(c) for c in contours_b]
    for i, (ca, ba) in enumerate(zip(contours_a, boxes_a)):
        for j, (cb, bb) in enumerate(zip(contours_b, boxes_b)):
            x0, y0 = min(ba[0], bb[0]), min(ba[1], bb[1])
            x1, y1 = max(ba[0] + ba[2], bb[0] + bb[2]), max(ba[1] + ba[3], bb[1] + bb[3])
            if ba[0] > bb[0] + bb[2] or bb[0] > ba[0] + ba[2] or ba[1] > bb[1] + bb[3] or bb[1] > ba[1] + ba[3]:
                continue
            ma = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            mb = np.zeros_like(ma)
            cv2.drawContours(ma, [ca - (x0, y0)], -1, 1, -1)
            cv2.drawContours(mb, [cb - (x0, y0)], -1, 1, -1)
            union = float((ma | mb).sum())
            out[i, j] = float((ma & mb).sum()) / union if union else 0.0
    return out


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """匈牙利演算法（最小化總成本）。支援非方陣；回傳 (列, 行) 配對。

    零件數量少（數十片以內），以純 Python 實作已足夠快，不需另外安裝 SciPy。
    """
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:                          # 演算法要求 列數 ≤ 行數
        cost = cost.T
    n, m = cost.shape
    if n == 0:
        return []
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)      # p[j] = 第 j 行配給哪一列（1 起算，0 = 未配）
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:                          # 沿增廣路徑更新配對
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(b, a) for a, b in pairs] if transposed else pairs
