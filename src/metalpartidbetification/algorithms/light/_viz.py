"""步驟內部共用：把標記圖轉成彩色圖，供中間結果檢視。"""
from __future__ import annotations

import numpy as np


def colorize_labels(labels: np.ndarray, boundary_value: int = -1) -> np.ndarray:
    """每個標記給一個固定的隨機顏色；0 = 黑，分界線 = 白。"""
    n = int(labels.max()) + 1 if labels.size else 1
    rng = np.random.default_rng(12345)
    palette = rng.integers(60, 255, size=(max(n, 1), 3), dtype=np.uint8)
    palette[0] = 0
    out = palette[np.clip(labels, 0, None)]
    out[labels == boundary_value] = 255
    return out
