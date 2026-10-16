"""模擬翻面拍攝：以影像鏡像（可加平移）模擬「零件翻面後再拍一次」。

注意：模擬參數（鏡像方向、位移）只交給 app 層做「評估用」，
配對演算法（matching 套件）不會拿到，也不可引用本模組。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core import FLIP_AXES, Roi

_CV_FLIP_CODE = {"h": 1, "v": 0, "hv": -1}   # OpenCV flip 代碼：1 = 左右、0 = 上下、-1 = 兩者

# 平移方位：代碼 → (箭頭, 中文名稱, 方向向量)。影像座標 y 軸向下，所以「往上」是 dy < 0
DIRECTIONS: dict[str, tuple[str, str, tuple[int, int]]] = {
    "up_left": ("↖", "左上", (-1, -1)), "up": ("↑", "上", (0, -1)), "up_right": ("↗", "右上", (1, -1)),
    "left": ("←", "左", (-1, 0)), "right": ("→", "右", (1, 0)),
    "down_left": ("↙", "左下", (-1, 1)), "down": ("↓", "下", (0, 1)), "down_right": ("↘", "右下", (1, 1)),
}


def shift_vector(direction: str, distance: float) -> tuple[int, int]:
    """方位 + 幅度 → (dx, dy)。幅度為實際移動的直線距離：斜向 50 px ≈ 兩軸各 35 px。"""
    ux, uy = DIRECTIONS[direction][2]
    norm = (ux * ux + uy * uy) ** 0.5
    return int(round(ux * distance / norm)), int(round(uy * distance / norm))


@dataclass(frozen=True)
class FlipSimParams:
    axis: str           # "h" / "v" / "hv"
    dx: int = 0         # 鏡像後再平移（像素）；本階段固定 0，之後開放到 50 px 內
    dy: int = 0


def simulate_flip(image: np.ndarray, params: FlipSimParams) -> np.ndarray:
    """產生模擬翻面後的影像：先鏡像，再平移；平移空出的邊緣以背景色補滿（避免黑邊被當成邊緣）。"""
    if params.axis not in FLIP_AXES:
        raise ValueError(f"未知的鏡像方向：{params.axis}")
    out = cv2.flip(image, _CV_FLIP_CODE[params.axis])
    if params.dx or params.dy:
        h, w = out.shape[:2]
        m = np.float32([[1, 0, params.dx], [0, 1, params.dy]])
        border = _border_color(image)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return out


def flip_point(x: float, y: float, size: tuple[int, int], params: FlipSimParams) -> tuple[float, float]:
    """把 A 圖上的點換算到 B 圖（評估用：產生標準答案）。size = (寬, 高)。"""
    w, h = size
    if params.axis in ("h", "hv"):
        x = (w - 1) - x
    if params.axis in ("v", "hv"):
        y = (h - 1) - y
    return x + params.dx, y + params.dy


def flip_roi(roi: Roi | None, size: tuple[int, int], params: FlipSimParams) -> Roi | None:
    """原圖的 ROI 換算到鏡像圖上（同樣鏡像 + 平移）。"""
    if roi is None:
        return None
    x, y, rw, rh = roi
    x0, y0 = flip_point(x, y, size, params)
    x1, y1 = flip_point(x + rw - 1, y + rh - 1, size, params)
    return int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)) + 1, int(abs(y1 - y0)) + 1


def _border_color(image: np.ndarray) -> tuple:
    """以影像最外圈像素的中位數估計背景色。"""
    edge = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    med = np.median(edge.reshape(-1, image.shape[2] if image.ndim == 3 else 1), axis=0)
    return tuple(float(v) for v in np.atleast_1d(med))
