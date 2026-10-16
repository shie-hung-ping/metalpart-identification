"""測試共用工具：產生已知零件數量的合成影像（僅用於驗證程式邏輯，不用於訓練）。

合成零件尺寸接近實拍樣本（腹寬約 120px），因此可直接用預設參數測試。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from metalpartidbetification.core import Pipeline
from metalpartidbetification.algorithms.light import build_default_pipeline


def t_polygon(cx: int, cy: int, angle: float) -> np.ndarray:
    """以 (cx, cy) 為中心、旋轉 angle 度的 T 字形多邊形（翼寬 360、腹寬 120、全高 400）。"""
    pts = np.array([[-180, -200], [180, -200], [180, -100], [60, -100], [60, 200],
                    [-60, 200], [-60, -100], [-180, -100]], np.float64)
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ rot.T + [cx, cy]).astype(np.int32)


def make_scene(polys: list[np.ndarray], size=(1300, 1900), seed=0) -> np.ndarray:
    """淺色平滑背景 + 有紋理的深色零件，零件外緣加一圈暗色側壁（模擬實拍的厚度/縫隙）。"""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w), 165, np.float32) + rng.normal(0, 1.5, (h, w))
    for poly in polys:
        cv2.polylines(img, [poly], True, 50, 10)          # 暗色側壁
    tex = 120 + rng.normal(0, 12, (h, w))                  # 表面紋理
    for poly in polys:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        mask = cv2.erode(mask, np.ones((11, 11), np.uint8))  # 內縮，保留側壁
        img[mask > 0] = tex[mask > 0]
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def synthetic_pipeline() -> Pipeline:
    return build_default_pipeline()


@pytest.fixture
def scene_separate():
    """3 片互不接觸的零件。"""
    polys = [t_polygon(350, 350, 0), t_polygon(1000, 420, 30), t_polygon(1500, 900, -60)]
    return make_scene(polys), len(polys)


@pytest.fixture
def scene_touching():
    """4 片零件，其中 2 片翼部側壁緊貼（模擬相黏）。"""
    polys = [t_polygon(350, 330, 0), t_polygon(350, 735, 180),   # 這兩片翼部相貼
             t_polygon(1050, 450, 90), t_polygon(1500, 950, 45)]
    return make_scene(polys), len(polys)
