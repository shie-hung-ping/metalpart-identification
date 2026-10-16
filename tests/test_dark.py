"""深色背景演算法測試：合成影像（已知數量）與實拍樣本圖。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import t_polygon
from metalpartidbetification.algorithms.dark import build_default_pipeline
from metalpartidbetification.fileio import read_image

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def make_dark_scene(polys, size=(1300, 1900), seed=0) -> np.ndarray:
    """深色背景 + 亮色有紋理的零件；零件之間畫一條細暗接縫（模擬緊靠）。"""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w), 18, np.float32) + rng.normal(0, 2, (h, w))
    tex = 170 + rng.normal(0, 12, (h, w))
    for poly in polys:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [poly], 1)
        img[m > 0] = tex[m > 0]
    for poly in polys:
        cv2.polylines(img, [poly], True, 60, 4)            # 細暗接縫（零件邊緣）
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_single_part_is_one():
    """最基本條件：單片零件只能數到 1。"""
    img = make_dark_scene([t_polygon(900, 600, 15)])
    assert build_default_pipeline().run(img).count == 1


def test_separate_parts_counted():
    polys = [t_polygon(350, 350, 0), t_polygon(1000, 420, 30), t_polygon(1500, 900, -60)]
    assert build_default_pipeline().run(make_dark_scene(polys)).count == 3


def test_touching_parts_are_split():
    polys = [t_polygon(350, 330, 0), t_polygon(350, 735, 180),   # 腹部相接，中間只有細接縫
             t_polygon(1050, 450, 90), t_polygon(1500, 950, 45)]
    assert build_default_pipeline().run(make_dark_scene(polys)).count == 4


def test_blank_dark_image_has_no_parts():
    img = np.full((300, 400, 3), 18, np.uint8)
    assert build_default_pipeline().run(img).count == 0


@pytest.mark.parametrize("name, expected", [
    ("dark_sample_01.webp", 1),     # 單片
    ("dark_sample_02.webp", 6),     # 半圓環 6 片，翼部互相卡合
])
def test_dark_samples(name, expected):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"缺少樣本圖 {name}")
    result = build_default_pipeline().run(read_image(path))
    assert result.count == expected
    assert result.algorithm == "深色背景"
