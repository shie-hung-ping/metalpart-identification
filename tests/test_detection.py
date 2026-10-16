"""辨識結果測試：合成影像（已知數量）與實拍樣本圖（目視確認 8 片）。"""
from pathlib import Path

import pytest

from conftest import synthetic_pipeline
from metalpartidbetification.fileio import read_image
from metalpartidbetification.algorithms.light import build_default_pipeline

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_01.webp"


def test_separate_parts_counted(scene_separate):
    img, n = scene_separate
    assert synthetic_pipeline().run(img).count == n


def test_touching_parts_are_split(scene_touching):
    img, n = scene_touching
    assert synthetic_pipeline().run(img).count == n


def test_blank_image_has_no_parts():
    import numpy as np
    img = np.full((300, 400, 3), 160, np.uint8)
    assert synthetic_pipeline().run(img).count == 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少樣本圖 samples/sample_01.webp")
def test_sample_image_counts_eight_parts():
    result = build_default_pipeline().run(read_image(SAMPLE))
    assert result.count == 8
    assert not any(p.suspicious for p in result.parts)


@pytest.mark.skipif(not SAMPLE.exists(), reason="缺少樣本圖 samples/sample_01.webp")
def test_sample_count_independent_of_roi_size():
    img = read_image(SAMPLE)
    pipe = build_default_pipeline()
    for roi in [(300, 50, 1300, 1000), (330, 60, 1150, 900)]:
        assert pipe.run(img, roi).count == 8
