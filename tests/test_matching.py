"""鏡像（翻面）配對測試：匈牙利演算法、合成圖與實拍樣本的配對正確率。

測試程式知道鏡像是怎麼產生的（用來產生標準答案），但配對演算法本身拿不到這些資訊。
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from conftest import make_scene, t_polygon
from metalpartidbetification.algorithms.light import build_default_pipeline
from metalpartidbetification.app import DetectionController
from metalpartidbetification.core import FLIP_AXES
from metalpartidbetification.imaging import FlipSimParams, flip_point, simulate_flip
from metalpartidbetification.matching import contour_fit, hungarian, match_flipped

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for shape in [(4, 4), (3, 5), (5, 3)]:
        cost = rng.random(shape)
        got = sum(cost[i, j] for i, j in hungarian(cost))
        k = min(shape)
        if shape[0] <= shape[1]:
            best = min(sum(cost[i, p[i]] for i in range(k)) for p in permutations(range(shape[1]), k))
        else:
            best = min(sum(cost[p[j], j] for j in range(k)) for p in permutations(range(shape[0]), k))
        assert got == pytest.approx(best)
    assert hungarian(np.zeros((0, 3))) == []


def test_contour_fit_prefers_mirror_for_asymmetric_shape():
    """不對稱的 L 形：鏡像後應比不鏡像更貼合。"""
    l_shape = np.array([[0, 0], [200, 0], [200, 40], [40, 40], [40, 150], [0, 150]], np.int32).reshape(-1, 1, 2)
    mirrored = l_shape.copy()
    mirrored[:, 0, 0] = 500 - mirrored[:, 0, 0]
    assert contour_fit(l_shape, mirrored, "h") > 0.9
    assert contour_fit(l_shape, mirrored, "h") > contour_fit(l_shape, mirrored, None) + 0.2


@pytest.mark.parametrize("axis", FLIP_AXES)
def test_synthetic_scene_pairs_are_correct(axis):
    polys = [t_polygon(350, 330, 0), t_polygon(350, 735, 180), t_polygon(1050, 450, 90), t_polygon(1500, 950, 45)]
    img = make_scene(polys)
    params = FlipSimParams(axis)
    pipe = build_default_pipeline()
    ra, rb = pipe.run(img), pipe.run(simulate_flip(img, params))
    size = (img.shape[1], img.shape[0])
    match = match_flipped(ra.parts, rb.parts, size)
    assert match.relation.axis == axis
    assert len(match.pairs) == 4 and not match.unmatched_a and not match.unmatched_b
    by_b = {p.id: p for p in rb.parts}
    by_a = {p.id: p for p in ra.parts}
    for pr in match.pairs:                       # 標準答案：A 中心點換算到 B 後應落在配對到的那一片附近
        x, y = flip_point(*by_a[pr.a_id].centroid, size, params)
        bx, by = by_b[pr.b_id].centroid
        assert np.hypot(bx - x, by - y) < 15


@pytest.mark.parametrize("algo, name, expected", [
    ("light", "sample_01.webp", 8),
    ("dark", "dark_sample_01.webp", 1),       # 最基本條件：單片鏡像後仍配出 1 組
    ("dark", "dark_sample_02.webp", 6),
])
@pytest.mark.parametrize("axis", FLIP_AXES)
def test_samples_mirror_pairing(algo, name, expected, axis):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"缺少樣本圖 {name}")
    ctrl = DetectionController(algo)
    ctrl.load_image(path)
    ctrl.run()
    run = ctrl.run_mirror(axis)
    assert run.result.count == expected
    assert len(run.match.pairs) == expected
    assert run.evaluation.axis_correct
    assert run.evaluation.green_wrong == 0                     # 高信心但配錯：必須為 0
    assert not run.match.pairs_at("red")                       # 無位移時不應出現多重候選
    assert all(run.evaluation.pair_truth.values())             # 全部配對（含黃色）實際上都正確
    assert [k for k, _ in run.timings] == ["原圖辨識", "鏡像產生", "鏡像辨識", "配對"]
    assert run.total == pytest.approx(sum(t for _, t in run.timings))


def _part(pid, pts):
    from metalpartidbetification.core import DetectedPart
    c = np.array(pts, np.int32).reshape(-1, 1, 2)
    import cv2
    area = float(cv2.contourArea(c))
    m = cv2.moments(c)
    return DetectedPart(pid, c, cv2.boundingRect(c), area, (m["m10"] / m["m00"], m["m01"] / m["m00"]), 1.0)


def test_merged_region_is_red_with_area_reason():
    """鏡像那邊把兩片辨識成一大塊：這塊同時可能屬於原圖兩片 → 紅色，並附上面積不一致的原因。"""
    a = [_part(1, [[100, 100], [200, 100], [200, 300], [100, 300]]),
         _part(2, [[210, 100], [310, 100], [310, 300], [210, 300]])]
    # 影像寬 1000，左右鏡像後兩片位於 x = 689~899，但被辨識成一整塊
    b = [_part(1, [[689, 100], [899, 100], [899, 300], [689, 300]])]
    m = match_flipped(a, b, (1000, 400))
    assert len(m.pairs) == 1
    pr = m.pairs[0]
    assert pr.level == "red" and pr.rival_a and "大小不一致" in pr.note
    assert m.normal_pairs == [] and m.suspicious_pairs == [pr]


def test_area_mismatch_alone_is_yellow():
    a = [_part(1, [[100, 100], [200, 100], [200, 300], [100, 300]])]
    b = [_part(1, [[740, 100], [899, 100], [899, 340], [740, 340]])]      # 同位置但大很多
    m = match_flipped(a, b, (1000, 400))
    assert m.pairs[0].level == "yellow" and "大小不一致" in m.pairs[0].note


def test_similar_area_pair_is_green():
    a = [_part(1, [[100, 100], [200, 100], [200, 300], [100, 300]])]
    b = [_part(1, [[799, 100], [899, 100], [899, 300], [799, 300]])]
    m = match_flipped(a, b, (1000, 400))
    assert len(m.normal_pairs) == 1 and m.pairs[0].level == "green"


def test_no_parts_gives_no_relation():
    """任一邊沒有零件時，不可硬猜鏡像方向。"""
    from metalpartidbetification.core import NO_RELATION
    m = match_flipped([], [], (100, 100))
    assert m.relation.axis == NO_RELATION and m.pairs == []


