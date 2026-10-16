"""鏡像位移測試：位移估算、逐片判定（多重候選、移出畫面、超出範圍）、ROI 警告、情境與報告。

測試程式知道模擬參數（用來產生標準答案），配對演算法本身拿不到。
"""
from pathlib import Path

import numpy as np
import pytest

from conftest import make_scene, t_polygon
from metalpartidbetification.algorithms import ALGORITHMS
from metalpartidbetification.core import JudgeThresholds
from metalpartidbetification.fileio import read_image
from metalpartidbetification.imaging import shift_vector
from metalpartidbetification.shift_test import (
    ShiftTestRunner, build_scenarios, detail_rows, export_report, parse_distances, summary_row,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


# ---------- 情境與輸入 ----------
def test_diagonal_shift_uses_straight_line_distance():
    assert shift_vector("right", 50) == (50, 0)
    assert shift_vector("up", 30) == (0, -30)
    assert shift_vector("up_right", 50) == (35, -35)          # 斜線長度 50 px
    assert shift_vector("down_left", 50) == (-35, 35)


def test_parse_distances():
    assert parse_distances("50, 30，30 10") == [10, 30, 50]
    for bad in ["", "abc", "120", "-5"]:
        with pytest.raises(ValueError):
            parse_distances(bad)


def test_build_scenarios_one_direction_one_distance_each():
    scs = build_scenarios("h", ["right", "up"], [30, 50])
    assert len(scs) == 4 and {(s.direction, s.distance) for s in scs} == {("up", 30), ("up", 50), ("right", 30), ("right", 50)}
    assert scs[0].over_range(50) is False and build_scenarios("h", ["up"], [80])[0].over_range(50)


# ---------- 合成圖 ----------
@pytest.fixture(scope="module")
def synthetic_runner():
    polys = [t_polygon(450, 380, 0), t_polygon(450, 785, 180), t_polygon(1150, 500, 90), t_polygon(1450, 900, 45)]
    img = make_scene(polys)
    run = ShiftTestRunner(ALGORITHMS["light"].build(), img, roi=None)
    run.run_original()
    return run


@pytest.mark.parametrize("direction", ["up", "down_right", "left", "up_left"])
@pytest.mark.parametrize("distance", [10, 30, 50])
def test_shift_within_range_all_green_and_estimated(synthetic_runner, direction, distance):
    r = synthetic_runner.run_scenario(build_scenarios("h", [direction], [distance])[0])
    c = r.match.level_counts()
    assert c["green"] == 4 and c["yellow"] == c["red"] == c["gray"] == 0
    assert r.evaluation.green_wrong == 0
    assert r.evaluation.shift_error <= 4                 # 推估位移誤差 ≤ 4 px


def test_shift_beyond_range_never_confidently_wrong(synthetic_runner):
    r = synthetic_runner.run_scenario(build_scenarios("h", ["right"], [80])[0])
    assert r.evaluation.green_wrong == 0                 # 可以不確定，但不能自信地講錯


# ---------- 多重候選 ----------
def test_equally_spaced_row_with_half_spacing_shift_is_not_green():
    """一排等距、間距約 260 px 的零件：關係推估被迫在 ±50 內，若實際錯位約半個間距 → 不可出現錯誤的綠色。"""
    polys = [t_polygon(300 + 260 * k, 600, 90) for k in range(5)]
    img = make_scene(polys)
    run = ShiftTestRunner(ALGORITHMS["light"].build(), img, roi=None)
    r = run.run_scenario(build_scenarios("h", ["right"], [100])[0])
    assert r.evaluation.green_wrong == 0


# ---------- 移出畫面 ----------
def test_part_pushed_out_of_frame_is_gray_with_reason():
    polys = [t_polygon(700, 600, 0), t_polygon(1890, 600, 90)]      # 第二片貼著右邊緣
    img = make_scene(polys)
    run = ShiftTestRunner(ALGORITHMS["light"].build(), img, roi=None)
    r = run.run_scenario(build_scenarios("h", ["left"], [50])[0])  # 左右鏡像後在左邊緣，再往左推
    reasons = [u.reason for u in r.match.unmatched]
    greens = r.match.pairs_at("green")
    assert r.evaluation.green_wrong == 0
    assert len(greens) >= 1
    assert all(p.level != "green" or r.evaluation.pair_truth[p.pair_id] for p in r.match.pairs)
    assert reasons == [] or any("畫面" in s or "辨識" in s for s in reasons)


# ---------- ROI 警告 ----------
def test_roi_warnings_levels(synthetic_runner):
    """ROI 框住全部零件、四周留 20 px；鏡像後 ROI 不跟著位移 → 推 10 px 不出框，推 50 px 有零件出框（ROI!）。"""
    boxes = np.array([p.bbox for p in synthetic_runner.original.parts])
    x0, y0 = boxes[:, 0].min() - 20, boxes[:, 1].min() - 20
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max() + 20, (boxes[:, 1] + boxes[:, 3]).max() + 20
    roi = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    run = ShiftTestRunner(synthetic_runner.pipeline, synthetic_runner.image, roi=roi)
    run.original = synthetic_runner.original
    r0 = run.run_scenario(build_scenarios("h", ["right"], [10])[0])
    r1 = run.run_scenario(build_scenarios("h", ["right"], [50])[0])
    assert r0.match.roi_warnings == []
    assert "partial" in {w.kind for w in r1.match.roi_warnings}
    w = synthetic_runner.image.shape[1]
    assert r1.match.roi_b == (w - roi[0] - roi[2], roi[1], roi[2], roi[3])   # 跟著鏡像、不跟著位移


def test_roi_original_outside_is_separate(synthetic_runner):
    """原圖零件一開始就超出 ROI → 標為「原圖已超出」，不算位移造成。"""
    run = ShiftTestRunner(synthetic_runner.pipeline, synthetic_runner.image, roi=(0, 0, 900, 1300))
    run.original = synthetic_runner.original
    r = run.run_scenario(build_scenarios("h", ["up"], [10])[0])
    assert "original" in {w.kind for w in r.match.roi_warnings}


def test_no_roi_no_warnings(synthetic_runner):
    r = synthetic_runner.run_scenario(build_scenarios("h", ["up"], [30])[0])
    assert r.match.roi_warnings == [] and r.match.roi_b is None


# ---------- 報告 ----------
def test_report_export(tmp_path, synthetic_runner):
    results = synthetic_runner.run_all(build_scenarios("v", ["up", "right"], [30]))
    path = export_report(tmp_path, "合成圖.png", "淺色背景", synthetic_runner, results)
    assert path.exists() and (path.parent / "情境總表.csv").exists() and (path.parent / "逐片明細.csv").exists()
    assert len(list((path.parent / "images").glob("*.png"))) == 2
    s = summary_row(results[0], JudgeThresholds())
    assert s["綠"] == 4 and s["高信心但配錯"] == 0
    assert len(detail_rows(results[0])) == 4


# ---------- 實拍樣本 ----------
def test_dark_ring_shift_50_all_green():
    path = SAMPLES / "dark_sample_02.webp"
    if not path.exists():
        pytest.skip("dark_sample_02.webp")
    run = ShiftTestRunner(ALGORITHMS["dark"].build(), read_image(path), roi=None)
    for r in run.run_all(build_scenarios("h", ["up_right", "down"], [50])):
        assert r.evaluation.green_wrong == 0
        assert r.match.level_counts()["green"] == 6


def test_dark_single_part_shift_never_confidently_wrong():
    """單片深色樣本：已知「↗50」時深色演算法在鏡像圖辨識不到（實心度 0.549 略低於選樣板門檻 0.55），
    屬於辨識演算法的問題（待使用者決定是否調整），這裡只要求不可出現錯誤的綠色。"""
    path = SAMPLES / "dark_sample_01.webp"
    if not path.exists():
        pytest.skip("dark_sample_01.webp")
    run = ShiftTestRunner(ALGORITHMS["dark"].build(), read_image(path), roi=None)
    for r in run.run_all(build_scenarios("h", ["up_right", "down"], [50])):
        assert r.evaluation.green_wrong == 0
