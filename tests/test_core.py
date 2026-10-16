"""核心層測試：步驟參數、註冊表、Pipeline 相依檢查與 ROI 座標還原。"""
import numpy as np
import pytest

from metalpartidbetification.core import Pipeline, PipelineError, create_step
from metalpartidbetification.algorithms.light import build_default_pipeline


def test_unknown_param_rejected():
    with pytest.raises(KeyError):
        create_step("binarize", not_a_param=1)


def test_unknown_step_rejected():
    with pytest.raises(KeyError):
        create_step("no_such_step")


def test_disabling_required_step_is_reported():
    pipe = build_default_pipeline()
    next(s for s in pipe.steps if s.key == "local_contrast").enabled = False
    with pytest.raises(PipelineError, match="缺少輸入資料"):
        pipe.validate()


def test_pipeline_without_parts_is_invalid():
    with pytest.raises(PipelineError):
        Pipeline([create_step("grayscale")]).validate()


def test_snapshots_follow_step_order(scene_separate):
    img, _ = scene_separate
    pipe = build_default_pipeline()
    result = pipe.run(img)
    keys = [s.step_key for s in result.snapshots]
    order = [s.key for s in pipe.steps]
    assert keys == sorted(keys, key=order.index)
    assert "blob_analysis" in keys


def test_roi_filters_parts_by_centroid(scene_separate):
    from conftest import synthetic_pipeline
    img, _ = scene_separate
    roi = (250, 250, 200, 200)         # 只框住左上那一片的中心附近（零件本身超出框外）
    result = synthetic_pipeline().run(img, roi)
    assert result.count == 1
    assert result.parts[0].id == 1                          # 篩選後重新編號
    assert result.parts[0].bbox[2] > roi[2]                 # 輪廓完整，不會被 ROI 裁掉
    assert result.binary.shape == img.shape[:2]
    assert not result.binary[:roi[1]].any()                 # ROI 外為 0


def test_tiny_roi_falls_back_to_full_image(scene_separate):
    img, _ = scene_separate
    result = build_default_pipeline().run(img, (0, 0, 3, 3))
    assert result.roi == (0, 0, img.shape[1], img.shape[0])
