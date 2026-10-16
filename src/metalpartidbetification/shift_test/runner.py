"""鏡像位移測試執行器：原圖辨識一次 → 每個情境：鏡像＋平移 → B 圖辨識 → 配對 → 評估 → 計時。

演算法與參數在開始時複製一份快照，測試全程使用快照，結果才會一致。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..app import MirrorEvaluation, evaluate_match
from ..core import JudgeThresholds, MatchResult, Pipeline, PipelineResult, Roi
from ..imaging import FlipSimParams, simulate_flip
from ..matching import match_flipped
from .scenarios import Scenario


@dataclass
class ScenarioResult:
    scenario: Scenario
    params: FlipSimParams
    image_b: np.ndarray
    result_b: PipelineResult
    match: MatchResult
    evaluation: MirrorEvaluation
    timings: list[tuple[str, float]]     # 原圖辨識（共用一次）、鏡像產生、鏡像辨識、配對

    @property
    def total(self) -> float:
        return sum(t for _, t in self.timings)


class ShiftTestRunner:
    def __init__(self, pipeline: Pipeline, image: np.ndarray, roi: Roi | None,
                 thresholds: JudgeThresholds | None = None) -> None:
        self.pipeline = copy.deepcopy(pipeline)          # 參數快照
        self.image = image
        self.roi = roi                                   # 只用於 ROI 警告，不套用到辨識
        self.thresholds = thresholds or JudgeThresholds()
        self.size = (image.shape[1], image.shape[0])
        self.original: PipelineResult | None = None

    def run_original(self) -> PipelineResult:
        """原圖辨識（整張圖，不套用 ROI）。"""
        self.original = self.pipeline.run(self.image, None)
        return self.original

    def run_scenario(self, sc: Scenario) -> ScenarioResult:
        if self.original is None:
            self.run_original()
        params = sc.params
        t0 = time.perf_counter()
        img_b = simulate_flip(self.image, params)
        t_flip = time.perf_counter() - t0
        t0 = time.perf_counter()
        res_b = self.pipeline.run(img_b, None)
        t_detect = time.perf_counter() - t0
        # 配對演算法拿不到 params（模擬參數）
        t0 = time.perf_counter()
        match = match_flipped(self.original.parts, res_b.parts, self.size, self.thresholds, self.roi)
        t_match = time.perf_counter() - t0
        ev = evaluate_match(self.original, res_b, match, params, self.size)
        timings = [("原圖辨識", self.original.elapsed), ("鏡像產生", t_flip), ("鏡像辨識", t_detect), ("配對", t_match)]
        return ScenarioResult(sc, params, img_b, res_b, match, ev, timings)

    def run_all(self, scenarios: list[Scenario], on_result: Callable[[int, ScenarioResult], None] | None = None,
                should_stop: Callable[[], bool] | None = None) -> list[ScenarioResult]:
        out = []
        for k, sc in enumerate(scenarios):
            if should_stop and should_stop():
                break
            r = self.run_scenario(sc)
            out.append(r)
            if on_result:
                on_result(k, r)
        return out
