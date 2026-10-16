"""流程協調層：串接 檔案讀取 → Pipeline → 繪圖 → 匯出。

GUI 只呼叫這裡的方法，不直接碰演算法或檔案格式；之後要做批次處理或命令列版本也能直接沿用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..algorithms import ALGORITHMS, DEFAULT_ALGORITHM
from ..core import JudgeThresholds, MatchResult, Pipeline, PipelineResult, Roi
from ..fileio import ExportedFiles, export_result, list_images, read_image, write_image
from ..imaging import FlipSimParams, flip_roi, simulate_flip
from ..matching import match_flipped
from ..rendering import draw_all, draw_pairs, draw_single
from .evaluation import MirrorEvaluation, evaluate_match

# 辨識頁的自動鏡像比對：沒有位移 → 位移搜尋範圍 0
RECOGNITION_THRESHOLDS = JudgeThresholds(search_range=0)


@dataclass
class MirrorRun:
    """一次自動鏡像比對的完整結果。"""
    params: FlipSimParams
    image: np.ndarray                   # 鏡像圖
    result: PipelineResult              # 鏡像圖的辨識結果
    match: MatchResult
    timings: list[tuple[str, float]]    # 分段耗時：原圖辨識、鏡像產生、鏡像辨識、配對
    evaluation: MirrorEvaluation

    @property
    def total(self) -> float:
        return sum(t for _, t in self.timings)


class DetectionController:
    def __init__(self, algorithm: str = DEFAULT_ALGORITHM) -> None:
        # 每組演算法各自保有一個 Pipeline（含參數），切換時參數不會被重設
        self._pipelines: dict[str, Pipeline] = {}
        self.algorithm = ""
        self.set_algorithm(algorithm)
        self.images: list[Path] = []
        self.current_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.roi: Roi | None = None
        self.result: PipelineResult | None = None
        self.mirror: MirrorRun | None = None

    # ---------- 演算法選擇 ----------
    @property
    def pipeline(self) -> Pipeline:
        return self._pipelines[self.algorithm]

    def set_algorithm(self, key: str) -> Pipeline:
        """切換演算法組（"light" / "dark"）；第一次使用時才建立該組 Pipeline。"""
        if key not in ALGORITHMS:
            raise KeyError(f"未知的演算法：{key}")
        if key not in self._pipelines:
            self._pipelines[key] = ALGORITHMS[key].build()
        self.algorithm = key
        self.result = None
        self.mirror = None
        return self.pipeline

    # ---------- 影像來源 ----------
    def open_folder(self, folder: str | Path) -> list[Path]:
        """掃描使用者選的資料夾，回傳圖片清單。"""
        self.images = list_images(folder)
        return self.images

    def load_image(self, path: str | Path) -> np.ndarray:
        """載入一張圖；若尺寸和上一張不同就清除 ROI（避免 ROI 超出範圍）。"""
        img = read_image(path)
        if self.current_image is not None and img.shape[:2] != self.current_image.shape[:2]:
            self.roi = None
        self.current_path, self.current_image, self.result, self.mirror = Path(path), img, None, None
        return img

    # ---------- 辨識 ----------
    def set_roi(self, roi: Roi | None) -> None:
        self.roi = roi

    def run(self) -> PipelineResult:
        if self.current_image is None:
            raise RuntimeError("尚未載入圖片")
        self.mirror = None
        self.result = self.pipeline.run(self.current_image, self.roi)
        return self.result

    # ---------- 自動鏡像比對 ----------
    def run_mirror(self, axis: str = "h") -> MirrorRun:
        """以原圖產生鏡像（模擬翻面，本階段無位移）→ 用同一組演算法辨識 → 配對 → 評估。"""
        r = self._require_result()
        h, w = r.original.shape[:2]
        params = FlipSimParams(axis=axis, dx=0, dy=0)

        t0 = time.perf_counter()
        img_b = simulate_flip(r.original, params)
        t_flip = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_b = self.pipeline.run(img_b, flip_roi(self.roi, (w, h), params))
        t_detect = time.perf_counter() - t0

        # 配對演算法只拿到兩份零件清單與影像尺寸，拿不到 params
        t0 = time.perf_counter()
        match = match_flipped(r.parts, res_b.parts, (w, h), RECOGNITION_THRESHOLDS)
        t_match = time.perf_counter() - t0

        timings = [("原圖辨識", r.elapsed), ("鏡像產生", t_flip), ("鏡像辨識", t_detect), ("配對", t_match)]
        evaluation = evaluate_match(r, res_b, match, params, (w, h))
        self.mirror = MirrorRun(params, img_b, res_b, match, timings, evaluation)
        return self.mirror

    def render_mirror(self) -> np.ndarray:
        m = self._require_mirror()
        return draw_all(m.result.original, m.result.parts, m.result.roi if self.roi else None)

    def render_pairs(self, selected: int | None = None) -> np.ndarray:
        r, m = self._require_result(), self._require_mirror()
        return draw_pairs(r.original, r.parts, m.image, m.result.parts, m.match, selected)

    def save_mirror_image(self, path: str | Path) -> None:
        """把鏡像圖（未標註的原始鏡像影像）存到使用者指定的位置。"""
        write_image(path, self._require_mirror().image)

    # ---------- 繪圖 ----------
    def render_all(self) -> np.ndarray:
        r = self._require_result()
        return draw_all(r.original, r.parts, r.roi if self.roi else None)

    def render_single(self, part_id: int) -> np.ndarray:
        r = self._require_result()
        return draw_single(r.original, r.parts, part_id, r.roi if self.roi else None)

    # ---------- 匯出 ----------
    def export(self, out_dir: str | Path) -> ExportedFiles:
        r = self._require_result()
        stem = self.current_path.stem if self.current_path else "result"
        return export_result(r, self.render_all(), out_dir, stem)

    def _require_result(self) -> PipelineResult:
        if self.result is None:
            raise RuntimeError("尚未執行辨識")
        return self.result

    def _require_mirror(self) -> MirrorRun:
        if self.mirror is None:
            raise RuntimeError("尚未執行鏡像比對")
        return self.mirror


