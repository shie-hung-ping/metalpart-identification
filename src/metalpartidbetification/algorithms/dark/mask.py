"""深色背景：建立零件遮罩（亮度二值化 → 形態學 → 排除邊界區域）。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import ParamSpec, PipelineContext, PipelineStep, register_step


@register_step
class DarkThreshold(PipelineStep):
    key = "dark_threshold"
    title = "亮度二值化"
    description = "零件比背景亮很多：以 Otsu 自動求亮度閾值，亮於閾值者為零件（白）。"
    requires = ("smooth",)
    provides = ("binary",)
    param_specs = (
        ParamSpec("method", "閾值方法", "choice", "otsu", choices=("otsu", "fixed"), help="otsu：自動；fixed：固定閾值"),
        ParamSpec("otsu_offset", "Otsu 偏移", "int", 0, -100, 100, 1, help="自動閾值再加上此值；正值 = 更嚴格"),
        ParamSpec("fixed_threshold", "固定閾值", "int", 80, 1, 254, 1),
    )

    def run(self, ctx: PipelineContext) -> None:
        g = ctx.data["smooth"]
        if self.params["method"] == "fixed":
            t = float(self.params["fixed_threshold"])
        else:
            t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t += float(self.params["otsu_offset"])
        ctx.data["binary"] = np.where(g > t, 255, 0).astype(np.uint8)


@register_step
class DarkMorphology(PipelineStep):
    key = "dark_morphology"
    title = "形態學（去灰塵、補孔）"
    description = "開運算去除背景上的灰塵亮點；填補零件內部的小孔（定位孔），大的空洞（零件之間的空隙）保留。"
    requires = ("binary",)
    provides = ("mask",)
    param_specs = (
        ParamSpec("open_size", "開運算大小", "int", 5, 0, 31, 2, help="0 = 不做"),
        ParamSpec("max_hole_area", "最大補孔面積", "int", 5000, 0, 100000, 500, help="小於此面積的內部孔洞才填補"),
    )

    def run(self, ctx: PipelineContext) -> None:
        m = (ctx.data["binary"] > 0).astype(np.uint8)
        k = int(self.params["open_size"])
        if k > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ctx.data["mask"] = _fill_small_holes(m, int(self.params["max_hole_area"])) * 255

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        return ctx.data["mask"]


@register_step
class DarkClearBorder(PipelineStep):
    key = "dark_clear_border"
    title = "排除邊界區域"
    description = "移除碰到影像邊緣的亮區（例如畫面上方的托盤邊），以及面積太小的亮點。"
    requires = ("mask",)
    provides = ("mask",)
    param_specs = (
        ParamSpec("min_area", "最小區域面積", "int", 3000, 0, 1_000_000, 500),
    )

    def run(self, ctx: PipelineContext) -> None:
        m = (ctx.data["mask"] > 0).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
        keep = np.zeros(n, bool)
        for i in range(1, n):
            keep[i] = i not in border and st[i, cv2.CC_STAT_AREA] >= int(self.params["min_area"])
        ctx.data["mask"] = np.where(keep[lab], 255, 0).astype(np.uint8)

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        return ctx.data["mask"]


def _fill_small_holes(m: np.ndarray, max_area: int) -> np.ndarray:
    """填補面積小於 max_area 的內部孔洞（以背景的連通區塊判斷：沒碰到邊界且夠小的就是孔）。"""
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    fill = np.zeros(n, bool)
    for i in range(1, n):
        fill[i] = i not in border and st[i, cv2.CC_STAT_AREA] < max_area
    out = m.copy()
    out[fill[lab]] = 1
    return out
