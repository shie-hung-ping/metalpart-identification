"""分割步驟：距離轉換產生種子 → 標記式分水嶺把畫面切成許多區域。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import ParamSpec, PipelineContext, PipelineStep, register_step
from ._viz import colorize_labels


@register_step
class DistanceSeeds(PipelineStep):
    key = "distance_seeds"
    title = "距離轉換找種子點"
    description = ("計算每個非邊緣像素到最近邊緣的距離，距離夠遠的地方就是某個區域的「中心」，"
                   "當作分水嶺的種子；再用開運算切斷細窄的連接，避免兩片零件共用一個種子。")
    requires = ("edge_mask",)
    provides = ("markers",)
    param_specs = (
        ParamSpec("min_distance", "最小離邊距離", "float", 18.0, 2, 100, 1, help="越大種子越少，太大會漏掉窄的零件"),
        ParamSpec("open_size", "種子開運算大小", "int", 21, 0, 81, 2, help="0 = 不做；越大越容易把相連的種子切開"),
    )

    def run(self, ctx: PipelineContext) -> None:
        not_edge = (ctx.data["edge_mask"] == 0).astype(np.uint8)
        dist = cv2.distanceTransform(not_edge, cv2.DIST_L2, 5)
        seeds = (dist > float(self.params["min_distance"])).astype(np.uint8)
        k = int(self.params["open_size"])
        if k > 1:
            seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        _, markers = cv2.connectedComponents(seeds)
        ctx.data["markers"] = markers.astype(np.int32)

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        return colorize_labels(ctx.data["markers"])


@register_step
class Watershed(PipelineStep):
    key = "watershed"
    title = "標記式分水嶺"
    description = "從每個種子開始沿著邊緣強度圖「淹水」，水在邊緣（山脊）處相遇形成分界線，把畫面切成多個區域。"
    requires = ("markers", "edge_strength")
    provides = ("labels",)

    def run(self, ctx: PipelineContext) -> None:
        relief = cv2.cvtColor(ctx.data["edge_strength"], cv2.COLOR_GRAY2BGR)  # OpenCV 分水嶺需要 3 通道
        ctx.data["labels"] = cv2.watershed(relief, ctx.data["markers"].copy())

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        return colorize_labels(ctx.data["labels"])
