"""形態學步驟：清除二值圖中的雜訊。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import ParamSpec, PipelineContext, PipelineStep, register_step


@register_step
class RemoveSmallEdges(PipelineStep):
    key = "remove_small_edges"
    title = "形態學去雜訊"
    description = ("移除面積太小的邊緣碎片（表面斑點、刮痕），只留下真正的輪廓線；"
                   "可選擇再做閉運算把斷掉的輪廓接起來。")
    requires = ("binary",)
    provides = ("edge_mask",)
    param_specs = (
        ParamSpec("min_area", "最小邊緣面積", "int", 300, 0, 10000, 50, help="小於此像素數的邊緣碎片會被刪除"),
        ParamSpec("close_size", "閉運算大小", "int", 0, 0, 31, 2, help="0 = 不做；越大越能接起斷線，但可能把相鄰零件黏在一起"),
    )

    def run(self, ctx: PipelineContext) -> None:
        b = ctx.data["binary"]
        n, lab, stats, _ = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(self.params["min_area"])
        mask = np.where(keep[lab], 255, 0).astype(np.uint8)
        c = int(self.params["close_size"])
        if c > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
        ctx.data["edge_mask"] = mask
