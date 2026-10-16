"""前處理步驟：灰階、去雜訊。"""
from __future__ import annotations

import cv2

from ...core import ParamSpec, PipelineContext, PipelineStep, Stage, register_step


@register_step
class Grayscale(PipelineStep):
    key = "grayscale"
    title = "灰階轉換"
    description = "把彩色影像轉成灰階，後續步驟只看亮度。"
    stage = Stage.PREPROCESS
    requires = ("image",)
    provides = ("gray",)

    def run(self, ctx: PipelineContext) -> None:
        img = ctx.data["image"]
        ctx.data["gray"] = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@register_step
class BilateralDenoise(PipelineStep):
    key = "bilateral"
    title = "雙邊濾波去雜訊"
    description = "平滑表面細小雜訊，同時保留零件邊緣（一般模糊會把邊緣也糊掉）。"
    stage = Stage.PREPROCESS
    requires = ("gray",)
    provides = ("gray",)
    param_specs = (
        ParamSpec("diameter", "鄰域直徑", "int", 9, 1, 25, 2, help="越大越平滑，也越慢"),
        ParamSpec("sigma_color", "亮度容忍度", "float", 30.0, 1, 150, 5, help="亮度差在此範圍內才會被平均"),
        ParamSpec("sigma_space", "空間範圍", "float", 9.0, 1, 50, 1),
    )

    def run(self, ctx: PipelineContext) -> None:
        p = self.params
        ctx.data["gray"] = cv2.bilateralFilter(
            ctx.data["gray"], int(p["diameter"]), float(p["sigma_color"]), float(p["sigma_space"])
        )
