"""深色背景前處理：灰階、高斯模糊。"""
from __future__ import annotations

import cv2

from ...core import ParamSpec, PipelineContext, PipelineStep, Stage, register_step


@register_step
class DarkGrayscale(PipelineStep):
    key = "dark_grayscale"
    title = "灰階轉換"
    description = "把彩色影像轉成灰階；深色背景下零件亮、背景暗，亮度就是主要判斷依據。"
    stage = Stage.PREPROCESS
    requires = ("image",)
    provides = ("gray",)

    def run(self, ctx: PipelineContext) -> None:
        img = ctx.data["image"]
        ctx.data["gray"] = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@register_step
class DarkGaussianBlur(PipelineStep):
    key = "dark_gaussian"
    title = "高斯模糊"
    description = "輕微模糊，壓低表面刮痕與背景灰塵造成的雜訊。"
    stage = Stage.PREPROCESS
    requires = ("gray",)
    provides = ("smooth",)
    param_specs = (
        ParamSpec("ksize", "模糊核大小", "int", 5, 1, 31, 2, help="奇數；越大越平滑，但細接縫也會變淡"),
    )

    def run(self, ctx: PipelineContext) -> None:
        k = int(self.params["ksize"]) | 1          # 確保為奇數
        ctx.data["smooth"] = cv2.GaussianBlur(ctx.data["gray"], (k, k), 0)
