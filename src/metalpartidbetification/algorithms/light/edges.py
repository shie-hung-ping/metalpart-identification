"""邊緣相關步驟：局部對比（邊緣強度）與二值化。

為什麼不直接對亮度做二值化：實拍圖中零件表面與背景亮度接近、且有暗角，
全域閾值會失敗；但零件輪廓、側壁與縫隙在「局部對比」上非常明顯。
"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import ParamSpec, PipelineContext, PipelineStep, register_step


@register_step
class LocalContrast(PipelineStep):
    key = "local_contrast"
    title = "局部對比（邊緣強度）"
    description = "計算每個像素周圍的亮度標準差：輪廓、側壁、縫隙處數值高，平坦表面與背景數值低。"
    requires = ("gray",)
    provides = ("edge_strength",)
    param_specs = (
        ParamSpec("window", "視窗大小", "int", 7, 3, 31, 2, help="越大邊緣越粗、越不受細紋影響"),
        ParamSpec("percentile", "正規化百分位", "float", 99.5, 90, 100, 0.5, help="以此百分位的值對應到 255"),
    )

    def run(self, ctx: PipelineContext) -> None:
        k = int(self.params["window"])
        g = ctx.data["gray"].astype(np.float32)
        mean = cv2.blur(g, (k, k))
        std = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mean * mean, 0))
        top = max(float(np.percentile(std, float(self.params["percentile"]))), 1e-6)
        ctx.data["edge_strength"] = np.clip(std * 255.0 / top, 0, 255).astype(np.uint8)


@register_step
class Binarize(PipelineStep):
    key = "binarize"
    title = "二值化（邊緣）"
    description = "把邊緣強度圖分成 邊緣(白) / 非邊緣(黑)。Otsu 自動求閾值後再乘上倍率，倍率越小抓到的弱邊緣越多。"
    requires = ("edge_strength",)
    provides = ("binary",)
    param_specs = (
        ParamSpec("method", "閾值方法", "choice", "otsu_scaled", choices=("otsu_scaled", "fixed"),
                  help="otsu_scaled：Otsu×倍率；fixed：固定閾值"),
        ParamSpec("otsu_factor", "Otsu 倍率", "float", 0.3, 0.05, 2.0, 0.05),
        ParamSpec("fixed_threshold", "固定閾值", "int", 30, 1, 254, 1),
    )

    def run(self, ctx: PipelineContext) -> None:
        e = ctx.data["edge_strength"]
        if self.params["method"] == "fixed":
            t = float(self.params["fixed_threshold"])
        else:
            otsu, _ = cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = otsu * float(self.params["otsu_factor"])
        ctx.data["binary"] = np.where(e > t, 255, 0).astype(np.uint8)
