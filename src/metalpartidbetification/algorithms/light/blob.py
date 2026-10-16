"""Blob 分析：把每個零件區域轉成輪廓並計算特徵，過濾雜訊並標示可疑（可能多片相黏）的區塊。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import DetectedPart, ParamSpec, PipelineContext, PipelineStep, register_step


@register_step
class BlobAnalysis(PipelineStep):
    key = "blob_analysis"
    title = "Blob 分析與計數"
    description = ("每個零件區域：閉運算平滑邊緣、開運算去除細長延伸、填補定位孔，計算面積、外框、重心、實心度；"
                   "面積太小視為雜訊捨棄；面積遠大於一般零件時標示「可能多片相黏」並估計片數。")
    requires = ("part_labels",)
    provides = ("parts",)
    param_specs = (
        ParamSpec("min_area", "最小區塊面積", "int", 3000, 0, 1_000_000, 500,
                  help="小於此像素數的區塊視為雜訊；較大的碎片會保留給「輪廓信心評分」判斷"),
        ParamSpec("close_size", "輪廓平滑大小", "int", 7, 0, 31, 2),
        ParamSpec("open_size", "去除細長延伸", "int", 15, 0, 61, 2,
                  help="開運算大小：移除沿著鄰近零件側壁延伸出去的細帶（避免一片零件把另一片「包」起來）"),
        ParamSpec("oversize_ratio", "相黏判定倍率", "float", 1.6, 1.1, 5.0, 0.1,
                  help="面積 ≥ 中位數面積 × 此倍率，就標示為可能多片相黏"),
    )

    def run(self, ctx: PipelineContext) -> None:
        part_labels = ctx.data["part_labels"]
        p = self.params
        kc, ko = int(p["close_size"]), int(p["open_size"])
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)) if kc > 1 else None
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)) if ko > 1 else None

        found: list[tuple[np.ndarray, float]] = []
        for gid in range(1, int(part_labels.max()) + 1):
            mask = (part_labels == gid).astype(np.uint8)
            if close_k is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
            if open_k is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)   # 只取外輪廓 → 等同填補內部定位孔
            area = float(cv2.contourArea(c))
            if area >= float(p["min_area"]):
                found.append((c, area))

        # 以面積中位數當作「單片零件」的參考面積
        ref = float(np.median([a for _, a in found])) if found else 0.0
        parts: list[DetectedPart] = []
        # 依位置排序（由上而下、由左而右），讓編號穩定
        found.sort(key=lambda ca: (cv2.boundingRect(ca[0])[1] // 100, cv2.boundingRect(ca[0])[0]))
        for i, (c, area) in enumerate(found, start=1):
            m = cv2.moments(c)
            cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0.0, 0.0)
            hull_area = float(cv2.contourArea(cv2.convexHull(c)))
            est = 1
            if ref > 0 and len(found) >= 3 and area >= ref * float(p["oversize_ratio"]):
                est = max(2, int(round(area / ref)))   # 警告訊息由 Pipeline 在 ROI 篩選、重新編號後產生
            parts.append(DetectedPart(
                id=i, contour=c, bbox=tuple(int(v) for v in cv2.boundingRect(c)), area=area,
                centroid=(float(cx), float(cy)), solidity=area / hull_area if hull_area else 0.0,
                estimated_count=est,
            ))
        ctx.data["parts"] = parts

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        img = ctx.data["image"]
        out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for part in ctx.data["parts"]:
            color = (0, 215, 255) if part.suspicious else (0, 0, 255)
            cv2.drawContours(out, [part.contour], -1, color, 3)
        return out
