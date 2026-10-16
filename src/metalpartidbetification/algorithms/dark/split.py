"""深色背景：把相連的零件切開（接縫切割 → 距離轉換分水嶺 → Blob 候選輪廓）。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import DetectedPart, ParamSpec, PipelineContext, PipelineStep, register_step
from .mask import _fill_small_holes


@register_step
class DarkSeamCut(PipelineStep):
    key = "dark_seam_cut"
    title = "接縫切割（黑帽運算）"
    description = ("緊靠的零件之間會有一條細細的暗接縫：黑帽運算（閉運算 − 原圖）會把比周圍暗的細線凸顯出來，"
                   "把這些接縫從零件遮罩中挖掉，相連的零件就被切開。")
    requires = ("mask", "smooth")
    provides = ("cut_mask", "seams")
    param_specs = (
        ParamSpec("kernel", "接縫最大寬度", "int", 15, 3, 61, 2, help="比此寬度窄的暗線才視為接縫"),
        ParamSpec("threshold", "接縫強度閾值", "int", 35, 1, 255, 1, help="黑帽值高於此值才算接縫；越小切得越多"),
        ParamSpec("max_hole_area", "切割後補孔面積", "int", 5000, 0, 100000, 500, help="切割後再補回定位孔"),
    )

    def run(self, ctx: PipelineContext) -> None:
        k = int(self.params["kernel"])
        bh = cv2.morphologyEx(ctx.data["smooth"], cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        seams = (bh > int(self.params["threshold"])).astype(np.uint8)
        mask = (ctx.data["mask"] > 0).astype(np.uint8)
        cut = mask & (1 - seams)
        # 定位孔也會被黑帽運算凸顯而被挖掉，切割後把小孔補回來（接縫是開放的線，不會被補回）
        cut = _fill_small_holes(cut, int(self.params["max_hole_area"]))
        ctx.data["seams"] = (seams & mask) * 255
        ctx.data["cut_mask"] = cut * 255

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = cv2.cvtColor(ctx.data["cut_mask"], cv2.COLOR_GRAY2BGR)
        out[ctx.data["seams"] > 0] = (0, 0, 255)        # 紅色 = 被切掉的接縫
        return out


@register_step
class DarkWatershed(PipelineStep):
    key = "dark_watershed"
    title = "距離轉換 + 分水嶺"
    description = ("計算遮罩內每點到邊緣的距離，距離大的地方是零件中心（種子）；"
                   "從種子沿距離圖淹水，在零件之間的細頸處相遇形成分界。")
    requires = ("cut_mask",)
    provides = ("regions",)
    param_specs = (
        ParamSpec("min_distance", "種子最小離邊距離", "float", 25.0, 2, 150, 1, help="越大種子越少；太大會漏掉窄的零件"),
        ParamSpec("seed_ratio", "種子相對門檻", "float", 0.70, 0.1, 1.0, 0.05,
                  help="種子需達到所在區塊最大離邊距離的此比例；避免零件較窄的翼部被當成另一片零件的中心"),
        ParamSpec("open_size", "種子開運算大小", "int", 15, 0, 81, 2, help="切斷種子之間細窄的連接"),
    )

    def run(self, ctx: PipelineContext) -> None:
        cut = (ctx.data["cut_mask"] > 0).astype(np.uint8)
        dist = cv2.distanceTransform(cut, cv2.DIST_L2, 5)
        # 每個連通區塊各自計算門檻 = max(最小離邊距離, 區塊內最大距離 × 相對門檻)
        n_comp, comp = cv2.connectedComponents(cut)
        comp_max = np.zeros(n_comp, np.float32)
        np.maximum.at(comp_max, comp.ravel(), dist.ravel())
        thr = np.maximum(float(self.params["min_distance"]), comp_max * float(self.params["seed_ratio"]))
        seeds = ((dist > thr[comp]) & (cut > 0)).astype(np.uint8)
        k = int(self.params["open_size"])
        if k > 1:
            seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, markers = cv2.connectedComponents(seeds)
        markers = markers + 1                    # 1 = 背景，2.. = 種子
        markers[(cut > 0) & (seeds == 0)] = 0    # 0 = 待淹水區
        relief = cv2.normalize(-dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        labels = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
        labels[cut == 0] = 1                      # 遮罩外一律為背景
        ctx.data["regions"] = labels

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        labels = ctx.data["regions"]
        rng = np.random.default_rng(7)
        palette = rng.integers(60, 255, size=(int(labels.max()) + 2, 3), dtype=np.uint8)
        palette[0] = 255
        palette[1] = 0
        return palette[np.clip(labels, 0, None)]


@register_step
class DarkBlobs(PipelineStep):
    key = "dark_blobs"
    title = "Blob 分析（候選輪廓）"
    description = "每個分水嶺區域取外輪廓，計算面積、實心度；太小的視為雜訊捨棄。"
    requires = ("regions",)
    provides = ("candidates",)
    param_specs = (
        ParamSpec("min_area", "最小區塊面積", "int", 3000, 0, 1_000_000, 500),
        ParamSpec("close_size", "輪廓平滑大小", "int", 5, 0, 31, 2),
    )

    def run(self, ctx: PipelineContext) -> None:
        labels = ctx.data["regions"]
        k = int(self.params["close_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
        out: list[DetectedPart] = []
        for lbl in range(2, int(labels.max()) + 1):
            m = (labels == lbl).astype(np.uint8)
            if not m.any():
                continue
            if kernel is not None:
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area < int(self.params["min_area"]):
                continue
            mo = cv2.moments(c)
            hull = float(cv2.contourArea(cv2.convexHull(c)))
            out.append(DetectedPart(
                id=len(out) + 1, contour=c, bbox=tuple(int(v) for v in cv2.boundingRect(c)), area=area,
                centroid=(mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]), solidity=area / hull if hull else 0.0,
            ))
        ctx.data["candidates"] = out

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = ctx.data["image"].copy()
        for c in ctx.data["candidates"]:
            cv2.drawContours(out, [c.contour], -1, (0, 0, 255), 3)
        return out
