"""區域步驟：把分水嶺切出的區域分類（零件 / 陰影 / 背景），再把同一片零件被切碎的區域合併回來。"""
from __future__ import annotations

import cv2
import numpy as np

from ...core import ParamSpec, PipelineContext, PipelineStep, register_step
from ._viz import colorize_labels

PART, SHADOW, BACKGROUND = "part", "shadow", "background"


@register_step
class ClassifyRegions(PipelineStep):
    key = "classify_regions"
    title = "區域分類（零件/陰影/背景）"
    description = ("碰到影像邊界的區域 = 背景；其餘區域依平均亮度做一次 Otsu 分成暗、亮兩群，"
                   "暗且表面平滑的 = 陰影或縫隙，其他 = 零件表面。")
    requires = ("labels", "gray", "edge_strength")
    provides = ("region_kind",)
    param_specs = (
        ParamSpec("shadow_texture_max", "陰影最大紋理", "float", 30.0, 0, 255, 1,
                  help="暗區域的平均邊緣強度低於此值才算陰影；有紋理的暗區（例如鏽斑零件）仍算零件"),
        ParamSpec("min_contrast", "暗亮群最小差距", "float", 25.0, 0, 255, 1,
                  help="暗群與亮群平均亮度差小於此值時，視為畫面中沒有陰影（全部當零件）"),
    )

    def run(self, ctx: PipelineContext) -> None:
        labels = ctx.data["labels"]
        n = int(labels.max()) + 1
        flat = labels.ravel()
        valid = flat > 0
        # 用 bincount 一次算出每個區域的面積、平均亮度、平均紋理
        area = np.bincount(flat[valid], minlength=n).astype(np.float64)
        gray_sum = np.bincount(flat[valid], weights=ctx.data["gray"].ravel()[valid], minlength=n)
        tex_sum = np.bincount(flat[valid], weights=ctx.data["edge_strength"].ravel()[valid], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_gray = gray_sum / area
            mean_tex = tex_sum / area

        # 分水嶺會把影像最外圈設為 -1，因此往內一圈判斷是否碰到邊界
        inner = labels[1:-1, 1:-1] if min(labels.shape) > 2 else labels
        border = set(np.unique(np.concatenate([inner[0], inner[-1], inner[:, 0], inner[:, -1]])).tolist())

        kinds: dict[int, str] = {}
        candidates = [l for l in range(1, n) if area[l] > 0 and l not in border]
        for l in range(1, n):
            if area[l] > 0 and l in border:
                kinds[l] = BACKGROUND
        if len(candidates) >= 2:
            vals = np.array([mean_gray[l] for l in candidates], np.float64)
            vals8 = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
            thr, _ = cv2.threshold(vals8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark, bright = vals[vals <= thr], vals[vals > thr]
            # 兩群亮度差太小 → 代表沒有真正的陰影，不做區分
            if dark.size == 0 or bright.size == 0 or bright.mean() - dark.mean() < float(self.params["min_contrast"]):
                thr = -1.0
        else:
            thr = -1.0
        tex_max = float(self.params["shadow_texture_max"])
        for l in candidates:
            dark = mean_gray[l] <= thr
            kinds[l] = SHADOW if dark and mean_tex[l] < tex_max else PART
        ctx.data["region_kind"] = kinds

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        # 綠 = 零件、藍 = 陰影、灰 = 背景、白 = 分界線
        labels = ctx.data["labels"]
        colors = {PART: (80, 200, 80), SHADOW: (200, 90, 40), BACKGROUND: (70, 70, 70)}
        lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        for l, k in ctx.data["region_kind"].items():
            lut[l] = colors[k]
        out = lut[np.clip(labels, 0, None)]
        out[labels == -1] = 255
        return out


@register_step
class MergeRegions(PipelineStep):
    key = "merge_regions"
    title = "依邊界強度合併區域"
    description = ("同一片零件可能被切成好幾塊：相鄰兩塊零件區域之間的分界線若邊緣強度很弱（不是真的輪廓），"
                   "就合併成同一片；真正的零件交界有側壁/縫隙，邊緣強度高，不會被合併。")
    requires = ("labels", "region_kind", "edge_strength")
    provides = ("part_labels",)
    param_specs = (
        ParamSpec("merge_threshold", "合併閾值", "float", 80.0, 0, 255, 5,
                  help="分界線平均邊緣強度低於此值就合併；越大越容易合併"),
        ParamSpec("min_boundary", "最短共用邊界", "int", 15, 1, 500, 1, help="共用邊界太短（只碰到一點）不合併"),
    )

    def run(self, ctx: PipelineContext) -> None:
        labels = ctx.data["labels"]
        kinds: dict[int, str] = ctx.data["region_kind"]
        n = int(labels.max()) + 1
        sums, counts = _boundary_stats(labels, ctx.data["edge_strength"], n)

        # Union-Find：由最弱的分界線開始合併
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        thr = float(self.params["merge_threshold"])
        min_len = int(self.params["min_boundary"])
        for (a, b), c in sorted(counts.items(), key=lambda kv: sums[kv[0]] / kv[1]):
            if kinds.get(a) != PART or kinds.get(b) != PART or c < min_len:
                continue
            if sums[(a, b)] / c < thr:
                parent[find(a)] = find(b)

        # 把每個合併群組重新編號為 1..G，非零件為 0
        lut = np.zeros(n, np.int32)
        group_ids: dict[int, int] = {}
        for l, k in kinds.items():
            if k == PART:
                root = find(l)
                lut[l] = group_ids.setdefault(root, len(group_ids) + 1)
        ctx.data["part_labels"] = lut[np.clip(labels, 0, None)]

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        return colorize_labels(ctx.data["part_labels"])


def _boundary_stats(labels: np.ndarray, strength: np.ndarray, n: int):
    """統計每一對相鄰區域之間分界線（labels == -1）的邊緣強度總和與像素數。"""
    h, w = labels.shape
    ys, xs = np.nonzero(labels[1:-1, 1:-1] == -1)
    ys += 1
    xs += 1
    if ys.size == 0:
        return {}, {}
    # 取 8 鄰域的標記，找出最小與最大的正標記 → 視為這個分界像素兩側的區域
    neigh = np.stack([labels[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=1)
    big = np.iinfo(np.int32).max
    lo = np.where(neigh > 0, neigh, big).min(axis=1)
    hi = neigh.max(axis=1)
    ok = (lo != big) & (lo < hi)
    code = lo[ok].astype(np.int64) * n + hi[ok]
    vals = strength[ys[ok], xs[ok]].astype(np.float64)
    uniq, inv = np.unique(code, return_inverse=True)
    s = np.bincount(inv, weights=vals)
    c = np.bincount(inv)
    sums, counts = {}, {}
    for u, si, ci in zip(uniq.tolist(), s.tolist(), c.tolist()):
        key = (u // n, u % n)
        sums[key], counts[key] = si, ci
    return sums, counts
