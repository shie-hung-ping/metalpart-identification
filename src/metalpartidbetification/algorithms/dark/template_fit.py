"""深色背景：自我樣板比對（只處理有疑慮的輪廓）。

與淺色版概念相同，但依深色背景重新設計：深色背景下零件遮罩很可靠，
因此擺放評分以「樣板是否落在亮的零件區域上」為主，落在暗處（空隙、背景）就直接拒絕。

  ⑦ DarkReview     選樣板（對稱度最高的完整單片）、分出高信心單片與有疑慮輪廓
  ⑧ DarkPlacement  在有疑慮輪廓上逐片擺放樣板（旋轉＋平移，不可重疊）
  ⑨ DarkFinalize   合併結果、編號
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core import DetectedPart, ParamSpec, PipelineContext, PipelineStep, register_step

C_OK, C_DOUBT, C_TEMPLATE, C_PLACED = (60, 200, 60), (0, 215, 255), (255, 120, 0), (255, 0, 255)


# ------------------------------ 幾何工具（深色版自有） ------------------------------
def _center(contour: np.ndarray) -> np.ndarray:
    m = cv2.moments(contour)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else contour[:, 0, :].mean(0)


def _rotate_points(pts: np.ndarray, deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return pts @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])


def _raster(pts: np.ndarray, size: int) -> np.ndarray:
    """把以畫布中心為原點的點集畫成 size×size 的填滿遮罩。"""
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [(pts + size / 2).round().astype(np.int32)], 1)
    return m


def reflect_symmetry(contour: np.ndarray, step_deg: int = 3, grid: int = 96) -> float:
    """鏡像對稱度：把輪廓轉到各角度後左右翻轉，取與原形狀的最大 IoU。"""
    pts = contour[:, 0, :].astype(np.float64) - _center(contour)
    scale = (grid * 0.45) / max(np.abs(pts).max(), 1.0)          # 正規化到固定大小的格子
    pts = pts * scale
    best = 0.0
    for deg in range(0, 180, step_deg):
        p = _rotate_points(pts, deg)
        a = _raster(p, grid)
        b = _raster(p * [-1, 1], grid)                             # 對垂直軸鏡射
        best = max(best, float((a & b).sum()) / max(float((a | b).sum()), 1.0))
    return best


@dataclass
class DarkReviewResult:
    template: DetectedPart | None
    confident: list[DetectedPart] = field(default_factory=list)
    doubtful: list[DetectedPart] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)
    symmetry: dict[int, float] = field(default_factory=dict)


# ------------------------------ ⑦ 選樣板 ------------------------------
@register_step
class DarkReview(PipelineStep):
    key = "dark_review"
    title = "輪廓信心評分（選樣板）"
    description = ("對每個候選輪廓計算鏡像對稱度與實心度，對稱度最高的完整單片當樣板（藍）；"
                   "面積與形狀和樣板一致者為高信心單片（綠），其餘為有疑慮輪廓（黃）。")
    requires = ("candidates",)
    provides = ("review",)
    param_specs = (
        ParamSpec("min_symmetry", "樣板最低對稱度", "float", 0.80, 0, 1, 0.01),
        ParamSpec("solidity_min", "實心度下限", "float", 0.55, 0, 1, 0.01),
        ParamSpec("solidity_max", "實心度上限", "float", 0.75, 0, 1, 0.01),
        ParamSpec("area_tolerance", "面積容許差", "float", 0.20, 0.05, 1, 0.05),
        ParamSpec("shape_tolerance", "形狀容許差", "float", 0.25, 0.01, 2, 0.01, help="與樣板的 Hu 矩形狀距離上限"),
    )

    def run(self, ctx: PipelineContext) -> None:
        p = self.params
        cands: list[DetectedPart] = ctx.data["candidates"]
        sym = {c.id: reflect_symmetry(c.contour) for c in cands}
        in_range = [c for c in cands if p["solidity_min"] <= c.solidity <= p["solidity_max"]]
        pool = [c for c in in_range if sym[c.id] >= p["min_symmetry"]]
        tpl = max(pool, key=lambda c: sym[c.id], default=None)
        res = DarkReviewResult(template=tpl, symmetry=sym)
        for c in cands:
            if tpl is None:
                res.doubtful.append(c)
                res.reasons[c.id] = "找不到可當樣板的完整單片"
                continue
            ratio = c.area / tpl.area
            dist = 0.0 if c is tpl else cv2.matchShapes(c.contour, tpl.contour, cv2.CONTOURS_MATCH_I1, 0)
            same = abs(ratio - 1) <= p["area_tolerance"] and dist <= p["shape_tolerance"] and c in in_range
            if c is tpl or same:
                res.confident.append(c)
            else:
                res.doubtful.append(c)
                res.reasons[c.id] = (f"面積為樣板的 {ratio:.1f} 倍，疑似多片相連" if ratio > 1 + p["area_tolerance"]
                                     else f"面積僅樣板的 {ratio:.1f} 倍，疑似碎片" if ratio < 1 - p["area_tolerance"]
                                     else "形狀與樣板不一致")
        ctx.data["review"] = res

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        r: DarkReviewResult = ctx.data["review"]
        out = ctx.data["image"].copy()
        cv2.drawContours(out, [c.contour for c in r.confident], -1, C_OK, 3)
        cv2.drawContours(out, [c.contour for c in r.doubtful], -1, C_DOUBT, 3)
        if r.template is not None:
            cv2.drawContours(out, [r.template.contour], -1, C_TEMPLATE, 6)
        return out


# ------------------------------ ⑧ 擺放樣板 ------------------------------
@dataclass
class DarkPlacementItem:
    contour: np.ndarray
    score: float
    ok: bool
    note: str = ""


@register_step
class DarkPlacement(PipelineStep):
    key = "dark_placement"
    title = "樣板比對（只處理有疑慮輪廓）"
    description = ("在有疑慮的輪廓上，把樣板旋轉＋平移逐片擺放：樣板必須幾乎完全落在亮的零件區域上、"
                   "不能和已確認的零件重疊，並以覆蓋疑慮輪廓最多的位置為優先。")
    requires = ("review", "mask", "image")
    provides = ("placements",)
    param_specs = (
        ParamSpec("angle_step", "旋轉角度間隔", "int", 3, 1, 15, 1),
        ParamSpec("search_scale", "搜尋縮放比例", "float", 0.25, 0.1, 1.0, 0.05),
        ParamSpec("min_on_part", "最低落在零件上比例", "float", 0.88, 0.5, 1, 0.01,
                  help="樣板面積中落在亮區（零件遮罩）的比例下限；防止擺到空隙或背景"),
        ParamSpec("max_overlap", "最大重疊比例", "float", 0.08, 0, 0.5, 0.01),
        ParamSpec("min_cover", "最低覆蓋率", "float", 0.70, 0.3, 1, 0.05,
                  help="樣板覆蓋「剩餘疑慮區域」的比例（以兩者中較小者為分母）"),
        ParamSpec("stop_ratio", "停止剩餘比例", "float", 0.40, 0.1, 1, 0.05),
    )

    def run(self, ctx: PipelineContext) -> None:
        r: DarkReviewResult = ctx.data["review"]
        items: list[DarkPlacementItem] = []
        ctx.data["placements"] = items
        if r.template is None:
            items += [DarkPlacementItem(c.contour, 0.0, False, r.reasons.get(c.id, "")) for c in r.doubtful]
            return
        if not r.doubtful:
            return
        p = self.params
        h, w = ctx.data["image"].shape[:2]
        s = float(p["search_scale"])
        size_small = (max(1, round(w * s)), max(1, round(h * s)))
        part_mask = (ctx.data["mask"] > 0).astype(np.uint8)
        occupied = np.zeros((h, w), np.uint8)
        cv2.drawContours(occupied, [c.contour for c in r.confident], -1, 1, -1)

        tpl_pts = r.template.contour[:, 0, :].astype(np.float64) - _center(r.template.contour)
        tpl_area = r.template.area
        bank = _build_bank(tpl_pts * s, int(p["angle_step"]))

        for group in _cluster(r.doubtful, (h, w)):
            target = np.zeros((h, w), np.uint8)
            cv2.drawContours(target, [c.contour for c in group], -1, 1, -1)
            target &= 1 - occupied
            why = "；".join(r.reasons.get(c.id, "") for c in group)
            placed = 0
            while target.sum() >= tpl_area * p["stop_ratio"]:
                hit = _best_pose(bank, target, occupied, part_mask, size_small, s, p)
                if hit is None:
                    break
                deg, cx, cy, score = hit
                contour = (_rotate_points(tpl_pts, deg) + [cx, cy]).round().astype(np.int32).reshape(-1, 1, 2)
                pm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(pm, [contour], 1)
                items.append(DarkPlacementItem(contour, score, True))
                occupied |= pm
                target &= 1 - pm
                placed += 1
            if target.sum() >= tpl_area * p["stop_ratio"]:
                cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) >= tpl_area * p["stop_ratio"]:
                        note = "樣板擺放後仍有無法解釋的區域" if placed else f"樣板無法擺放（{why}）"
                        items.append(DarkPlacementItem(c, 0.0, False, note))

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = ctx.data["image"].copy()
        cv2.drawContours(out, [c.contour for c in ctx.data["review"].confident], -1, C_OK, 2)
        for it in ctx.data["placements"]:
            cv2.drawContours(out, [it.contour], -1, C_PLACED if it.ok else C_DOUBT, 4)
        return out


def _build_bank(pts_small: np.ndarray, step: int):
    """預先產生樣板各角度的縮小遮罩：(角度, 遮罩, 像素數, 半徑)。"""
    radius = int(np.ceil(np.sqrt((pts_small ** 2).sum(1)).max())) + 2
    size = 2 * radius + 1
    bank = []
    for deg in range(0, 360, step):
        q = _rotate_points(pts_small, deg) + radius
        m = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m, [q.round().astype(np.int32)], 1)
        bank.append((deg, m.astype(np.float32), float(m.sum()), radius))
    return bank


def _best_pose(bank, target, occupied, part_mask, size_small, s, p):
    """在 target 周邊搜尋最佳擺放 → (角度, 中心x, 中心y, 分數)；沒有合格位置回傳 None。"""
    def small(a):
        return cv2.resize(a.astype(np.float32), size_small, interpolation=cv2.INTER_AREA)
    t, o, f = small(target), small(occupied), small(part_mask)
    ys, xs = np.nonzero(t > 0.5)
    if xs.size == 0:
        return None
    radius = bank[0][3]
    pad = 2 * radius + 2
    x0, y0 = xs.min() - radius, ys.min() - radius
    x1, y1 = xs.max() + radius + 1, ys.max() + radius + 1

    def window(a):
        a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        return a[y0 + pad - radius:y1 + pad + radius, x0 + pad - radius:x1 + pad + radius]

    tw, ow, fw = window(t), window(o), window(f)
    remain = float(t.sum())
    best = None
    for deg, m, area, _ in bank:
        on_part = cv2.matchTemplate(fw, m, cv2.TM_CCORR) / area
        overlap = cv2.matchTemplate(ow, m, cv2.TM_CCORR) / area
        cover = np.minimum(cv2.matchTemplate(tw, m, cv2.TM_CCORR) / min(area, remain), 1.0)
        ok = (on_part >= p["min_on_part"]) & (overlap <= p["max_overlap"]) & (cover >= p["min_cover"])
        score = np.where(ok, cover + on_part, -1.0)
        _, mx, _, (lx, ly) = cv2.minMaxLoc(score)
        if mx > 0 and (best is None or mx > best[3]):
            best = (deg, (lx + x0) / s, (ly + y0) / s, float(mx))
    return best


def _cluster(parts: list[DetectedPart], shape, gap: int = 15) -> list[list[DetectedPart]]:
    """距離在 gap 像素內的輪廓視為同一群。"""
    canvas = np.zeros(shape, np.uint8)
    cv2.drawContours(canvas, [c.contour for c in parts], -1, 1, -1)
    canvas = cv2.dilate(canvas, np.ones((gap, gap), np.uint8))
    _, comp = cv2.connectedComponents(canvas)
    groups: dict[int, list[DetectedPart]] = {}
    for c in parts:
        x, y = c.contour[0, 0]
        groups.setdefault(int(comp[y, x]), []).append(c)
    return list(groups.values())


# ------------------------------ ⑨ 結果合併 ------------------------------
@register_step
class DarkFinalize(PipelineStep):
    key = "dark_finalize"
    title = "結果合併與計數"
    description = "最終零件 = 高信心單片 + 樣板擺放成功者；無法判定的區域以黃色列出、不計入總數。"
    requires = ("review", "placements")
    provides = ("parts",)

    def run(self, ctx: PipelineContext) -> None:
        r: DarkReviewResult = ctx.data["review"]
        rows = [(c.contour, "segment", r.symmetry.get(c.id, 1.0), 1, "") for c in r.confident]
        for it in ctx.data["placements"]:
            rows.append((it.contour, "template", min(1.0, it.score / 2), 1, "") if it.ok
                        else (it.contour, "unresolved", 0.0, 0, it.note + "（未計入總數）"))
        parts = []
        for contour, source, conf, count, note in rows:
            area = float(cv2.contourArea(contour))
            hull = float(cv2.contourArea(cv2.convexHull(contour)))
            cx, cy = _center(contour)
            parts.append(DetectedPart(0, contour, tuple(int(v) for v in cv2.boundingRect(contour)), area,
                                      (float(cx), float(cy)), area / hull if hull else 0.0,
                                      count, source, float(conf), note))
        parts.sort(key=lambda d: (d.bbox[1] // 100, d.bbox[0]))
        for i, d in enumerate(parts, start=1):
            d.id = i
        ctx.data["parts"] = parts

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = ctx.data["image"].copy()
        for d in ctx.data["parts"]:
            col = C_DOUBT if d.suspicious else (C_PLACED if d.source == "template" else (0, 0, 255))
            cv2.drawContours(out, [d.contour], -1, col, 3)
        return out
