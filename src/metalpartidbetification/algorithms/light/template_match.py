"""自我樣板比對：從同一張圖中挑出最有把握的「完整單片」輪廓當樣板，
只用它去辨識有疑慮的輪廓（疑似多片相連、疑似碎片）。

前提（使用者提供）：零件彼此不會重疊，只會緊靠 → 擺放的樣板之間不可重疊。

步驟：
  ⑨ ReviewContours   輪廓信心評分：選樣板、分出高信心單片與有疑慮輪廓
  ⑩ TemplatePlacement 在有疑慮輪廓內，用樣板（旋轉＋平移）逐片擺放
  ⑪ FinalizeParts    合併 高信心單片 + 樣板擺放 + 無法判定 → 最終零件清單
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core import DetectedPart, ParamSpec, PipelineContext, PipelineStep, register_step

YELLOW, GREEN, BLUE, MAGENTA = (0, 215, 255), (60, 200, 60), (255, 120, 0), (255, 0, 255)


# ============================== 共用工具 ==============================
def _mask_of(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 1, -1)
    return m


def _centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(contour)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (0.0, 0.0)


def mirror_symmetry(contour: np.ndarray, step_deg: int = 2) -> float:
    """鏡像對稱度：對所有通過重心的軸鏡射，取最大 IoU（T 型零件約 0.85~0.92）。"""
    x, y, w, h = cv2.boundingRect(contour)
    size = 2 * max(w, h) + 4
    cx, cy = _centroid(contour)
    shifted = (contour.astype(np.float64) - [cx, cy] + [size / 2, size / 2]).astype(np.int32)
    m = np.zeros((size, size), np.uint8)
    cv2.drawContours(m, [shifted], -1, 1, -1)
    s = max(16, size // 4)                       # 縮小 4 倍計算，速度快且精度足夠
    small = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
    best = 0.0
    for ang in range(0, 180, step_deg):
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1), (s, s),
                           flags=cv2.INTER_NEAREST)
        f = r[:, ::-1]
        best = max(best, float((r & f).sum()) / max(float((r | f).sum()), 1.0))
    return best


@dataclass
class Review:
    """⑨ 的輸出。"""
    template: DetectedPart | None
    confident: list[DetectedPart] = field(default_factory=list)   # 高信心單片
    doubtful: list[DetectedPart] = field(default_factory=list)    # 有疑慮輪廓
    reasons: dict[int, str] = field(default_factory=dict)         # 有疑慮的原因 {id: 說明}
    symmetry: dict[int, float] = field(default_factory=dict)


@dataclass
class Placement:
    """⑩ 的一次擺放（或無法判定的殘留區域）。"""
    contour: np.ndarray
    score: float
    resolved: bool          # True = 樣板擺放成功；False = 無法判定的殘留
    note: str = ""


# ============================== ⑨ 輪廓信心評分 ==============================
@register_step
class ReviewContours(PipelineStep):
    key = "review_contours"
    title = "輪廓信心評分（選樣板）"
    description = ("評估每個輪廓是否為「完整單片」：鏡像對稱度、實心度、面積與形狀是否和樣板一致。"
                   "對稱度最高者當樣板；與樣板一致者為高信心單片（綠），其餘標為有疑慮（黃）。")
    requires = ("parts",)
    provides = ("review",)
    param_specs = (
        ParamSpec("min_symmetry", "樣板最低對稱度", "float", 0.80, 0, 1, 0.01, help="對稱度低於此值的輪廓不能當樣板"),
        ParamSpec("solidity_min", "實心度下限", "float", 0.55, 0, 1, 0.01),
        ParamSpec("solidity_max", "實心度上限", "float", 0.72, 0, 1, 0.01),
        ParamSpec("area_tolerance", "面積容許差", "float", 0.25, 0.05, 1, 0.05, help="與樣板面積相差在此比例內才算單片"),
        ParamSpec("shape_tolerance", "形狀容許差", "float", 0.30, 0.01, 2, 0.01, help="與樣板的形狀距離（Hu 矩）上限"),
    )

    def run(self, ctx: PipelineContext) -> None:
        p = self.params
        parts: list[DetectedPart] = ctx.data["parts"]
        sym = {c.id: mirror_symmetry(c.contour) for c in parts}

        # 選樣板：實心度在 T 型範圍內、對稱度最高
        candidates = [c for c in parts
                      if p["solidity_min"] <= c.solidity <= p["solidity_max"] and sym[c.id] >= p["min_symmetry"]]
        template = max(candidates, key=lambda c: sym[c.id], default=None)
        review = Review(template=template, symmetry=sym)
        if template is None:
            review.doubtful = list(parts)
            review.reasons = {c.id: "找不到可當樣板的完整單片" for c in parts}
            ctx.data["review"] = review
            return

        for c in parts:
            ratio = c.area / template.area
            dist = 0.0 if c is template else cv2.matchShapes(c.contour, template.contour, cv2.CONTOURS_MATCH_I1, 0)
            if c is template or (abs(ratio - 1) <= p["area_tolerance"] and dist <= p["shape_tolerance"]
                                 and p["solidity_min"] <= c.solidity <= p["solidity_max"]):
                review.confident.append(c)
            else:
                review.doubtful.append(c)
                if ratio > 1 + p["area_tolerance"]:
                    review.reasons[c.id] = f"面積為樣板的 {ratio:.1f} 倍，疑似多片相連"
                elif ratio < 1 - p["area_tolerance"]:
                    review.reasons[c.id] = f"面積僅樣板的 {ratio:.1f} 倍，疑似碎片"
                else:
                    review.reasons[c.id] = "形狀與樣板不一致"
        ctx.data["review"] = review

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        r: Review = ctx.data["review"]
        out = ctx.data["image"].copy()
        for c in r.confident:
            cv2.drawContours(out, [c.contour], -1, GREEN, 3)
        for c in r.doubtful:
            cv2.drawContours(out, [c.contour], -1, YELLOW, 3)
        if r.template is not None:
            cv2.drawContours(out, [r.template.contour], -1, BLUE, 6)
        return out


# ============================== ⑩ 樣板比對 ==============================
@register_step
class TemplatePlacement(PipelineStep):
    key = "template_placement"
    title = "樣板比對（只處理有疑慮輪廓）"
    description = ("在每組有疑慮的輪廓上，把樣板旋轉＋平移找出最吻合的位置逐片擺放："
                   "評分 = 樣板覆蓋疑慮輪廓的比例 + 樣板外框貼合影像邊緣的程度；"
                   "擺放處不可與已確認的零件重疊（零件不會重疊）。")
    requires = ("review", "edge_strength", "image")
    provides = ("placements",)
    param_specs = (
        ParamSpec("angle_step", "旋轉角度間隔", "int", 3, 1, 15, 1, help="越小越準、越慢"),
        ParamSpec("search_scale", "搜尋縮放比例", "float", 0.25, 0.1, 1.0, 0.05, help="先縮小影像搜尋以加速"),
        ParamSpec("edge_weight", "邊緣貼合權重", "float", 0.5, 0, 2, 0.1),
        ParamSpec("max_overlap", "最大重疊比例", "float", 0.08, 0, 0.5, 0.01, help="與已確認零件重疊超過此比例就不能擺放"),
        ParamSpec("min_cover", "最低覆蓋率", "float", 0.75, 0.3, 1, 0.05,
                  help="擺放的樣板需覆蓋到「剩餘疑慮區域」的比例（以兩者中較小者為分母）"),
        ParamSpec("min_score", "最低擺放分數", "float", 0.90, 0, 2, 0.05, help="分數 = 覆蓋率 + 邊緣貼合權重 × 邊緣貼合度"),
        ParamSpec("stop_ratio", "停止剩餘比例", "float", 0.40, 0.1, 1, 0.05,
                  help="疑慮區域剩餘面積小於「樣板面積 × 此值」就停止擺放"),
    )

    def run(self, ctx: PipelineContext) -> None:
        review: Review = ctx.data["review"]
        placements: list[Placement] = []
        ctx.data["placements"] = placements
        if review.template is None or not review.doubtful:
            # 沒有樣板：有疑慮輪廓全部列為無法判定
            for c in review.doubtful:
                placements.append(Placement(c.contour, 0.0, False, review.reasons.get(c.id, "")))
            return

        p = self.params
        h, w = ctx.data["image"].shape[:2]
        s = float(p["search_scale"])
        sw, sh = max(1, round(w * s)), max(1, round(h * s))
        edge = cv2.resize(ctx.data["edge_strength"], (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

        # 已確認的零件（高信心單片）= 不可擺放的區域
        occupied = np.zeros((h, w), np.uint8)
        for c in review.confident:
            cv2.drawContours(occupied, [c.contour], -1, 1, -1)

        tpl = _TemplateBank(review.template.contour, s, int(p["angle_step"]))

        # 相鄰的有疑慮輪廓合成一組一起處理（碎片常常是同一片零件被切開）
        for group in _group_adjacent(review.doubtful, (h, w)):
            target = np.zeros((h, w), np.uint8)
            for c in group:
                cv2.drawContours(target, [c.contour], -1, 1, -1)
            target &= 1 - occupied
            reason = "；".join(review.reasons.get(c.id, "") for c in group)
            placed_any = False
            while True:
                remain = int(target.sum())
                if remain < tpl.area * p["stop_ratio"]:
                    break
                best = tpl.search(target, occupied, edge, s, p)
                if best is None:
                    break
                contour, score = best
                pm = _mask_of(contour, (h, w))
                placements.append(Placement(contour, score, True))
                occupied |= pm
                target &= 1 - pm
                placed_any = True
            # 剩下夠大的區域仍無法用樣板解釋 → 列為無法判定
            leftover = int(target.sum())
            if leftover >= tpl.area * p["stop_ratio"]:
                cnts, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) >= tpl.area * p["stop_ratio"]:
                        note = ("樣板擺放後仍有無法解釋的區域" if placed_any else f"樣板無法擺放（{reason}）")
                        placements.append(Placement(c, 0.0, False, note))

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = ctx.data["image"].copy()
        review: Review = ctx.data["review"]
        for c in review.confident:
            cv2.drawContours(out, [c.contour], -1, GREEN, 2)
        for pl in ctx.data["placements"]:
            cv2.drawContours(out, [pl.contour], -1, MAGENTA if pl.resolved else YELLOW, 4)
        return out


class _TemplateBank:
    """預先產生樣板在各角度下的遮罩與外框（縮小尺寸），供 matchTemplate 搜尋。"""

    def __init__(self, contour: np.ndarray, scale: float, angle_step: int) -> None:
        self.contour = contour.astype(np.float64)
        self.cx, self.cy = _centroid(contour)
        self.area = float(cv2.contourArea(contour))
        pts = (self.contour[:, 0, :] - [self.cx, self.cy]) * scale      # 以重心為原點、縮小
        self.radius = int(np.ceil(np.sqrt((pts ** 2).sum(1)).max())) + 2
        size = 2 * self.radius + 1
        self.items = []   # (角度, 遮罩, 外框, 遮罩像素數, 外框像素數)
        for ang in range(0, 360, angle_step):
            rad = np.deg2rad(ang)
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            q = (pts @ rot.T + self.radius).round().astype(np.int32).reshape(-1, 1, 2)
            m = np.zeros((size, size), np.uint8)
            cv2.drawContours(m, [q], -1, 1, -1)
            o = np.zeros((size, size), np.uint8)
            cv2.drawContours(o, [q], -1, 1, 1)
            self.items.append((ang, m.astype(np.float32), o.astype(np.float32), float(m.sum()), float(o.sum())))

    def contour_at(self, ang: float, x: float, y: float) -> np.ndarray:
        """樣板旋轉 ang 度、重心移到 (x, y)（原圖座標）時的輪廓。"""
        rad = np.deg2rad(ang)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts = (self.contour[:, 0, :] - [self.cx, self.cy]) @ rot.T + [x, y]
        return pts.round().astype(np.int32).reshape(-1, 1, 2)

    def search(self, target: np.ndarray, occupied: np.ndarray, edge: np.ndarray, s: float, p: dict):
        """在 target 附近搜尋最佳擺放；回傳 (輪廓, 分數)，找不到合格位置回傳 None。"""
        sh, sw = edge.shape
        t_small = cv2.resize(target, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        o_small = cv2.resize(occupied, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        ys, xs = np.nonzero(t_small > 0.5)
        if xs.size == 0:
            return None
        r = self.radius
        # 只在疑慮區域外框 + 樣板半徑的範圍內搜尋（加速）；邊界補 0 讓樣板可部分超出影像
        x0, x1 = xs.min() - r, xs.max() + r + 1
        y0, y1 = ys.min() - r, ys.max() + r + 1
        pad = 2 * r + 2

        def crop(img):
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            return padded[y0 + pad - r:y1 + pad + r, x0 + pad - r:x1 + pad + r]

        tc, oc, ec = crop(t_small), crop(o_small), crop(edge)
        remain = float(t_small.sum())
        best = None
        for ang, m, o, m_sum, o_sum in self.items:
            cover = cv2.matchTemplate(tc, m, cv2.TM_CCORR)           # 樣板落在疑慮區域內的像素數
            overlap = cv2.matchTemplate(oc, m, cv2.TM_CCORR) / m_sum  # 與已確認零件重疊比例
            fit = cv2.matchTemplate(ec, o, cv2.TM_CCORR) / o_sum      # 樣板外框上的平均邊緣強度
            # 覆蓋率以「樣板」與「剩餘區域」較小者為分母：相連區域看樣板是否落在區域內，碎片看碎片是否被樣板蓋住
            cover_ratio = np.minimum(cover / min(m_sum, remain), 1.0)
            score = cover_ratio + p["edge_weight"] * fit
            valid = (overlap <= p["max_overlap"]) & (cover_ratio >= p["min_cover"])
            score = np.where(valid, score, -1.0)
            _, mx, _, loc = cv2.minMaxLoc(score)
            if mx > 0 and (best is None or mx > best[0]):
                best = (mx, ang, loc)
        if best is None or best[0] < p["min_score"]:
            return None
        score, ang, (lx, ly) = best
        # matchTemplate 位置 → 樣板中心（縮小座標）→ 原圖座標
        cx_small = lx + x0 - r + r
        cy_small = ly + y0 - r + r
        return self.contour_at(ang, cx_small / s, cy_small / s), float(score)


def _group_adjacent(parts: list[DetectedPart], shape: tuple[int, int], gap: int = 15) -> list[list[DetectedPart]]:
    """把彼此距離在 gap 像素內的輪廓分成同一組。"""
    if not parts:
        return []
    lab = np.zeros(shape, np.int32)
    for i, c in enumerate(parts, start=1):
        cv2.drawContours(lab, [c.contour], -1, i, -1)
    grown = cv2.dilate((lab > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    _, comp = cv2.connectedComponents(grown)
    groups: dict[int, list[DetectedPart]] = {}
    for i, c in enumerate(parts, start=1):
        cx, cy = c.contour[0, 0]
        groups.setdefault(int(comp[cy, cx]), []).append(c)
    return list(groups.values())


# ============================== ⑪ 結果合併 ==============================
@register_step
class FinalizeParts(PipelineStep):
    key = "finalize_parts"
    title = "結果合併與計數"
    description = "最終零件 = 高信心單片 + 樣板擺放成功的零件；無法判定的區域以黃色列出、不計入總數，請人工確認。"
    requires = ("review", "placements", "image")
    provides = ("parts",)

    def run(self, ctx: PipelineContext) -> None:
        review: Review = ctx.data["review"]
        out: list[DetectedPart] = []
        for c in review.confident:
            out.append(_make_part(c.contour, "segment", review.symmetry.get(c.id, 1.0), 1, ""))
        for pl in ctx.data["placements"]:
            if pl.resolved:
                out.append(_make_part(pl.contour, "template", min(1.0, pl.score), 1, ""))
            else:
                out.append(_make_part(pl.contour, "unresolved", 0.0, 0, pl.note + "（未計入總數）"))
        # 依位置排序（由上而下、由左而右）後編號
        out.sort(key=lambda d: (d.bbox[1] // 100, d.bbox[0]))
        for i, d in enumerate(out, start=1):
            d.id = i
        ctx.data["parts"] = out

    def visualize(self, ctx: PipelineContext) -> np.ndarray:
        out = ctx.data["image"].copy()
        for d in ctx.data["parts"]:
            color = YELLOW if d.suspicious else (MAGENTA if d.source == "template" else (0, 0, 255))
            cv2.drawContours(out, [d.contour], -1, color, 3)
        return out


def _make_part(contour: np.ndarray, source: str, confidence: float, count: int, note: str) -> DetectedPart:
    area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    return DetectedPart(
        id=0, contour=contour, bbox=tuple(int(v) for v in cv2.boundingRect(contour)), area=area,
        centroid=_centroid(contour), solidity=area / hull if hull else 0.0,
        estimated_count=count, source=source, confidence=float(confidence), note=note,
    )
