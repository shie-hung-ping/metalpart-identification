"""翻面配對主流程：① 推估關係（鏡像方向＋位移）→ ② 位置配對 → ③ 輪廓細節驗證
→ ④ 逐片判定（綠／黃／紅／灰）→ ⑤ ROI 警告。

輸入只有：兩份零件清單、影像尺寸、使用者框的 ROI，以及「B 是 A 翻面拍攝」這個已知事實（由圖片編號得知）。
不知道鏡像方向與位移量（模擬參數），也不使用任何表面紋理。
"""
from __future__ import annotations

from ..core import DetectedPart, JudgeThresholds, MatchPair, MatchResult, Roi
from .assign import hungarian, iou_matrix
from .contour_check import contour_fit
from .geometry import mirror_contour
from .judge import judge_pairs, residual, unmatched_reasons
from .relation import estimate_relation
from .roi_check import check_roi

W_IOU, W_FIT = 0.6, 0.4  # 信心度 = 0.6 × 重疊度 + 0.4 × 鏡像貼合度


def match_flipped(parts_a: list[DetectedPart], parts_b: list[DetectedPart], size: tuple[int, int],
                  thresholds: JudgeThresholds | None = None, roi: Roi | None = None) -> MatchResult:
    """size = (寬, 高)。只拿「有計入總數」的零件配對（無法判定的區域不參與）。

    thresholds.search_range = 位移搜尋範圍；0 表示假設沒有位移（辨識頁的自動鏡像比對）。
    roi = 使用者框的範圍，只用於 ROI 警告，不影響配對。
    """
    th = thresholds or JudgeThresholds()
    pa = [p for p in parts_a if p.estimated_count > 0]
    pb = [p for p in parts_b if p.estimated_count > 0]
    relation = estimate_relation(pa, pb, size, th.search_range)

    moved = [mirror_contour(p.contour, relation.axis, size, relation.dx, relation.dy) for p in pa]
    ious = iou_matrix(moved, [p.contour for p in pb])
    chosen = [(i, j) for i, j in hungarian(1.0 - ious) if ious[i, j] >= th.candidate_floor]

    pairs: list[MatchPair] = []
    for i, j in sorted(chosen, key=lambda ij: pa[ij[0]].id):
        fit_m = contour_fit(pa[i].contour, pb[j].contour, relation.axis)
        fit_p = contour_fit(pa[i].contour, pb[j].contour, None)
        conf = W_IOU * float(ious[i, j]) + W_FIT * fit_m
        small, big = sorted((max(pa[i].area, 1.0), max(pb[j].area, 1.0)))
        pairs.append(MatchPair(len(pairs) + 1, pa[i].id, pb[j].id, float(ious[i, j]), fit_m, fit_p, conf,
                               area_ratio=big / small, residual=residual(pa[i], pb[j], relation, size)))

    judge_pairs(pairs, ious, {p.id: k for k, p in enumerate(pa)}, {p.id: k for k, p in enumerate(pb)}, th)

    used_a = {pa[i].id for i, _ in chosen}
    used_b = {pb[j].id for _, j in chosen}
    unmatched = unmatched_reasons([p for p in pa if p.id not in used_a], [p for p in pb if p.id not in used_b],
                                  relation, size)
    result = MatchResult(relation=relation, pairs=pairs, unmatched=unmatched)
    check_roi(pa, pb, result, roi, size)
    return result
