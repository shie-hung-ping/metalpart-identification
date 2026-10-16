"""逐片判定：每組配對判為 綠（高信心）／黃（無法確定）／紅（疑似可配對到多個零件），
沒配到的零件判為 灰（無法對應），並附上原因。只使用輪廓、位置與系統推估出的關係。
"""
from __future__ import annotations

import numpy as np

from ..core import (
    LEVEL_GREEN, LEVEL_RED, LEVEL_YELLOW, DetectedPart, FlipRelation, JudgeThresholds, MatchPair, UnmatchedPart,
)
from .geometry import centroid, mirror_contour


def judge_pairs(pairs: list[MatchPair], ious: np.ndarray, idx_a: dict[int, int], idx_b: dict[int, int],
                th: JudgeThresholds) -> None:
    """就地設定每組配對的 level / reasons / candidates。ious 的列 = A、行 = B（以 idx_a / idx_b 對照編號）。"""
    b_ids = {v: k for k, v in idx_b.items()}
    a_ids = {v: k for k, v in idx_a.items()}
    for pr in pairs:
        i, j = idx_a[pr.a_id], idx_b[pr.b_id]
        best = float(ious[i, j])
        floor = max(th.candidate_floor, best - th.ambiguity_margin)
        # 多重候選：這片 A 也和其他 B 零件重疊得差不多；或這片 B 也被其他 A 零件搶
        cand = [(b_ids[jj], float(ious[i, jj])) for jj in np.argsort(-ious[i]) if ious[i, jj] >= floor]
        rival = [(a_ids[ii], float(ious[ii, j])) for ii in np.argsort(-ious[:, j]) if ii != i and ious[ii, j] >= floor]

        # 高信心的三項條件（紅色時也一併列出沒過的項目，方便判讀）
        reasons = []
        if pr.iou < th.min_iou:
            reasons.append(f"重疊度 {pr.iou:.2f} 低於 {th.min_iou:.2f}")
        if pr.residual > th.max_residual:
            reasons.append(f"位置殘差 {pr.residual:.0f} px 超過 {th.max_residual:.0f} px")
        if pr.area_ratio > th.max_area_ratio:
            reasons.append(f"兩邊輪廓大小不一致：面積相差 {pr.area_ratio:.1f} 倍")

        if len(cand) >= 2 or rival:                       # 多重候選優先於其他判定
            pr.level = LEVEL_RED
            pr.candidates = cand
            pr.rival_a = rival
            multi = []
            if len(cand) >= 2:
                multi.append("可配對到多個零件：" + "、".join(f"鏡像 #{b}（{v:.2f}）" for b, v in cand))
            if rival:
                multi.append("此鏡像零件也可能屬於：" + "、".join(f"原圖 #{a}（{v:.2f}）" for a, v in rival))
            pr.reasons = multi + reasons
        else:
            pr.level = LEVEL_YELLOW if reasons else LEVEL_GREEN
            pr.reasons = reasons


def residual(part_a: DetectedPart, part_b: DetectedPart, rel: FlipRelation, size: tuple[int, int]) -> float:
    """A 零件中心點依推估關係換算到 B 後，和 B 零件中心點的距離。"""
    moved = mirror_contour(part_a.contour, rel.axis, size, rel.dx, rel.dy)
    return float(np.hypot(*(centroid(moved) - np.array(part_b.centroid))))


def unmatched_reasons(un_a: list[DetectedPart], un_b: list[DetectedPart], rel: FlipRelation,
                      size: tuple[int, int], margin: int = 3) -> list[UnmatchedPart]:
    """灰色（無法對應）的原因：依推估關係換算後落在畫面外 → 移出畫面；否則 → 辨識結果不一致。"""
    w, h = size
    out: list[UnmatchedPart] = []
    for p in un_a:
        moved = mirror_contour(p.contour, rel.axis, size, rel.dx, rel.dy).reshape(-1, 2)
        outside = (moved[:, 0] < margin) | (moved[:, 0] > w - 1 - margin) | (moved[:, 1] < margin) | (moved[:, 1] > h - 1 - margin)
        reason = "移出畫面（翻面位移後推估位置超出照片）" if outside.any() else "辨識結果不一致（鏡像圖中該位置沒有辨識到零件）"
        out.append(UnmatchedPart("A", p.id, reason))
    for p in un_b:
        pts = p.contour.reshape(-1, 2)
        edge = (pts[:, 0] < margin).any() or (pts[:, 0] > w - 1 - margin).any() or \
               (pts[:, 1] < margin).any() or (pts[:, 1] > h - 1 - margin).any()
        reason = "位於畫面邊緣，可能是位移後才部分進入畫面" if edge else "辨識結果不一致（原圖中該位置沒有辨識到零件）"
        out.append(UnmatchedPart("B", p.id, reason))
    return out
