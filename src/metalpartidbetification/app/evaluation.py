"""評估用：以「模擬參數」產生標準答案，檢查配對是否正確。

只有 app 層與 shift_test 會使用；配對演算法（matching）本身拿不到模擬參數。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core import LEVEL_GREEN, MatchResult, PipelineResult
from ..imaging import FlipSimParams, flip_point


@dataclass
class MirrorEvaluation:
    true_axis: str
    axis_correct: bool
    correct: int                    # 綠色配對中，配對正確的組數
    checked: int                    # 綠色配對中，有標準答案可對照的組數
    wrong_pairs: list[int] = field(default_factory=list)   # 綠色但配錯的配對編號（高信心但其實配錯）
    suspicious: int = 0             # 非綠色的配對數（不列入正確率）
    true_shift: tuple[int, int] = (0, 0)
    shift_error: float = 0.0        # 推估位移與實際位移的距離（像素）
    pair_truth: dict[int, bool | None] = field(default_factory=dict)   # 每組配對：True 正確／False 錯誤／None 無法對照

    @property
    def green_wrong(self) -> int:
        """最關鍵的指標：高信心但其實配錯的組數，必須為 0。"""
        return len(self.wrong_pairs)


def evaluate_match(res_a: PipelineResult, res_b: PipelineResult, match: MatchResult,
                   params: FlipSimParams, size: tuple[int, int], tol: float = 30.0) -> MirrorEvaluation:
    """A 每片零件中心點依模擬參數換算到 B，距離 tol 像素內最近的 B 零件即為正解。"""
    by_b = {p.id: p for p in res_b.parts if p.estimated_count > 0}
    truth: dict[int, int] = {}
    for pa in res_a.parts:
        if pa.estimated_count <= 0:
            continue
        x, y = flip_point(pa.centroid[0], pa.centroid[1], size, params)
        best = min(by_b.values(), key=lambda pb: (pb.centroid[0] - x) ** 2 + (pb.centroid[1] - y) ** 2, default=None)
        if best is not None and ((best.centroid[0] - x) ** 2 + (best.centroid[1] - y) ** 2) ** 0.5 <= tol:
            truth[pa.id] = best.id

    pair_truth = {pr.pair_id: (truth[pr.a_id] == pr.b_id) if pr.a_id in truth else None for pr in match.pairs}
    greens = [pr for pr in match.pairs if pr.level == LEVEL_GREEN and pair_truth[pr.pair_id] is not None]
    wrong = [pr.pair_id for pr in greens if not pair_truth[pr.pair_id]]
    rel = match.relation
    err = ((rel.dx - params.dx) ** 2 + (rel.dy - params.dy) ** 2) ** 0.5
    return MirrorEvaluation(params.axis, rel.axis == params.axis, len(greens) - len(wrong), len(greens), wrong,
                            len(match.suspicious_pairs), (params.dx, params.dy), err, pair_truth)
