"""鏡像位移測試的情境：鏡像方向 × 平移方位 × 位移幅度（每個情境只有一個方位、一個幅度）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..core import FLIP_AXIS_NAMES
from ..imaging import DIRECTIONS, FlipSimParams, shift_vector

MAX_DISTANCE = 100        # 幅度輸入上限（像素）


@dataclass(frozen=True)
class Scenario:
    axis: str             # 鏡像方向 "h" / "v" / "hv"
    direction: str        # 平移方位代碼（見 imaging.DIRECTIONS）
    distance: int         # 位移幅度（直線距離，像素）

    @property
    def params(self) -> FlipSimParams:
        dx, dy = shift_vector(self.direction, self.distance)
        return FlipSimParams(self.axis, dx, dy)

    @property
    def label(self) -> str:
        arrow, name, _ = DIRECTIONS[self.direction]
        return f"{FLIP_AXIS_NAMES[self.axis]}｜{arrow}{name} {self.distance} px"

    @property
    def short_label(self) -> str:
        return f"{DIRECTIONS[self.direction][0]} {self.distance}"

    def over_range(self, search_range: int) -> bool:
        """幅度超出配對演算法的位移搜尋範圍 → 結果僅供對照。"""
        return self.distance > search_range


def parse_distances(text: str) -> list[int]:
    """解析「30, 50」這類輸入（逗號或空白分隔）；不合法時拋出 ValueError（訊息可直接顯示給使用者）。"""
    items = [t for t in text.replace("，", ",").replace(" ", ",").split(",") if t.strip()]
    if not items:
        raise ValueError("請輸入至少一個位移幅度")
    values = []
    for t in items:
        try:
            v = int(float(t))
        except ValueError:
            raise ValueError(f"「{t}」不是數字") from None
        if not 0 <= v <= MAX_DISTANCE:
            raise ValueError(f"位移幅度需在 0～{MAX_DISTANCE} px 之間（輸入了 {v}）")
        values.append(v)
    return sorted(set(values))


def build_scenarios(axis: str, directions: list[str], distances: list[int]) -> list[Scenario]:
    """依畫面上方位按鈕的順序，組合出所有情境。"""
    order = [d for d in DIRECTIONS if d in directions]
    return [Scenario(axis, d, dist) for d in order for dist in distances]
