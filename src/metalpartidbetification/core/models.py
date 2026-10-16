"""核心資料模型：在各層之間傳遞的資料結構（不依賴 GUI）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ROI 格式：(x, y, 寬, 高)，以原圖像素座標表示
Roi = tuple[int, int, int, int]


@dataclass
class DetectedPart:
    """一個被辨識到的零件。座標皆為原圖座標。"""

    id: int
    contour: np.ndarray                 # OpenCV 輪廓，形狀 (N, 1, 2)
    bbox: tuple[int, int, int, int]     # (x, y, w, h)
    area: float                         # 面積（像素）
    centroid: tuple[float, float]       # 重心 (x, y)
    solidity: float                     # 實心度 = 面積 / 凸包面積
    estimated_count: int = 1            # 此輪廓計入的片數（>1 = 推估多片相黏；0 = 列出但不計入）
    source: str = "segment"             # 來源："segment" 分割 / "template" 樣板比對 / "unresolved" 無法判定
    confidence: float = 1.0             # 信心度 0~1
    note: str = ""                      # 需要人工確認時的說明

    @property
    def suspicious(self) -> bool:
        """是否需要人工確認（無法判定，或以面積推估為多片）。"""
        return self.source == "unresolved" or self.estimated_count != 1


@dataclass
class Snapshot:
    """某一步驟執行後的中間結果圖，供 GUI 逐步檢視。"""

    step_key: str
    title: str
    image: np.ndarray


@dataclass
class PipelineContext:
    """Pipeline 執行期間在步驟之間流動的資料袋。

    data 的常用鍵值（由各步驟的 requires / provides 宣告）：
      "image"          完整彩色原圖 (BGR)
      "gray"           灰階影像
      "edge_strength"  局部對比 / 邊緣強度圖 (uint8)
      "binary"         二值化結果 (uint8, 0/255)
      "edge_mask"      去雜訊後的邊緣遮罩 (uint8, 0/255)
      "markers"        分水嶺種子標記 (int32)
      "labels"         分水嶺區域標記 (int32，-1 為分界線)
      "region_kind"    區域分類 {label: "part"/"shadow"/"background"}
      "part_labels"    合併後的零件標記圖 (int32，0 為非零件)
      "parts"          list[DetectedPart]（整張圖的所有零件，ROI 篩選由 Pipeline 處理）
      "review"         輪廓信心評分結果（樣板、高信心單片、有疑慮輪廓）
      "placements"     樣板比對的擺放結果
    """

    data: dict[str, Any] = field(default_factory=dict)
    snapshots: list[Snapshot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Pipeline 對外輸出的最終結果（座標皆為原圖座標）。"""

    original: np.ndarray                # 完整原圖 (BGR)
    roi: Roi                            # 實際使用的辨識範圍
    binary: np.ndarray | None           # 與原圖同尺寸的二值化結果（ROI 外為 0）
    parts: list[DetectedPart]
    snapshots: list[Snapshot]
    warnings: list[str]
    timings: list[tuple[str, float]] = field(default_factory=list)   # 每個步驟的耗時（步驟名稱, 秒）
    algorithm: str = ""                                              # 使用的演算法名稱

    @property
    def count(self) -> int:
        """零件總數（可疑相黏的區塊以估計片數計入）。"""
        return sum(p.estimated_count for p in self.parts)

    @property
    def elapsed(self) -> float:
        """辨識總耗時（秒）。"""
        return sum(t for _, t in self.timings)


# ============================== 翻面（鏡像）配對 ==============================
# 鏡像方向代碼："h" = 左右鏡像、"v" = 上下鏡像、"hv" = 左右＋上下
FLIP_AXES = ("h", "v", "hv")
NO_RELATION = "none"                    # 無法推估（任一邊沒有零件）
FLIP_AXIS_NAMES = {"h": "左右鏡像", "v": "上下鏡像", "hv": "左右＋上下鏡像", NO_RELATION: "無法推估（沒有可比較的零件）"}


@dataclass
class FlipRelation:
    """推估的兩張圖關係：B = A 依 axis 鏡像後，再平移 (dx, dy)。"""

    axis: str
    dx: int = 0
    dy: int = 0
    score: float = 0.0                  # 吻合度（翻轉後零件遮罩的 IoU）
    candidates: dict[str, float] = field(default_factory=dict)   # 各方向的吻合度


@dataclass(frozen=True)
class JudgeThresholds:
    """逐片判定的門檻（GUI 只顯示、不可修改）。"""

    min_iou: float = 0.50            # 高信心：翻轉＋位移後的重疊度下限
    max_residual: float = 10.0       # 高信心：中心點位置殘差上限（像素）
    max_area_ratio: float = 1.5      # 高信心：兩片面積比上限
    ambiguity_margin: float = 0.20   # 多重候選：與最佳候選的重疊度差距在此範圍內即算候選
    candidate_floor: float = 0.30    # 重疊度低於此值不算候選（也是可配對的最低重疊度）
    search_range: int = 50           # 位移搜尋範圍（像素）


# 逐片判定等級（顏色）
LEVEL_GREEN = "green"     # 高信心，配對正確
LEVEL_YELLOW = "yellow"   # 無法確定是否配對成功
LEVEL_RED = "red"         # 疑似可配對到多個零件
LEVEL_GRAY = "gray"       # 無法對應
LEVEL_NAMES = {LEVEL_GREEN: "高信心，配對正確", LEVEL_YELLOW: "無法確定是否配對成功",
               LEVEL_RED: "疑似可配對到多個零件", LEVEL_GRAY: "無法對應"}


@dataclass
class MatchPair:
    """一組配對：A 圖的零件 ↔ B 圖的零件（皆為零件編號）。"""

    pair_id: int
    a_id: int
    b_id: int
    iou: float              # A 輪廓翻轉＋位移後與 B 輪廓的重疊度
    fit_mirror: float       # 以中心點對齊後，A 輪廓「鏡像」與 B 輪廓的貼合度（0~1）
    fit_plain: float        # 同上但「不鏡像」，作為對照
    confidence: float
    area_ratio: float = 1.0 # 兩片面積比（大 ÷ 小）
    residual: float = 0.0   # A 中心點換算到 B 後，與 B 零件中心點的距離（像素）
    level: str = LEVEL_GREEN
    reasons: list[str] = field(default_factory=list)              # 非綠色的原因
    candidates: list[tuple[int, float]] = field(default_factory=list)   # 紅色時：所有候選 (B 零件編號, 重疊度)
    rival_a: list[tuple[int, float]] = field(default_factory=list)      # 紅色時：同樣搶這片 B 零件的 A 零件

    @property
    def suspicious(self) -> bool:
        """非高信心的配對（黃、紅）。"""
        return self.level != LEVEL_GREEN

    @property
    def note(self) -> str:
        return "；".join(self.reasons)


@dataclass
class UnmatchedPart:
    """無法對應的零件（灰色）。side = "A"（原圖）或 "B"（鏡像圖）。"""

    side: str
    part_id: int
    reason: str


# ROI 警告種類
ROI_PARTIAL = "partial"          # ROI!：輪廓有一部分在 ROI 外 → 位移可能過大
ROI_LOST = "lost"                # LOST：整片在 ROI 外，或推估位置在 ROI／畫面外而沒配對到 → 可能遺失
ROI_ORIGINAL = "original"        # 原圖一開始就超出 ROI（和位移造成的分開）


@dataclass
class RoiWarning:
    side: str            # "A" / "B"
    part_id: int         # 零件編號（LOST 且 B 圖找不到時為 A 的編號）
    kind: str
    detail: str


@dataclass
class MatchResult:
    relation: FlipRelation
    pairs: list[MatchPair]              # 全部配對（含黃、紅）
    unmatched: list[UnmatchedPart] = field(default_factory=list)
    roi_warnings: list[RoiWarning] = field(default_factory=list)
    roi_b: Roi | None = None            # B 圖上的 ROI（跟著鏡像、不跟著位移）

    @property
    def unmatched_a(self) -> list[int]:
        return [u.part_id for u in self.unmatched if u.side == "A"]

    @property
    def unmatched_b(self) -> list[int]:
        return [u.part_id for u in self.unmatched if u.side == "B"]

    def pairs_at(self, level: str) -> list[MatchPair]:
        return [p for p in self.pairs if p.level == level]

    @property
    def normal_pairs(self) -> list[MatchPair]:
        """高信心配對（綠色）。"""
        return self.pairs_at(LEVEL_GREEN)

    @property
    def suspicious_pairs(self) -> list[MatchPair]:
        """非高信心的配對（黃色＋紅色）。"""
        return [p for p in self.pairs if p.suspicious]

    def level_counts(self) -> dict[str, int]:
        """四種判定的數量（灰色以「無法對應的零件數」計）。"""
        return {LEVEL_GREEN: len(self.pairs_at(LEVEL_GREEN)), LEVEL_YELLOW: len(self.pairs_at(LEVEL_YELLOW)),
                LEVEL_RED: len(self.pairs_at(LEVEL_RED)), LEVEL_GRAY: len(self.unmatched)}
