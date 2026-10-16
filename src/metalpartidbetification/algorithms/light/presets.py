"""預設流程：定義各步驟的執行順序（GUI 控制選單即依此順序編號顯示）。"""
from __future__ import annotations

from ...core import Pipeline, create_step

# 執行順序；前處理在前，之後依序為辨識技術
DEFAULT_ORDER = (
    "grayscale",            # 前處理
    "bilateral",            # 前處理
    "local_contrast",       # ① 局部對比
    "binarize",             # ② 二值化
    "remove_small_edges",   # ③ 形態學去雜訊
    "distance_seeds",       # ④ 距離轉換找種子
    "watershed",            # ⑤ 分水嶺
    "classify_regions",     # ⑥ 區域分類
    "merge_regions",        # ⑦ 區域合併
    "blob_analysis",        # ⑧ Blob 分析（候選輪廓）
    "review_contours",      # ⑨ 輪廓信心評分（選樣板）
    "template_placement",   # ⑩ 樣板比對（只處理有疑慮輪廓）
    "finalize_parts",       # ⑪ 結果合併與計數
)


def build_default_pipeline() -> Pipeline:
    """建立預設參數的 Pipeline（預設值依 1920×1080 淺色背景金屬 T 型片樣本調整）。"""
    return Pipeline([create_step(key) for key in DEFAULT_ORDER], name="淺色背景")
