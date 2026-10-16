"""深色背景演算法的執行順序（GUI 控制選單依此順序編號顯示）。"""
from __future__ import annotations

from ...core import Pipeline, create_step

DEFAULT_ORDER = (
    "dark_grayscale",       # 前處理
    "dark_gaussian",        # 前處理
    "dark_threshold",       # ① 亮度二值化
    "dark_morphology",      # ② 形態學（去灰塵、補孔）
    "dark_clear_border",    # ③ 排除邊界區域
    "dark_seam_cut",        # ④ 接縫切割
    "dark_watershed",       # ⑤ 距離轉換 + 分水嶺
    "dark_blobs",           # ⑥ Blob 分析（候選輪廓）
    "dark_review",          # ⑦ 輪廓信心評分（選樣板）
    "dark_placement",       # ⑧ 樣板比對
    "dark_finalize",        # ⑨ 結果合併與計數
)


def build_default_pipeline() -> Pipeline:
    """建立預設參數的深色背景 Pipeline。"""
    return Pipeline([create_step(key) for key in DEFAULT_ORDER], name="深色背景")
