"""可選用的演算法組清單（GUI 下拉選單依此產生）。

每組演算法是獨立的子套件（light / dark），彼此不共用任何影像演算法程式碼；
這裡只負責「列出」與「建立」各組的 Pipeline。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core import Pipeline


@dataclass(frozen=True)
class AlgorithmInfo:
    key: str                          # 內部識別名稱
    name: str                         # 顯示名稱
    description: str                  # 給使用者看的說明
    build: Callable[[], Pipeline]     # 建立預設參數的 Pipeline


def _build_light() -> Pipeline:
    from .light import build_default_pipeline   # 延遲匯入：只在選用時載入
    return build_default_pipeline()


def _build_dark() -> Pipeline:
    from .dark import build_default_pipeline
    return build_default_pipeline()


ALGORITHMS: dict[str, AlgorithmInfo] = {
    "light": AlgorithmInfo("light", "淺色背景", "零件與背景亮度接近：以局部對比（邊緣）分割，再以自我樣板比對處理有疑慮輪廓", _build_light),
    "dark": AlgorithmInfo("dark", "深色背景", "零件明顯比背景亮：以亮度二值化 + 接縫切割分割，再以自我樣板比對處理有疑慮輪廓", _build_dark),
}
DEFAULT_ALGORITHM = "light"
