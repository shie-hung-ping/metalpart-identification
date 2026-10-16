"""淺色背景演算法：以局部對比（邊緣）分割 + 自我樣板比對。

匯入各模組時，其中的步驟會自動註冊到 core.registry。
規則：本套件不可引用 algorithms.dark（兩組演算法完全獨立）。
"""
from . import blob, edges, morphology, preprocess, regions, segmentation, template_match  # noqa: F401  (匯入即註冊)
from .presets import DEFAULT_ORDER, build_default_pipeline

__all__ = ["DEFAULT_ORDER", "build_default_pipeline"]
