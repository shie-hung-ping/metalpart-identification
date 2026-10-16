"""深色背景演算法：亮度二值化 + 接縫切割 + 分水嶺分割，再以自我樣板比對處理有疑慮輪廓。

匯入各模組時，其中的步驟會自動註冊到 core.registry（步驟名稱皆以 dark_ 開頭，避免與淺色版衝突）。
規則：本套件不可引用 algorithms.light（兩組演算法完全獨立）。
"""
from . import mask, preprocess, split, template_fit  # noqa: F401  (匯入即註冊)
from .presets import DEFAULT_ORDER, build_default_pipeline

__all__ = ["DEFAULT_ORDER", "build_default_pipeline"]
