"""演算法層：各組影像演算法（light = 淺色背景、dark = 深色背景），彼此完全獨立。"""
from .registry import ALGORITHMS, DEFAULT_ALGORITHM, AlgorithmInfo

__all__ = ["ALGORITHMS", "DEFAULT_ALGORITHM", "AlgorithmInfo"]
