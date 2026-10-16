"""翻面（鏡像）配對演算法：判斷 A 圖與其翻面圖 B 中，哪些零件是同一片。

規則：本套件只依賴 core；不可引用 imaging（模擬參數）、algorithms（辨識演算法）、shift_test、gui。
"""
from .assign import hungarian, iou_matrix
from .contour_check import contour_fit
from .matcher import match_flipped
from .relation import estimate_relation
from .roi_check import mirror_roi

__all__ = ["contour_fit", "estimate_relation", "hungarian", "iou_matrix", "match_flipped", "mirror_roi"]
