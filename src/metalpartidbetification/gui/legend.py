"""顏色圖例元件（辨識頁與鏡像位移測試頁共用）：說明每種顏色、標記的意義。"""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel

from ..shift_test import LEGEND_ROWS


class LegendWidget(QFrame):
    def __init__(self, rows: list[tuple[str, str, str]] | None = None, title: str = "圖例", parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setVerticalSpacing(2)
        head = QLabel(title)
        head.setStyleSheet("font-weight: bold;")
        grid.addWidget(head, 0, 0, 1, 3)
        for r, (name, desc, color) in enumerate(rows or LEGEND_ROWS, start=1):
            sw = QLabel("■")
            sw.setStyleSheet(f"color: {color}; font-size: 16px;")
            grid.addWidget(sw, r, 0)
            n = QLabel(name)
            n.setStyleSheet("font-weight: bold;")
            grid.addWidget(n, r, 1)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet("color: #555;")
            grid.addWidget(d, r, 2)
        grid.setColumnStretch(2, 1)


# 辨識頁只用到四種配對判定（沒有位移、不檢查 ROI）
PAIR_LEVEL_ROWS = LEGEND_ROWS[:4]
