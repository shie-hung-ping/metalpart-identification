"""零件清單：顯示總數與每個零件的資訊；點選某一項 → 在結果圖上單獨突顯該零件。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from ..core import PipelineResult


class PartsPanel(QWidget):
    part_selected = Signal(object)    # 零件編號；None = 顯示全部

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.count_label = QLabel("零件數：—")
        self.count_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        lay.addWidget(self.count_label)

        # 辨識耗時：滑鼠停在上面會列出每個步驟的耗時
        self.time_label = QLabel("辨識耗時：—")
        self.time_label.setStyleSheet("font-size: 14px; color: #3a8ee6;")
        lay.addWidget(self.time_label)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_current)
        lay.addWidget(self.list, 1)

        show_all = QPushButton("顯示全部")
        show_all.clicked.connect(self.clear_selection)
        lay.addWidget(show_all)

        self.warn_label = QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #d08000;")
        lay.addWidget(self.warn_label)

    def set_result(self, result: PipelineResult | None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        if result is None:
            self.count_label.setText("零件數：—")
            self.time_label.setText("辨識耗時：—")
            self.time_label.setToolTip("")
            self.warn_label.clear()
        else:
            self.count_label.setText(f"零件數：{result.count}")
            self.time_label.setText(f"辨識耗時：{result.elapsed:.2f} 秒（{result.algorithm}）")
            self.time_label.setToolTip("各步驟耗時：\n" + "\n".join(f"{t}　{s:.3f} 秒" for t, s in result.timings))
            source_text = {"segment": "分割", "template": "樣板比對", "unresolved": "無法判定"}
            for p in result.parts:
                text = (f"#{p.id}　{source_text.get(p.source, p.source)}　信心 {p.confidence:.2f}"
                        f"　面積 {p.area:,.0f} px")
                if p.suspicious:
                    text += "　⚠ " + (p.note or f"推估 {p.estimated_count} 片")
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, p.id)
                if p.suspicious:
                    item.setForeground(QColor(208, 128, 0))
                self.list.addItem(item)
            self.warn_label.setText("\n".join(result.warnings))
        self.list.blockSignals(False)

    def clear_selection(self) -> None:
        self.list.setCurrentRow(-1)
        self.part_selected.emit(None)

    def _on_current(self, item: QListWidgetItem | None, _prev) -> None:
        self.part_selected.emit(None if item is None else item.data(Qt.ItemDataRole.UserRole))
