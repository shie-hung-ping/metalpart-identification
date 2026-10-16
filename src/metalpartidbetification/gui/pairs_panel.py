"""配對清單（辨識頁）：鏡像比對的摘要、分段耗時、評估資訊、四種判定的配對清單與顏色圖例；
點選某一組 → 在「配對比較」突顯。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from ..app import MirrorRun
from ..core import FLIP_AXIS_NAMES, LEVEL_GRAY, LEVEL_GREEN, LEVEL_NAMES, LEVEL_RED, LEVEL_YELLOW, NO_RELATION
from .legend import PAIR_LEVEL_ROWS, LegendWidget

# 清單文字顏色（和結果圖的四種判定顏色對應）
LEVEL_QCOLORS = {LEVEL_GREEN: QColor(40, 150, 40), LEVEL_YELLOW: QColor(190, 140, 0),
                 LEVEL_RED: QColor(210, 40, 40), LEVEL_GRAY: QColor(130, 130, 130)}
LEVEL_MARK = {LEVEL_GREEN: "", LEVEL_YELLOW: "?", LEVEL_RED: "!"}


class PairsPanel(QWidget):
    pair_selected = Signal(object)     # 配對編號；None = 顯示全部

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        title = QLabel("鏡像配對")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        lay.addWidget(title)

        self.summary = QLabel("—")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

        # 分段耗時 + 總耗時
        self.timing = QLabel()
        self.timing.setTextFormat(Qt.TextFormat.RichText)
        self.timing.setStyleSheet("color: #3a8ee6; font-size: 13px;")
        lay.addWidget(self.timing)

        # 評估用資訊：以模擬參數計算，配對演算法本身拿不到
        self.evaluation = QLabel()
        self.evaluation.setWordWrap(True)
        self.evaluation.setStyleSheet("background: #f2f2f2; color: #555; border: 1px dashed #999; padding: 4px;")
        lay.addWidget(self.evaluation)

        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_current)
        lay.addWidget(self.list, 1)

        show_all = QPushButton("顯示全部配對")
        show_all.clicked.connect(self.clear_selection)
        lay.addWidget(show_all)

        lay.addWidget(LegendWidget(PAIR_LEVEL_ROWS, "顏色意義"))
        self.set_run(None)

    def set_run(self, run: MirrorRun | None, count_a: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        if run is None:
            self.summary.setText("未執行（勾選「自動鏡像比對」後會在辨識完成時自動執行）")
            self.timing.clear()
            self.evaluation.hide()
        else:
            m, rel = run.match, run.match.relation
            c = m.level_counts()
            self.summary.setText(
                f"原圖 {count_a} 片／鏡像 {run.result.count} 片\n"
                f"綠 {c[LEVEL_GREEN]}　黃 {c[LEVEL_YELLOW]}　紅 {c[LEVEL_RED]}　灰 {c[LEVEL_GRAY]}\n"
                f"推估關係：{FLIP_AXIS_NAMES.get(rel.axis, rel.axis)}（吻合度 {rel.score:.2f}）")
            # 兩次辨識的耗時各佔一行、加粗；其餘分段（鏡像產生、配對）與總耗時列在下方
            t = dict(run.timings)
            self.timing.setText(
                f"<b>原圖辨識：{t['原圖辨識']:.2f} 秒</b><br>"
                f"<b>鏡像辨識：{t['鏡像辨識']:.2f} 秒</b><br>"
                f"鏡像產生 {t['鏡像產生']:.2f} 秒　配對 {t['配對']:.2f} 秒<br>"
                f"總耗時 {run.total:.2f} 秒")
            e = run.evaluation
            verdict = "無法推估" if rel.axis == NO_RELATION else ("推估正確" if e.axis_correct else "推估錯誤")
            self.evaluation.setText(
                f"【評估用】實際鏡像方向：{FLIP_AXIS_NAMES.get(e.true_axis, e.true_axis)}（{verdict}）\n"
                f"高信心配對實際正確 {e.correct}／{e.checked}"
                + (f"，⚠ 高信心但配錯：P{', P'.join(map(str, e.wrong_pairs))}" if e.wrong_pairs else ""))
            self.evaluation.show()
            for pr in m.pairs:
                text = (f"P{pr.pair_id}{LEVEL_MARK[pr.level]}　{LEVEL_NAMES[pr.level]}\n"
                        f"　原圖 #{pr.a_id} ↔ 鏡像 #{pr.b_id}　重疊 {pr.iou:.2f}　殘差 {pr.residual:.0f} px")
                if pr.reasons:
                    text += "\n　" + pr.note
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, pr.pair_id)
                item.setForeground(LEVEL_QCOLORS[pr.level])
                self.list.addItem(item)
            for u in m.unmatched:
                side = "原圖" if u.side == "A" else "鏡像"
                item = QListWidgetItem(f"#{u.part_id}x　{LEVEL_NAMES[LEVEL_GRAY]}（{side}）\n　{u.reason}")
                item.setData(Qt.ItemDataRole.UserRole, None)
                item.setForeground(LEVEL_QCOLORS[LEVEL_GRAY])
                self.list.addItem(item)
        self.list.blockSignals(False)

    def clear_selection(self) -> None:
        self.list.setCurrentRow(-1)
        self.pair_selected.emit(None)

    def _on_current(self, item: QListWidgetItem | None, _prev) -> None:
        self.pair_selected.emit(None if item is None else item.data(Qt.ItemDataRole.UserRole))
