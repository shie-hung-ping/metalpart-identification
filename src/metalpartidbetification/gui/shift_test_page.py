"""鏡像位移測試頁：

- 唯讀顯示沿用自辨識頁的設定（測試圖片、演算法、參數、ROI）
- 使用者選擇：鏡像方向（單選，預設左右）、平移方位（可複選，一開始不選）、位移幅度（可多個，一開始空白）
- 判定門檻只顯示；位移模擬方式的說明可展開
- 按「開始測試」才執行；測試期間鎖定辨識頁；可中途停止；結果可匯出
- 辨識頁換圖／演算法／參數／ROI 後同步：清除舊結果（未匯出時提示已清除），顯示新圖片預覽（含 ROI 框）
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from ..core import FLIP_AXES, FLIP_AXIS_NAMES, LEVEL_GRAY, LEVEL_GREEN, LEVEL_RED, LEVEL_YELLOW, JudgeThresholds
from ..imaging import DIRECTIONS
from ..rendering import draw_pairs, draw_roi_box
from ..shift_test import (
    METHOD_NOTE_HTML, ScenarioResult, ShiftTestRunner, build_scenarios, detail_rows, export_report, parse_distances,
    summary_row,
)
from .image_view import ImageView
from .legend import LegendWidget
from .pairs_panel import LEVEL_QCOLORS
from .recognition_page import RecognitionPage
from .shift_test_worker import ShiftTestWorker

SUMMARY_COLS = ["情境", "綠", "黃", "紅", "灰", "ROI!", "LOST", "高信心但配錯", "推估位移", "實際位移",
                "原圖辨識(秒)", "鏡像辨識(秒)", "配對(秒)", "總耗時(秒)"]
DETAIL_COLS = ["配對", "判定", "原圖零件", "鏡像零件", "重疊度", "位置殘差(px)", "面積比", "候選", "原因", "ROI 警告",
               "評估用：實際是否正確"]
GRID_POS = {"up_left": (0, 0), "up": (0, 1), "up_right": (0, 2), "left": (1, 0), "right": (1, 2),
            "down_left": (2, 0), "down": (2, 1), "down_right": (2, 2)}


class ShiftTestPage(QWidget):
    running_changed = Signal(bool)        # 測試開始／結束 → 主視窗據此鎖定／解鎖辨識頁
    status_message = Signal(str)

    def __init__(self, recognition: RecognitionPage, parent=None) -> None:
        super().__init__(parent)
        self.rec = recognition
        self.thresholds = JudgeThresholds()
        self.worker: ShiftTestWorker | None = None
        self.runner: ShiftTestRunner | None = None
        self.results: list[ScenarioResult] = []
        self.scenarios = []
        self._t_start = 0.0
        self._image_name = ""
        self._algo_name = ""
        self._last_dir = ""
        self._exported = False                     # 目前的測試結果是否已匯出（決定清除時要不要提示）

        self._clock = QTimer(self)                 # 更新「已耗時」
        self._clock.setInterval(200)
        self._clock.timeout.connect(self._tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        self.info = QLabel()                       # 沿用設定（唯讀）
        self.info.setWordWrap(True)
        self.info.setStyleSheet("background: #eef4ff; border: 1px solid #b9cdf0; padding: 6px;")
        root.addWidget(self.info)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_settings())
        split.addWidget(self._build_center())
        split.addWidget(self._build_results())
        split.setSizes([330, 800, 520])
        root.addWidget(split, 1)
        self.refresh_info()
        self._show_preview()

    # ================= 版面 =================
    def _build_settings(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 4, 4, 4)

        # 鏡像方向：單選，預設左右
        g_axis = QGroupBox("鏡像方向")
        la = QHBoxLayout(g_axis)
        self.axis_group = QButtonGroup(self)
        for axis in FLIP_AXES:
            rb = QRadioButton(FLIP_AXIS_NAMES[axis].replace("鏡像", ""))
            rb.setProperty("axis", axis)
            rb.setChecked(axis == "h")
            self.axis_group.addButton(rb)
            la.addWidget(rb)
        self.axis_group.buttonToggled.connect(lambda *_: self._update_plan())
        lay.addWidget(g_axis)

        # 平移方位：3×3 方向按鈕，可複選，一開始都不選；中間「全」= 全選／全部取消
        g_dir = QGroupBox("平移方位（可複選）")
        gl = QGridLayout(g_dir)
        self.dir_buttons: dict[str, QToolButton] = {}
        for code, (r, c) in GRID_POS.items():
            b = QToolButton()
            arrow, name, _ = DIRECTIONS[code]
            b.setText(f"{arrow}\n{name}")
            b.setCheckable(True)
            b.setFixedSize(56, 46)
            b.toggled.connect(lambda *_: self._update_plan())
            self.dir_buttons[code] = b
            gl.addWidget(b, r, c)
        all_btn = QToolButton()
        all_btn.setText("全")
        all_btn.setToolTip("全選／全部取消")
        all_btn.setFixedSize(56, 46)
        all_btn.clicked.connect(self._toggle_all_dirs)
        gl.addWidget(all_btn, 1, 1)
        lay.addWidget(g_dir)

        # 位移幅度：可輸入多個，一開始空白
        g_dist = QGroupBox("位移幅度（px）")
        ld = QVBoxLayout(g_dist)
        self.dist_edit = QLineEdit()
        self.dist_edit.setPlaceholderText("例如：30, 50（0～100，逗號分隔）")
        self.dist_edit.textChanged.connect(lambda *_: self._update_plan())
        ld.addWidget(self.dist_edit)
        self.dist_hint = QLabel()
        self.dist_hint.setWordWrap(True)
        self.dist_hint.setStyleSheet("color: #b36b00;")
        ld.addWidget(self.dist_hint)
        lay.addWidget(g_dist)

        # 判定門檻（只顯示）
        g_th = QGroupBox("判定門檻（只顯示）")
        lf = QFormLayout(g_th)
        th = self.thresholds
        for k, v in [("重疊度", f"≥ {th.min_iou:.2f}"), ("位置殘差", f"≤ {th.max_residual:.0f} px"),
                     ("面積比", f"≤ {th.max_area_ratio:.1f} 倍"), ("多重候選差距", f"≤ {th.ambiguity_margin:.2f}"),
                     ("位移搜尋範圍", f"{th.search_range} px")]:
            lf.addRow(k, QLabel(v))
        lay.addWidget(g_th)

        # 位移模擬方式說明（預設收合）
        self.note_btn = QToolButton()
        self.note_btn.setText("位移模擬方式說明")
        self.note_btn.setCheckable(True)
        self.note_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.note_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.note_btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.note = QLabel(METHOD_NOTE_HTML)
        self.note.setTextFormat(Qt.TextFormat.RichText)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("background: #f6f6f6; border: 1px dashed #999; padding: 6px; color: #333;")
        self.note.hide()

        def toggle_note(on: bool) -> None:
            self.note.setVisible(on)
            self.note_btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

        self.note_btn.toggled.connect(toggle_note)
        lay.addWidget(self.note_btn)
        lay.addWidget(self.note)

        # 情境數、預估時間、開始／停止、進度
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        lay.addWidget(self.plan_label)
        row = QHBoxLayout()
        self.start_btn = QPushButton("開始測試")
        self.start_btn.clicked.connect(self.start)
        row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        row.addWidget(self.stop_btn)
        lay.addLayout(row)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.progress_label = QLabel()
        lay.addWidget(self.progress_label)
        lay.addStretch(1)

        self.settings_widget = inner
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _build_center(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.banner = QLabel()                     # 警告提示列（高信心但配錯、ROI 警告）
        self.banner.setWordWrap(True)
        self.banner.hide()
        lay.addWidget(self.banner)
        self.view_label = QLabel("左：原圖　右：鏡像＋位移（點選右側情境總表的一列以檢視）")
        self.view_label.setStyleSheet("color: gray;")
        lay.addWidget(self.view_label)
        self.view = ImageView()
        lay.addWidget(self.view, 1)
        lay.addWidget(LegendWidget(title="圖例（顏色與標記的意義）"))
        return w

    def _build_results(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("情境總表（「高信心但配錯」大於 0 的列以紅底標示，必須為 0）"))
        self.table = QTableWidget(0, len(SUMMARY_COLS))
        self.table.setHorizontalHeaderLabels(SUMMARY_COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_scenario_selected)
        lay.addWidget(self.table, 3)

        lay.addWidget(QLabel("逐片明細（點選一列可在結果圖中突顯該配對與它的候選）"))
        self.detail = QTableWidget(0, len(DETAIL_COLS))
        self.detail.setHorizontalHeaderLabels(DETAIL_COLS)
        self.detail.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.detail.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.detail.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detail.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.detail.itemSelectionChanged.connect(self._on_detail_selected)
        lay.addWidget(self.detail, 3)

        self.timing_label = QLabel()
        self.timing_label.setWordWrap(True)
        self.timing_label.setStyleSheet("color: #3a8ee6;")
        lay.addWidget(self.timing_label)
        self.export_btn = QPushButton("匯出報告…")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setEnabled(False)
        lay.addWidget(self.export_btn)
        return w

    # ================= 沿用設定（唯讀） =================
    def refresh_info(self) -> None:
        ctrl = self.rec.ctrl
        img = ctrl.current_path.name if ctrl.current_path else "（辨識頁尚未選擇圖片）"
        roi = (f"已設定 {ctrl.roi}：不套用到辨識（一律辨識整張圖），只用於檢查零件是否超出 ROI"
               if ctrl.roi else "未設定 ROI，不檢查零件是否超出範圍")
        self.info.setText(
            f"<b>沿用辨識頁的設定（唯讀，測試期間鎖定）</b><br>"
            f"測試圖片：{img}｜演算法：{ctrl.pipeline.name}｜參數：沿用辨識頁控制選單目前的設定<br>ROI：{roi}")
        self._update_plan()

    # ================= 與辨識頁同步 =================
    def on_context_changed(self) -> None:
        """辨識頁換圖、換演算法、改參數或改 ROI：更新唯讀資訊、清除舊結果、顯示新圖片預覽。"""
        if self.worker is not None and self.worker.isRunning():
            return                                 # 測試期間辨識頁已鎖定，理論上不會發生
        self.refresh_info()
        if self.results:
            name, unsaved = self._image_name, not self._exported
            self._clear_results()
            if unsaved:
                msg = f"上一次的測試結果（{name}）尚未匯出，已清除"
                self.banner.setText("ⓘ " + msg)
                self.banner.setStyleSheet("padding: 6px; font-weight: bold; background: #e8eefc; color: #234;")
                self.banner.show()
                self.status_message.emit(msg)
        self._show_preview()

    def _clear_results(self) -> None:
        self.results = []
        self.runner = None
        self.scenarios = []
        self.table.setRowCount(0)
        self.detail.setRowCount(0)
        self.timing_label.clear()
        self.progress.setValue(0)
        self.progress_label.clear()
        self.banner.hide()
        self.export_btn.setEnabled(False)
        self._exported = False

    def _show_preview(self) -> None:
        """尚未測試時：顯示辨識頁目前的圖片與 ROI 框，作為檢視基準。"""
        if self.results:
            return
        ctrl = self.rec.ctrl
        if ctrl.current_image is None:
            self.view.set_image(None)
            self.view_label.setText("請先在辨識頁選擇要測試的圖片")
            return
        self.view.set_image(draw_roi_box(ctrl.current_image, ctrl.roi))
        name = ctrl.current_path.name if ctrl.current_path else ""
        self.view_label.setText(f"預覽：{name}" + ("（藍色虛線框＝ROI）" if ctrl.roi else "（未設定 ROI）")
                                + "｜設定方位與幅度後按「開始測試」")

    # ================= 設定 → 情境 =================
    def _axis(self) -> str:
        return self.axis_group.checkedButton().property("axis")

    def _toggle_all_dirs(self) -> None:
        target = not all(b.isChecked() for b in self.dir_buttons.values())
        for b in self.dir_buttons.values():
            b.setChecked(target)

    def _current_scenarios(self) -> tuple[list, str]:
        """回傳 (情境清單, 錯誤訊息)。"""
        dirs = [c for c, b in self.dir_buttons.items() if b.isChecked()]
        text = self.dist_edit.text().strip()
        if not dirs:
            return [], "請選擇至少一個平移方位"
        if not text:
            return [], "請輸入位移幅度"
        try:
            dists = parse_distances(text)
        except ValueError as exc:
            return [], str(exc)
        return build_scenarios(self._axis(), dirs, dists), ""

    def _update_plan(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        scs, err = self._current_scenarios()
        over = sorted({s.distance for s in scs if s.over_range(self.thresholds.search_range)})
        self.dist_hint.setText(f"超過 {self.thresholds.search_range} px 的幅度（{', '.join(map(str, over))}）已超出搜尋範圍，結果僅供對照"
                               if over else "")
        has_img = self.rec.ctrl.current_image is not None
        if not has_img:
            err = err or ""
            self.plan_label.setText("請先在辨識頁選擇要測試的圖片")
        elif err:
            self.plan_label.setText(err)
        else:
            last = self.rec.ctrl.result.elapsed if self.rec.ctrl.result else None
            est = f"預估約 {last * (len(scs) + 1) + 0.3 * len(scs):.0f} 秒" if last else "預估時間：請先在辨識頁完成一次辨識"
            self.plan_label.setText(f"將執行 {len(scs)} 個情境（{FLIP_AXIS_NAMES[self._axis()]}），{est}")
        self.start_btn.setEnabled(has_img and not err)

    # ================= 執行 =================
    def start(self) -> None:
        scs, err = self._current_scenarios()
        ctrl = self.rec.ctrl
        if err or ctrl.current_image is None:
            QMessageBox.information(self, "鏡像位移測試", err or "請先在辨識頁選擇要測試的圖片")
            return
        self.running_changed.emit(True)            # 鎖定辨識頁（並等待它目前的辨識結束）
        self.scenarios = scs
        self.results = []
        self._image_name = ctrl.current_path.name if ctrl.current_path else "image"
        self._algo_name = ctrl.pipeline.name
        self.runner = ShiftTestRunner(ctrl.pipeline, ctrl.current_image, ctrl.roi, self.thresholds)
        self.table.setRowCount(0)
        self.detail.setRowCount(0)
        self.view.set_image(None)
        self.banner.hide()
        self.timing_label.clear()
        self.progress.setRange(0, len(scs))
        self.progress.setValue(0)
        self._set_running(True)
        self._t_start = time.perf_counter()
        self._clock.start()
        self.worker = ShiftTestWorker(self.runner, scs, self)
        self.worker.original_done.connect(lambda r: self.progress_label.setText(
            f"原圖辨識完成：{r.count} 片（{r.elapsed:.2f} 秒）"))
        self.worker.scenario_done.connect(self._on_scenario_done)
        self.worker.failed.connect(lambda m: QMessageBox.warning(self, "測試失敗", m))
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.status_message.emit(f"鏡像位移測試開始：{self._image_name}，{len(scs)} 個情境")

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.progress_label.setText("停止中…（目前的情境跑完後停止）")

    def _set_running(self, running: bool) -> None:
        self.settings_widget.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.export_btn.setEnabled(not running and bool(self.results))

    def _tick(self) -> None:
        done = len(self.results)
        self.progress_label.setText(f"進度：情境 {done}／{len(self.scenarios)}｜已耗時 {time.perf_counter() - self._t_start:.1f} 秒")

    def _on_scenario_done(self, k: int, r: ScenarioResult) -> None:
        self.results.append(r)
        self.progress.setValue(len(self.results))
        s = summary_row(r, self.thresholds)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for c, key in enumerate(SUMMARY_COLS):
            item = QTableWidgetItem(str(s[key]))
            if key in ("綠", "黃", "紅", "灰") and s[key]:
                item.setForeground({"綠": LEVEL_QCOLORS[LEVEL_GREEN], "黃": LEVEL_QCOLORS[LEVEL_YELLOW],
                                    "紅": LEVEL_QCOLORS[LEVEL_RED], "灰": LEVEL_QCOLORS[LEVEL_GRAY]}[key])
            if s["高信心但配錯"]:
                item.setBackground(QColor(255, 210, 210))
            self.table.setItem(row, c, item)
        if row == 0:
            self.table.selectRow(0)
        self._update_banner()

    def _on_finished(self) -> None:
        self._clock.stop()
        elapsed = time.perf_counter() - self._t_start
        self._set_running(False)
        self._exported = False                     # 新結果尚未匯出
        self.running_changed.emit(False)           # 解鎖辨識頁
        done = len(self.results)
        stopped = done < len(self.scenarios)
        self.progress_label.setText(f"{'已停止' if stopped else '完成'}：{done}／{len(self.scenarios)} 個情境，總耗時 {elapsed:.1f} 秒")
        if self.results:
            det = [dict(r.timings)["鏡像辨識"] for r in self.results]
            mat = [dict(r.timings)["配對"] for r in self.results]
            orig = self.results[0].timings[0][1]
            self.timing_label.setText(
                f"<b>耗時摘要</b>：原圖辨識 {orig:.2f} 秒（一次）｜鏡像辨識 平均 {sum(det) / len(det):.2f} 秒、最長 {max(det):.2f} 秒"
                f"｜配對 平均 {sum(mat) / len(mat):.2f} 秒｜測試總耗時 {elapsed:.1f} 秒")
        self.status_message.emit(f"鏡像位移測試{'已停止' if stopped else '完成'}：{done} 個情境，{elapsed:.1f} 秒")

    def _update_banner(self) -> None:
        msgs, bad = [], False
        wrong = [r for r in self.results if r.evaluation.green_wrong]
        if wrong:
            bad = True
            msgs.append(f"⚠ 有 {len(wrong)} 個情境出現「高信心但其實配錯」（最早：{wrong[0].scenario.label}）")
        roi_hits = [r for r in self.results if any(w.kind in ("partial", "lost") for w in r.match.roi_warnings)]
        if roi_hits:
            n_p = sum(w.kind == "partial" for r in roi_hits for w in r.match.roi_warnings)
            n_l = sum(w.kind == "lost" for r in roi_hits for w in r.match.roi_warnings)
            msgs.append(f"⚠ 有零件超出 ROI：位移可能過大（ROI!）{n_p} 處、可能遺失（LOST）{n_l} 處，"
                        f"共 {len(roi_hits)} 個情境（最早：{roi_hits[0].scenario.label}）")
        self.banner.setVisible(bool(msgs))
        self.banner.setText("\n".join(msgs))
        self.banner.setStyleSheet("padding: 6px; font-weight: bold; " +
                                  ("background: #fdd; color: #900;" if bad else "background: #fff3cd; color: #7a5b00;"))

    # ================= 檢視 =================
    def _selected_result(self) -> ScenarioResult | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.results[rows[0].row()] if rows and rows[0].row() < len(self.results) else None

    def _render(self, r: ScenarioResult, pair_id=None) -> None:
        o = self.runner.original
        self.view.set_image(draw_pairs(o.original, o.parts, r.image_b, r.result_b.parts, r.match, pair_id,
                                       roi_a=self.runner.roi), keep_view=True)
        self.view_label.setText(f"{r.scenario.label}｜左：原圖　右：鏡像＋位移"
                                + ("（超出搜尋範圍，結果僅供對照）" if r.scenario.over_range(self.thresholds.search_range) else ""))

    def _on_scenario_selected(self) -> None:
        r = self._selected_result()
        if r is None:
            return
        self._render(r)
        rows = detail_rows(r)
        self.detail.setRowCount(len(rows))
        for i, d in enumerate(rows):
            for c, key in enumerate(DETAIL_COLS):
                item = QTableWidgetItem(str(d[key]))
                item.setData(Qt.ItemDataRole.UserRole, d["pair_id"])
                if key == "判定":
                    item.setForeground(LEVEL_QCOLORS[d["level"]])
                if key == "評估用：實際是否正確" and d[key] == "錯誤" and d["level"] == LEVEL_GREEN:
                    item.setBackground(QColor(255, 210, 210))
                self.detail.setItem(i, c, item)

    def _on_detail_selected(self) -> None:
        r = self._selected_result()
        items = self.detail.selectedItems()
        if r is None or not items:
            return
        self._render(r, items[0].data(Qt.ItemDataRole.UserRole))

    # ================= 匯出 =================
    def export(self) -> None:
        if not self.results or self.runner is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇報告輸出位置", self._last_dir)
        if not folder:
            return
        self._last_dir = folder
        try:
            path = export_report(folder, self._image_name, self._algo_name, self.runner, self.results)
        except Exception as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        self._exported = True
        self.status_message.emit(f"已匯出鏡像位移測試報告：{path.parent}")
