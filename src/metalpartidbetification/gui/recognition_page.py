"""辨識頁（原本的主視窗內容，搬移至此，功能不變）：
左側 = 資料夾/圖片清單 + 控制選單；中間 = 影像分頁；右側 = 零件清單 + 鏡像配對清單。

另外提供 set_locked()：鏡像位移測試進行中時，鎖定設定並暫停辨識（仍可查看結果）。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..algorithms import ALGORITHMS
from ..app import DetectionController, MirrorRun
from ..core import FLIP_AXES, FLIP_AXIS_NAMES, PipelineResult
from .image_view import ImageView
from .pairs_panel import PairsPanel
from .parts_panel import PartsPanel
from .pipeline_panel import PipelinePanel
from .worker import DetectionWorker

TAB_ORIGINAL, TAB_BINARY, TAB_RESULT, TAB_STEP, TAB_MIRROR, TAB_PAIRS = range(6)


class RecognitionPage(QWidget):
    status_message = Signal(str)      # 交給主視窗的狀態列顯示
    context_changed = Signal()        # 圖片、演算法、參數或 ROI 改變（測試頁據此同步：更新資訊、清除舊結果）

    def __init__(self, controller: DetectionController | None = None, parent=None) -> None:
        super().__init__(parent)
        self.ctrl = controller or DetectionController()
        self.worker: DetectionWorker | None = None
        self._rerun_pending = False
        self._selected_step: str | None = None
        self._last_export_dir = ""
        self._locked = False

        # 參數變更後延遲 400ms 再重跑，避免連續調整時一直重算
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self.run_detection)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # 鎖定提示列（鏡像位移測試進行中才顯示）
        self.lock_banner = QLabel("🔒 鏡像位移測試進行中：圖片、資料夾、演算法、參數與辨識功能已鎖定（仍可查看結果）")
        self.lock_banner.setStyleSheet("background: #fff3cd; color: #7a5b00; padding: 6px; font-weight: bold;")
        self.lock_banner.hide()
        outer.addWidget(self.lock_banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_widget = self._build_left()
        splitter.addWidget(self.left_widget)
        splitter.addWidget(self._build_center())
        # 右側：上 = 零件清單、下 = 鏡像配對清單
        right = QSplitter(Qt.Orientation.Vertical)
        self.parts_panel = PartsPanel()
        self.parts_panel.part_selected.connect(self._show_part)
        right.addWidget(self.parts_panel)
        self.pairs_panel = PairsPanel()
        self.pairs_panel.pair_selected.connect(self._show_pair)
        right.addWidget(self.pairs_panel)
        splitter.addWidget(right)
        splitter.setSizes([380, 780, 360])
        outer.addWidget(splitter, 1)

    def _status(self, msg: str) -> None:
        self.status_message.emit(msg)

    # ================= 鎖定（鏡像位移測試期間） =================
    def set_locked(self, locked: bool) -> None:
        """鎖定圖片切換、資料夾、演算法、參數，並暫停辨識；結果分頁仍可查看。"""
        self._locked = locked
        self.left_widget.setEnabled(not locked)
        self.orig_view.roi_enabled = not locked
        self.clear_roi_btn.setEnabled(not locked)
        self.lock_banner.setVisible(locked)
        if locked:
            self._debounce.stop()
            if self.worker is not None and self.worker.isRunning():
                self.worker.wait()                # 等目前的辨識結束，避免和測試同時運算

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ================= 版面 =================
    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        folder_btn = QPushButton("選擇圖片資料夾…")
        folder_btn.clicked.connect(self.choose_folder)
        lay.addWidget(folder_btn)
        self.folder_label = QLabel("（尚未選擇）")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: gray;")
        lay.addWidget(self.folder_label)

        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(160)
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        lay.addWidget(self.image_list)

        row = QHBoxLayout()
        run_btn = QPushButton("執行辨識")
        run_btn.clicked.connect(self.run_detection)
        row.addWidget(run_btn)
        self.auto_box = QCheckBox("調整參數後自動重跑")
        self.auto_box.setChecked(True)
        row.addWidget(self.auto_box)
        lay.addLayout(row)

        # 演算法選擇：淺色背景 / 深色背景（兩組完全獨立，各自保存參數）
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("演算法："))
        self.algo_box = QComboBox()
        for key, info in ALGORITHMS.items():
            self.algo_box.addItem(info.name, key)
            self.algo_box.setItemData(self.algo_box.count() - 1, info.description, Qt.ItemDataRole.ToolTipRole)
        self.algo_box.setCurrentIndex(self.algo_box.findData(self.ctrl.algorithm))
        self.algo_box.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_row.addWidget(self.algo_box, 1)
        lay.addLayout(algo_row)

        # 自動鏡像比對：辨識完成後自動產生鏡像（模擬翻面）→ 辨識鏡像 → 配對
        mirror_row = QHBoxLayout()
        self.mirror_box = QCheckBox("自動鏡像比對")
        self.mirror_box.setChecked(True)
        self.mirror_box.toggled.connect(lambda _: self.run_detection())
        mirror_row.addWidget(self.mirror_box)
        self.mirror_axis_box = QComboBox()
        for axis in FLIP_AXES:
            self.mirror_axis_box.addItem(FLIP_AXIS_NAMES[axis], axis)
        self.mirror_axis_box.setToolTip("模擬翻面時使用的鏡像方向（配對演算法不會得知此設定）")
        self.mirror_axis_box.currentIndexChanged.connect(lambda _: self.run_detection())
        mirror_row.addWidget(self.mirror_axis_box, 1)
        lay.addLayout(mirror_row)

        title = QLabel("控制選單（依執行順序）")
        title.setStyleSheet("font-weight: bold; margin-top: 6px;")
        lay.addWidget(title)
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self._rebuild_pipeline_panel()
        lay.addWidget(self.panel_scroll, 1)
        return w

    def _rebuild_pipeline_panel(self) -> None:
        """依目前選用的演算法重建控制選單。"""
        panel = PipelinePanel(self.ctrl.pipeline)
        panel.params_changed.connect(self._on_params_changed)
        panel.view_requested.connect(self._view_step)
        self.panel_scroll.setWidget(panel)   # 舊的面板會由 QScrollArea 自動釋放

    def _on_algorithm_changed(self, index: int) -> None:
        key = self.algo_box.itemData(index)
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()                # 等目前的辨識結束，避免切換途中讀到另一組 Pipeline
        self.ctrl.set_algorithm(key)
        self._selected_step = None
        self._rebuild_pipeline_panel()
        self._status(f"已切換為「{ALGORITHMS[key].name}」演算法")
        self.context_changed.emit()
        self.run_detection()

    def _build_center(self) -> QWidget:
        self.tabs = QTabWidget()

        # 原圖分頁：可拖曳框選 ROI
        orig = QWidget()
        ol = QVBoxLayout(orig)
        ol.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        hint = QLabel("按住左鍵拖曳框選辨識範圍（ROI：只計數重心在框內的零件）｜滾輪縮放｜右鍵拖曳平移｜雙擊還原")
        hint.setStyleSheet("color: gray;")
        bar.addWidget(hint, 1)
        self.clear_roi_btn = QPushButton("清除 ROI")
        self.clear_roi_btn.clicked.connect(lambda: self._on_roi_changed(None))
        bar.addWidget(self.clear_roi_btn)
        ol.addLayout(bar)
        self.orig_view = ImageView(roi_enabled=True)
        self.orig_view.roi_changed.connect(self._on_roi_changed)
        ol.addWidget(self.orig_view)
        self.tabs.addTab(orig, "原圖 / ROI")

        self.binary_view = ImageView()
        self.tabs.addTab(self.binary_view, "二值化結果")

        res = QWidget()
        rl = QVBoxLayout(res)
        rl.setContentsMargins(0, 0, 0, 0)
        export_bar = QHBoxLayout()
        export_bar.addStretch(1)
        export_btn = QPushButton("匯出結果…")
        export_btn.clicked.connect(self.export_result)
        export_bar.addWidget(export_btn)
        rl.addLayout(export_bar)
        self.result_view = ImageView()
        rl.addWidget(self.result_view)
        self.tabs.addTab(res, "辨識結果")

        step = QWidget()
        sl = QVBoxLayout(step)
        sl.setContentsMargins(0, 0, 0, 0)
        self.step_label = QLabel("在左側控制選單按「查看」以檢視各步驟的中間結果")
        sl.addWidget(self.step_label)
        self.step_view = ImageView()
        sl.addWidget(self.step_view)
        self.tabs.addTab(step, "步驟檢視")

        # 鏡像結果分頁：鏡像圖的辨識結果 + 儲存鏡像圖
        mir = QWidget()
        ml = QVBoxLayout(mir)
        ml.setContentsMargins(0, 0, 0, 0)
        mbar = QHBoxLayout()
        self.mirror_label = QLabel("尚未執行鏡像比對")
        mbar.addWidget(self.mirror_label, 1)
        save_btn = QPushButton("儲存鏡像圖…")
        save_btn.clicked.connect(self.save_mirror_image)
        mbar.addWidget(save_btn)
        ml.addLayout(mbar)
        self.mirror_view = ImageView()
        ml.addWidget(self.mirror_view)
        self.tabs.addTab(mir, "鏡像結果")

        # 配對比較分頁：原圖（左）與鏡像（右）並排，依逐片判定著色（顏色意義見右下圖例）
        pairs = QWidget()
        pl = QVBoxLayout(pairs)
        pl.setContentsMargins(0, 0, 0, 0)
        self.pairs_label = QLabel("左：原圖　右：鏡像　綠＝高信心配對正確、黃＝無法確定、紅＝疑似可配對到多個零件、灰＝無法對應（圖例見右下）")
        self.pairs_label.setStyleSheet("color: gray;")
        pl.addWidget(self.pairs_label)
        self.pairs_view = ImageView()
        pl.addWidget(self.pairs_view)
        self.tabs.addTab(pairs, "配對比較")
        return self.tabs

    # ================= 影像來源 =================
    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾")
        if not folder:
            return
        images = self.ctrl.open_folder(folder)
        self.folder_label.setText(folder)
        self.image_list.clear()
        for p in images:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.image_list.addItem(item)
        if images:
            self.image_list.setCurrentRow(0)
        else:
            self._status("此資料夾沒有支援的圖片")

    def _on_image_selected(self, item: QListWidgetItem | None, _prev) -> None:
        if item is None:
            return
        try:
            img = self.ctrl.load_image(item.data(Qt.ItemDataRole.UserRole))
        except IOError as exc:
            QMessageBox.warning(self, "讀取失敗", str(exc))
            return
        self.orig_view.set_image(img)
        self.orig_view.set_roi(self.ctrl.roi)
        for v in (self.binary_view, self.result_view, self.step_view, self.mirror_view, self.pairs_view):
            v.set_image(None)
        self.parts_panel.set_result(None)
        self.pairs_panel.set_run(None)
        self.mirror_label.setText("尚未執行鏡像比對")
        self.context_changed.emit()
        self.run_detection()

    # ================= 辨識 =================
    def _on_roi_changed(self, roi) -> None:
        if self._locked:
            return
        self.ctrl.set_roi(roi)
        self.orig_view.set_roi(roi)
        self._status(f"ROI：{roi}" if roi else "已清除 ROI，辨識整張圖")
        self.context_changed.emit()
        self.run_detection()

    def _on_params_changed(self) -> None:
        if self._locked:
            return
        self.context_changed.emit()
        if self.auto_box.isChecked():
            self._debounce.start()

    def run_detection(self) -> None:
        if self.ctrl.current_image is None or self._locked:
            return
        if self.worker is not None and self.worker.isRunning():
            self._rerun_pending = True   # 正在算：等這次結束後再跑一次
            return
        self._status("辨識中…")
        axis = self.mirror_axis_box.currentData() if self.mirror_box.isChecked() else None
        self.worker = DetectionWorker(self.ctrl, axis, self)
        self.worker.succeeded.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self) -> None:
        if self._rerun_pending:
            self._rerun_pending = False
            self.run_detection()

    def _on_result(self, result: PipelineResult, mirror: MirrorRun | None) -> None:
        self.binary_view.set_image(result.binary, keep_view=True)
        self.result_view.set_image(self.ctrl.render_all(), keep_view=True)
        self.parts_panel.set_result(result)
        if self._selected_step:
            self._view_step(self._selected_step, switch_tab=False)
        self._show_mirror(mirror, result.count)
        msg = f"完成［{result.algorithm}］：辨識到 {result.count} 個零件（耗時 {result.elapsed:.2f} 秒）"
        if mirror is not None:
            c = mirror.match.level_counts()
            t = dict(mirror.timings)
            msg += (f"｜鏡像 {mirror.result.count} 個：綠 {c['green']}、黃 {c['yellow']}、紅 {c['red']}、灰 {c['gray']}"
                    f"｜原圖辨識 {t['原圖辨識']:.2f} 秒、鏡像辨識 {t['鏡像辨識']:.2f} 秒、總耗時 {mirror.total:.2f} 秒")
        if result.warnings:
            msg += f"（{len(result.warnings)} 則警告）"
        self._status(msg)
        if self.tabs.currentIndex() == TAB_ORIGINAL:
            self.tabs.setCurrentIndex(TAB_RESULT)

    def _on_failed(self, message: str) -> None:
        self._status("辨識失敗")
        QMessageBox.warning(self, "辨識失敗", message)

    # ================= 鏡像比對 =================
    def _show_mirror(self, mirror: MirrorRun | None, count_a: int) -> None:
        self.pairs_panel.set_run(mirror, count_a)
        if mirror is None:
            self.mirror_view.set_image(None)
            self.pairs_view.set_image(None)
            self.mirror_label.setText("未執行鏡像比對（左側「自動鏡像比對」未勾選）")
            return
        self.mirror_view.set_image(self.ctrl.render_mirror(), keep_view=True)
        self.pairs_view.set_image(self.ctrl.render_pairs(), keep_view=True)
        self.mirror_label.setText(
            f"鏡像（{FLIP_AXIS_NAMES[mirror.params.axis]}）：辨識到 {mirror.result.count} 個零件"
            f"（鏡像產生 {mirror.timings[1][1]:.2f} 秒＋鏡像辨識 {mirror.timings[2][1]:.2f} 秒）")

    def _show_pair(self, pair_id) -> None:
        if self.ctrl.mirror is None:
            return
        self.pairs_view.set_image(self.ctrl.render_pairs(pair_id), keep_view=True)
        self.tabs.setCurrentIndex(TAB_PAIRS)

    def save_mirror_image(self) -> None:
        if self.ctrl.mirror is None:
            QMessageBox.information(self, "儲存鏡像圖", "目前沒有鏡像圖（請勾選「自動鏡像比對」並完成辨識）")
            return
        stem = self.ctrl.current_path.stem if self.ctrl.current_path else "image"
        start = str(Path(self._last_export_dir or ".") / f"{stem}_M.png")
        path, _ = QFileDialog.getSaveFileName(self, "儲存鏡像圖", start, "PNG 圖片 (*.png);;JPEG 圖片 (*.jpg);;BMP 圖片 (*.bmp)")
        if not path:
            return
        try:
            self.ctrl.save_mirror_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "儲存失敗", str(exc))
            return
        self._last_export_dir = str(Path(path).parent)
        self._status(f"已儲存鏡像圖：{path}")

    # ================= 檢視 =================
    def _show_part(self, part_id) -> None:
        if self.ctrl.result is None:
            return
        img = self.ctrl.render_all() if part_id is None else self.ctrl.render_single(part_id)
        self.result_view.set_image(img, keep_view=True)
        self.tabs.setCurrentIndex(TAB_RESULT)

    def _view_step(self, step_key: str, switch_tab: bool = True) -> None:
        self._selected_step = step_key
        r = self.ctrl.result
        snap = next((s for s in r.snapshots if s.step_key == step_key), None) if r else None
        if snap is None:
            self.step_label.setText("此步驟目前沒有結果（尚未執行或已停用）")
            self.step_view.set_image(None)
        else:
            self.step_label.setText(f"步驟：{snap.title}（整張圖）")
            self.step_view.set_image(snap.image, keep_view=True)
        if switch_tab:
            self.tabs.setCurrentIndex(TAB_STEP)

    # ================= 匯出 =================
    def export_result(self) -> None:
        if self.ctrl.result is None:
            QMessageBox.information(self, "匯出", "請先執行辨識")
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾", self._last_export_dir)
        if not folder:
            return
        self._last_export_dir = folder
        try:
            files = self.ctrl.export(folder)
        except Exception as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        names = [p.name for p in (files.binary, files.result, files.csv) if p]
        self._status(f"已匯出到 {Path(folder)}：" + "、".join(names))
