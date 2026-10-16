"""主視窗：最上方分頁標籤切換「辨識｜鏡像位移測試」，兩頁各自保留狀態；狀態列共用。"""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from ..app import DetectionController
from .recognition_page import RecognitionPage
from .shift_test_page import ShiftTestPage


class MainWindow(QMainWindow):
    def __init__(self, controller: DetectionController | None = None) -> None:
        super().__init__()
        self.setWindowTitle("金屬零件計數（AOI）")
        self.resize(1600, 950)

        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        self.recognition = RecognitionPage(controller)
        self.shift_test = ShiftTestPage(self.recognition)
        self.pages.addTab(self.recognition, "辨識")
        self.pages.addTab(self.shift_test, "鏡像位移測試")
        self.setCentralWidget(self.pages)

        for page in (self.recognition, self.shift_test):
            page.status_message.connect(self.statusBar().showMessage)
        # 辨識頁換圖／演算法／參數／ROI → 測試頁同步（更新資訊、清除舊結果）；測試期間鎖定辨識頁
        self.recognition.context_changed.connect(self.shift_test.on_context_changed)
        self.shift_test.running_changed.connect(self.recognition.set_locked)
        self.pages.currentChanged.connect(lambda i: self.shift_test.refresh_info() if i == 1 else None)
        self.statusBar().showMessage("請先選擇圖片資料夾")
