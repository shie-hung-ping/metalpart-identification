"""背景執行緒：在背景跑辨識（及自動鏡像比對），避免 GUI 畫面卡住。"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..app import DetectionController


class DetectionWorker(QThread):
    succeeded = Signal(object, object)   # (PipelineResult, MirrorRun 或 None)
    failed = Signal(str)

    def __init__(self, controller: DetectionController, mirror_axis: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.mirror_axis = mirror_axis   # None = 不做鏡像比對

    def run(self) -> None:
        try:
            result = self.controller.run()
            mirror = self.controller.run_mirror(self.mirror_axis) if self.mirror_axis else None
            self.succeeded.emit(result, mirror)
        except Exception as exc:  # 錯誤訊息交給 GUI 顯示
            self.failed.emit(str(exc))
