"""鏡像位移測試的背景執行緒：逐一執行情境，每完成一個就回報；可中途停止。"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..shift_test import Scenario, ShiftTestRunner


class ShiftTestWorker(QThread):
    original_done = Signal(object)        # 原圖辨識結果（PipelineResult）
    scenario_done = Signal(int, object)   # (情境索引, ScenarioResult)
    failed = Signal(str)

    def __init__(self, runner: ShiftTestRunner, scenarios: list[Scenario], parent=None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.scenarios = scenarios
        self._stop = False

    def stop(self) -> None:
        self._stop = True                 # 目前這個情境跑完後停止

    def run(self) -> None:
        try:
            self.original_done.emit(self.runner.run_original())
            self.runner.run_all(self.scenarios, on_result=lambda k, r: self.scenario_done.emit(k, r),
                                should_stop=lambda: self._stop)
        except Exception as exc:  # 錯誤訊息交給 GUI 顯示
            self.failed.emit(str(exc))
