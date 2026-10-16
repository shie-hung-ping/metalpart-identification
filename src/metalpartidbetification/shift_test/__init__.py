"""鏡像位移測試核心（不依賴 GUI）：情境、執行、報告。"""
from .notes import LEGEND_ROWS, METHOD_NOTE_HTML
from .report import detail_rows, export_report, summary_row
from .runner import ScenarioResult, ShiftTestRunner
from .scenarios import MAX_DISTANCE, Scenario, build_scenarios, parse_distances

__all__ = ["LEGEND_ROWS", "METHOD_NOTE_HTML", "MAX_DISTANCE", "Scenario", "ScenarioResult", "ShiftTestRunner",
           "build_scenarios", "detail_rows", "export_report", "parse_distances", "summary_row"]
