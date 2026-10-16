def main() -> None:
    """程式進入點：啟動 GUI（指令：uv run metalpartidbetification）。"""
    import sys

    from PySide6.QtWidgets import QApplication

    from .gui import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
