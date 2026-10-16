"""控制選單：依執行順序顯示每個影像技術步驟。

- 前處理步驟收合在一個群組內（不編號）
- 其餘辨識步驟以 ①②③… 編號並用箭頭串接，讓使用者清楚看到技術的先後順序
- 每個步驟可啟用/停用、調整參數、查看該步驟的中間結果
參數輸入元件依步驟的 ParamSpec 自動產生，新增插件時不需修改這個檔案。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from ..core import ParamSpec, Pipeline, PipelineStep, Stage

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


class StepCard(QFrame):
    """單一步驟的卡片：標題、啟用勾選、說明、參數、查看結果按鈕。"""

    changed = Signal()
    view_requested = Signal(str)

    def __init__(self, step: PipelineStep, number: str, parent=None) -> None:
        super().__init__(parent)
        self.step = step
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)

        head = QHBoxLayout()
        self.enable_box = QCheckBox(f"{number} {step.title}" if number else step.title)
        self.enable_box.setChecked(step.enabled)
        self.enable_box.setStyleSheet("font-weight: bold;")
        self.enable_box.toggled.connect(self._on_enabled)
        head.addWidget(self.enable_box, 1)
        view_btn = QPushButton("查看")
        view_btn.setToolTip("在「步驟檢視」分頁顯示這一步的結果")
        view_btn.setFixedWidth(48)
        view_btn.clicked.connect(lambda: self.view_requested.emit(step.key))
        head.addWidget(view_btn)
        lay.addLayout(head)

        desc = QLabel(step.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(desc)

        if step.param_specs:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            for spec in step.param_specs:
                w = self._make_editor(spec)
                w.setToolTip(spec.help)
                form.addRow(spec.label, w)
            lay.addLayout(form)

    def _on_enabled(self, on: bool) -> None:
        self.step.enabled = on
        self.changed.emit()

    def _make_editor(self, spec: ParamSpec) -> QWidget:
        """依參數型別建立輸入元件，值改變時直接寫回 step.params。"""
        value = self.step.params[spec.key]

        def setter(v) -> None:
            self.step.params[spec.key] = v
            self.changed.emit()

        if spec.kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.minimum or 0), int(spec.maximum or 1_000_000))
            w.setSingleStep(int(spec.step or 1))
            w.setValue(int(value))
            w.setKeyboardTracking(False)        # 輸入完成才觸發，避免每打一個字就重跑
            w.valueChanged.connect(setter)
        elif spec.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(2)
            w.setRange(float(spec.minimum or 0), float(spec.maximum or 1e6))
            w.setSingleStep(float(spec.step or 0.1))
            w.setValue(float(value))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(setter)
        elif spec.kind == "choice":
            w = QComboBox()
            w.addItems(list(spec.choices))
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(setter)
        else:  # bool
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(setter)
        return w


class PipelinePanel(QWidget):
    params_changed = Signal()
    view_requested = Signal(str)

    def __init__(self, pipeline: Pipeline, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        pre = [s for s in pipeline.steps if s.stage == Stage.PREPROCESS]
        main = [s for s in pipeline.steps if s.stage != Stage.PREPROCESS]

        # 前處理群組：可展開/收合
        if pre:
            toggle = QToolButton()
            toggle.setText("前處理（" + " → ".join(s.title for s in pre) + "）")
            toggle.setCheckable(True)
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setArrowType(Qt.ArrowType.RightArrow)
            toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
            box = QWidget()
            box_lay = QVBoxLayout(box)
            box_lay.setContentsMargins(12, 0, 0, 0)
            for s in pre:
                box_lay.addWidget(self._card(s, ""))
            box.setVisible(False)

            def on_toggle(on: bool) -> None:
                box.setVisible(on)
                toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

            toggle.toggled.connect(on_toggle)
            lay.addWidget(toggle)
            lay.addWidget(box)
            lay.addWidget(self._arrow())

        # 主要辨識步驟：依序編號並以箭頭串接
        for i, s in enumerate(main):
            lay.addWidget(self._card(s, CIRCLED[i] if i < len(CIRCLED) else f"({i + 1})"))
            if i < len(main) - 1:
                lay.addWidget(self._arrow())
        lay.addStretch(1)

    def _card(self, step: PipelineStep, number: str) -> StepCard:
        card = StepCard(step, number)
        card.changed.connect(self.params_changed)
        card.view_requested.connect(self.view_requested)
        return card

    @staticmethod
    def _arrow() -> QLabel:
        a = QLabel("▼")
        a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a.setStyleSheet("color: #3a8ee6;")
        return a
