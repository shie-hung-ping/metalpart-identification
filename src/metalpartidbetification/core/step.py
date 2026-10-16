"""步驟（插件）介面約定：每一種影像技術都實作成一個 PipelineStep。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from .models import PipelineContext


class Stage(Enum):
    """步驟所屬階段：GUI 依此把前處理收成群組，其餘步驟依序編號。"""

    PREPROCESS = "preprocess"
    DETECTION = "detection"


@dataclass(frozen=True)
class ParamSpec:
    """參數描述。GUI 依 kind 自動產生對應的輸入元件。

    kind: "int" / "float" / "choice" / "bool"
    """

    key: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""


class PipelineStep(ABC):
    """所有影像步驟的基底類別。

    子類別需定義：
      key          唯一識別名稱（註冊用）
      title        顯示名稱
      description  給使用者看的技術說明
      stage        所屬階段
      requires     執行前 context.data 必須已有的鍵
      provides     執行後會寫入 context.data 的鍵
      param_specs  可調參數
    """

    key: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str] = ""
    stage: ClassVar[Stage] = Stage.DETECTION
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    param_specs: ClassVar[tuple[ParamSpec, ...]] = ()

    def __init__(self, **overrides: Any) -> None:
        # 先填入預設值，再套用呼叫端指定的參數
        self.params: dict[str, Any] = {p.key: p.default for p in self.param_specs}
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise KeyError(f"{self.key} 不認得的參數：{sorted(unknown)}")
        self.params.update(overrides)
        self.enabled: bool = True

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """執行影像處理，讀取 requires、寫入 provides。"""

    def visualize(self, ctx: PipelineContext) -> np.ndarray | None:
        """產生這一步的中間結果圖；預設顯示第一個 provides 的影像。"""
        for k in self.provides:
            value = ctx.data.get(k)
            if isinstance(value, np.ndarray) and value.dtype == np.uint8:
                return value
        return None
