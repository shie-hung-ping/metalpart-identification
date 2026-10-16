"""Pipeline 引擎：依序執行步驟、檢查資料相依、記錄中間結果、處理 ROI 裁切與座標還原。"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .models import PipelineContext, PipelineResult, Roi, Snapshot
from .step import PipelineStep


class PipelineError(RuntimeError):
    """Pipeline 設定錯誤或某步驟執行失敗。"""


class Pipeline:
    def __init__(self, steps: list[PipelineStep], name: str = "") -> None:
        self.steps = steps
        self.name = name          # 演算法名稱（例如「淺色背景」），會記錄在結果中

    def enabled_steps(self) -> list[PipelineStep]:
        return [s for s in self.steps if s.enabled]

    def validate(self) -> None:
        """檢查每個啟用步驟所需的資料，都已由前面的步驟提供（例如停用了關鍵步驟時提早報錯）。"""
        available = {"image"}
        for step in self.enabled_steps():
            missing = [k for k in step.requires if k not in available]
            if missing:
                raise PipelineError(f"「{step.title}」缺少輸入資料 {missing}，請確認前面的步驟是否被停用")
            available.update(step.provides)
        if "parts" not in available:
            raise PipelineError("沒有任何啟用的步驟會產出零件清單")

    def run(self, image: np.ndarray, roi: Roi | None = None) -> PipelineResult:
        """執行辨識，並只回報重心落在 ROI 內的零件。

        為什麼不是「裁切 ROI 再處理」：Otsu 閾值、亮度正規化等統計會隨裁切範圍改變，
        造成同一張圖框不同 ROI 得到不同結果；改成整張圖處理、ROI 只做篩選，結果才穩定，
        且零件碰到 ROI 邊框也不會被誤判為背景。
        """
        self.validate()
        h, w = image.shape[:2]
        x, y, rw, rh = _clip_roi(roi, w, h)

        ctx = PipelineContext(data={"image": image.copy()})
        timings: list[tuple[str, float]] = []
        for step in self.enabled_steps():
            t0 = time.perf_counter()
            try:
                step.run(ctx)
            except Exception as exc:  # 包裝成帶有步驟名稱的錯誤，方便 GUI 顯示
                raise PipelineError(f"「{step.title}」執行失敗：{exc}") from exc
            timings.append((step.title, time.perf_counter() - t0))   # 只計演算法時間，不含中間結果圖
            vis = step.visualize(ctx)
            if vis is not None:
                ctx.snapshots.append(Snapshot(step.key, step.title, vis))

        # 二值化結果：ROI 外補 0
        binary = None
        if (b := ctx.data.get("binary")) is not None:
            binary = np.zeros((h, w), np.uint8)
            binary[y:y + rh, x:x + rw] = b[y:y + rh, x:x + rw]

        # 只保留重心在 ROI 內的零件，並重新編號
        inside = [p for p in ctx.data.get("parts", [])
                  if x <= p.centroid[0] < x + rw and y <= p.centroid[1] < y + rh]
        parts = [replace(p, id=i) for i, p in enumerate(inside, start=1)]
        # 可疑零件的警告在重新編號後才產生，編號才會和清單一致
        warnings = list(ctx.warnings) + [
            f"#{p.id} {p.note or f'以面積推估為 {p.estimated_count} 片'}，請人工確認"
            for p in parts if p.suspicious
        ]
        return PipelineResult(image, (x, y, rw, rh), binary, parts, ctx.snapshots, warnings, timings, self.name)


def _clip_roi(roi: Roi | None, w: int, h: int) -> Roi:
    """把 ROI 限制在影像範圍內；沒給或太小就用整張圖。"""
    if roi is None:
        return 0, 0, w, h
    x, y, rw, rh = roi
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + rw)), min(h, int(y + rh))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, w, h
    return x0, y0, x1 - x0, y1 - y0
