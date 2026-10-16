"""可縮放、可平移的影像檢視元件；可開啟「滑鼠拖曳框選 ROI」功能。

操作：滾輪縮放、右鍵（或中鍵）拖曳平移、雙擊左鍵縮放至視窗大小；
      開啟 ROI 模式時，按住左鍵拖曳框選辨識範圍。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from .convert import to_pixmap


class ImageView(QGraphicsView):
    roi_changed = Signal(object)   # (x, y, w, h)，或 None 表示清除

    def __init__(self, parent=None, roi_enabled: bool = False) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        pen = QPen(QColor(0, 160, 255), 0)            # 寬度 0 = 不受縮放影響的細線
        pen.setStyle(Qt.PenStyle.DashLine)
        self._roi_item = QGraphicsRectItem()
        self._roi_item.setPen(pen)
        self._roi_item.setBrush(QColor(0, 160, 255, 30))
        self._roi_item.setZValue(10)
        self._roi_item.hide()
        self.scene().addItem(self._roi_item)
        self.roi_enabled = roi_enabled
        self._drag_start: QPointF | None = None       # ROI 拖曳起點（影像座標）
        self._pan_last: QPoint | None = None          # 平移拖曳上一點（視窗座標）
        self._has_image = False

    # ---------- 影像 ----------
    def set_image(self, image: np.ndarray | None, keep_view: bool = False) -> None:
        """顯示影像；keep_view=True 時保留目前縮放與位置（同尺寸影像切換時使用）。"""
        if image is None:
            self._pixmap_item.setPixmap(to_pixmap(np.zeros((1, 1), np.uint8)))
            self._has_image = False
            return
        same_size = self._has_image and self._pixmap_item.pixmap().size().toTuple() == (image.shape[1], image.shape[0])
        self._pixmap_item.setPixmap(to_pixmap(image))
        self.scene().setSceneRect(QRectF(0, 0, image.shape[1], image.shape[0]))
        self._has_image = True
        if not (keep_view and same_size):
            self.fit()

    def fit(self) -> None:
        if self._has_image:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ---------- ROI ----------
    def set_roi(self, roi) -> None:
        """由外部設定（或清除）顯示中的 ROI 框，不發出訊號。"""
        if roi is None:
            self._roi_item.hide()
        else:
            self._roi_item.setRect(QRectF(*roi))
            self._roi_item.show()

    def _clamp_to_image(self, p: QPointF) -> QPointF:
        r = self.scene().sceneRect()
        return QPointF(min(max(p.x(), 0), r.width()), min(max(p.y(), 0), r.height()))

    # ---------- 滑鼠事件 ----------
    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton and self.roi_enabled and self._has_image:
            self._drag_start = self._clamp_to_image(self.mapToScene(event.position().toPoint()))
            self._roi_item.setRect(QRectF(self._drag_start, self._drag_start))
            self._roi_item.show()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_last is not None:
            delta = event.position().toPoint() - self._pan_last
            self._pan_last = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        elif self._drag_start is not None:
            cur = self._clamp_to_image(self.mapToScene(event.position().toPoint()))
            self._roi_item.setRect(QRectF(self._drag_start, cur).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_last is not None and event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_last = None
            self.unsetCursor()
        elif self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            r = self._roi_item.rect()
            if r.width() < 16 or r.height() < 16:   # 太小視為誤觸，取消 ROI
                self._roi_item.hide()
                self.roi_changed.emit(None)
            else:
                self.roi_changed.emit((int(r.x()), int(r.y()), int(r.width()), int(r.height())))
        else:
            super().mouseReleaseEvent(event)
