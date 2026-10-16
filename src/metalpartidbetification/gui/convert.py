"""numpy 影像 ↔ Qt 影像轉換。"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_pixmap(image: np.ndarray) -> QPixmap:
    """把 OpenCV 影像（灰階或 BGR）轉成 QPixmap。"""
    img = np.ascontiguousarray(image)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())   # copy：避免 numpy 記憶體被釋放後畫面錯亂
