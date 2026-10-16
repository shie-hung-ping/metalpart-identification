"""影像讀寫：用 imdecode / imencode，避免 cv2.imread 在 Windows 讀不到中文路徑。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: str | Path) -> list[Path]:
    """列出資料夾（不含子資料夾）內所有支援格式的圖片，依檔名排序。"""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def read_image(path: str | Path) -> np.ndarray:
    """讀取圖片為 BGR 影像；失敗時拋出 IOError。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise IOError(f"無法讀取圖片：{path}")
    return img


def write_image(path: str | Path, image: np.ndarray) -> None:
    """依副檔名編碼並寫檔（支援中文路徑）。"""
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"無法編碼圖片：{path}")
    buf.tofile(str(path))
