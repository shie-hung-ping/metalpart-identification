"""結果輸出：二值化圖、辨識結果圖、零件清單 CSV。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import PipelineResult
from .images import write_image


@dataclass
class ExportedFiles:
    binary: Path | None
    result: Path
    csv: Path


def export_result(result: PipelineResult, result_image: np.ndarray, out_dir: str | Path, stem: str) -> ExportedFiles:
    """把一張圖的辨識結果存到使用者選擇的資料夾，檔名以原圖檔名為前綴。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    binary_path = None
    if result.binary is not None:
        binary_path = out / f"{stem}_binary.png"
        write_image(binary_path, result.binary)

    result_path = out / f"{stem}_result.png"
    write_image(result_path, result_image)

    csv_path = out / f"{stem}_parts.csv"
    # utf-8-sig：讓 Excel 正確顯示中文欄名
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["編號", "面積(px)", "外框x", "外框y", "外框寬", "外框高", "重心x", "重心y", "實心度",
                    "計入片數", "來源", "信心度", "備註"])
        for p in result.parts:
            x, y, bw, bh = p.bbox
            w.writerow([p.id, round(p.area), x, y, bw, bh, round(p.centroid[0], 1), round(p.centroid[1], 1),
                        round(p.solidity, 3), p.estimated_count, p.source, round(p.confidence, 3), p.note])
        w.writerow([])
        w.writerow(["演算法", result.algorithm])
        w.writerow(["耗時(秒)", round(result.elapsed, 3)])
        w.writerow(["總數", result.count])
    return ExportedFiles(binary_path, result_path, csv_path)
