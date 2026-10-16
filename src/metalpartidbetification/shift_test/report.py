"""鏡像位移測試報告：標記結果圖、逐片明細 CSV、情境總表 CSV、摘要 HTML。"""
from __future__ import annotations

import csv
import html
from pathlib import Path

from ..core import LEVEL_GRAY, LEVEL_GREEN, LEVEL_NAMES, LEVEL_RED, LEVEL_YELLOW, ROI_LOST, ROI_PARTIAL, JudgeThresholds
from ..fileio import write_image
from ..rendering import draw_pairs
from .notes import LEGEND_ROWS, METHOD_NOTE_HTML
from .runner import ScenarioResult, ShiftTestRunner


def summary_row(r: ScenarioResult, th: JudgeThresholds) -> dict:
    """情境總表的一列（GUI 與報告共用）。"""
    c = r.match.level_counts()
    t = dict(r.timings)
    rel = r.match.relation
    return {
        "情境": r.scenario.label + ("（超出搜尋範圍，對照）" if r.scenario.over_range(th.search_range) else ""),
        "綠": c[LEVEL_GREEN], "黃": c[LEVEL_YELLOW], "紅": c[LEVEL_RED], "灰": c[LEVEL_GRAY],
        "ROI!": sum(w.kind == ROI_PARTIAL for w in r.match.roi_warnings),
        "LOST": sum(w.kind == ROI_LOST for w in r.match.roi_warnings),
        "高信心但配錯": r.evaluation.green_wrong,
        "推估位移": f"({rel.dx:+d}, {rel.dy:+d})", "實際位移": f"({r.params.dx:+d}, {r.params.dy:+d})",
        "位移誤差(px)": round(r.evaluation.shift_error, 1),
        "原圖辨識(秒)": round(t["原圖辨識"], 3), "鏡像辨識(秒)": round(t["鏡像辨識"], 3),
        "配對(秒)": round(t["配對"], 3), "總耗時(秒)": round(r.total, 3),
    }


def detail_rows(r: ScenarioResult) -> list[dict]:
    """逐片明細（GUI 與報告共用）。"""
    rows = []
    roi_by = {(w.side, w.part_id): w for w in r.match.roi_warnings}
    for pr in r.match.pairs:
        truth = r.evaluation.pair_truth.get(pr.pair_id)
        wa, wb = roi_by.get(("A", pr.a_id)), roi_by.get(("B", pr.b_id))
        rows.append({
            "配對": f"P{pr.pair_id}", "原圖零件": pr.a_id, "鏡像零件": pr.b_id, "判定": LEVEL_NAMES[pr.level],
            "level": pr.level, "pair_id": pr.pair_id,
            "重疊度": round(pr.iou, 2), "位置殘差(px)": round(pr.residual, 1), "面積比": round(pr.area_ratio, 2),
            "候選": "、".join(f"#{b}({v:.2f})" for b, v in pr.candidates) if pr.candidates else "",
            "原因": pr.note, "ROI 警告": "；".join(w.detail for w in (wa, wb) if w),
            "評估用：實際是否正確": {True: "正確", False: "錯誤", None: "無法對照"}[truth],
        })
    for u in r.match.unmatched:
        w = roi_by.get((u.side, u.part_id))
        rows.append({
            "配對": "—", "原圖零件": u.part_id if u.side == "A" else "", "鏡像零件": u.part_id if u.side == "B" else "",
            "判定": LEVEL_NAMES[LEVEL_GRAY], "level": LEVEL_GRAY, "pair_id": None,
            "重疊度": "", "位置殘差(px)": "", "面積比": "", "候選": "", "原因": u.reason,
            "ROI 警告": w.detail if w else "", "評估用：實際是否正確": "",
        })
    return rows


def export_report(out_dir: str | Path, image_name: str, algorithm: str, runner: ShiftTestRunner,
                  results: list[ScenarioResult]) -> Path:
    """輸出到 out_dir/鏡像位移測試_<圖片名>/，回傳 HTML 摘要的路徑。"""
    stem = Path(image_name).stem
    out = Path(out_dir) / f"鏡像位移測試_{stem}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    th = runner.thresholds
    orig = runner.original

    summary = [summary_row(r, th) for r in results]
    with open(out / "情境總表.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["情境"])
        w.writeheader()
        w.writerows(summary)

    with open(out / "逐片明細.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["情境", "配對", "原圖零件", "鏡像零件", "判定", "重疊度", "位置殘差(px)", "面積比", "候選", "原因",
                  "ROI 警告", "評估用：實際是否正確"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            for row in detail_rows(r):
                w.writerow({"情境": r.scenario.label, **row})

    imgs = []
    for k, r in enumerate(results, start=1):
        name = f"{k:02d}_{r.scenario.axis}_{r.scenario.direction}_{r.scenario.distance}.png"
        write_image(out / "images" / name, draw_pairs(orig.original, orig.parts, r.image_b, r.result_b.parts,
                                                     r.match, roi_a=runner.roi, legend=True))
        imgs.append(name)

    head = list(summary[0].keys()) if summary else []
    rows_html = "".join(
        "<tr" + (' class="bad"' if s["高信心但配錯"] else "") + ">" + "".join(f"<td>{html.escape(str(s[h]))}</td>" for h in head)
        + f'<td><a href="images/{img}">結果圖</a></td></tr>' for s, img in zip(summary, imgs))
    legend = "".join(f'<li><span style="color:{c}">■</span> <b>{html.escape(k)}</b>：{html.escape(v)}</li>' for k, v, c in LEGEND_ROWS)
    page = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>鏡像位移測試 {html.escape(stem)}</title>
<style>body{{font-family:sans-serif;margin:24px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
tr.bad td{{background:#fdd}} .note{{background:#f6f6f6;padding:8px 12px;border:1px dashed #999}}</style>
<h1>鏡像位移測試：{html.escape(image_name)}</h1>
<p>演算法：{html.escape(algorithm)}（沿用辨識頁的參數）｜原圖辨識到 {orig.count} 片｜原圖辨識耗時 {orig.elapsed:.2f} 秒</p>
<p>判定門檻：重疊度 ≥ {th.min_iou:.2f}、位置殘差 ≤ {th.max_residual:.0f} px、面積比 ≤ {th.max_area_ratio:.1f} 倍、
多重候選差距 ≤ {th.ambiguity_margin:.2f}、位移搜尋範圍 {th.search_range} px</p>
<h2>圖例</h2><ul>{legend}</ul>
<div class="note">{METHOD_NOTE_HTML}</div>
<h2>情境總表</h2><p>「高信心但配錯」大於 0 的情境以紅底標示（必須為 0）。</p>
<table><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in head)}<th>結果圖</th></tr>{rows_html}</table>
<p>逐片明細請見「逐片明細.csv」。</p></html>"""
    (out / "摘要.html").write_text(page, encoding="utf-8")
    return out / "摘要.html"
