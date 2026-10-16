"""檔案讀寫、結果匯出、繪圖、Controller 測試。"""
import numpy as np

from metalpartidbetification.app import DetectionController
from metalpartidbetification.fileio import list_images, read_image, write_image
from metalpartidbetification.rendering import draw_all, draw_single


def test_unicode_path_roundtrip(tmp_path):
    folder = tmp_path / "中文資料夾"
    folder.mkdir()
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    write_image(folder / "圖片.png", img)
    (folder / "說明.txt").write_text("x", encoding="utf-8")
    files = list_images(folder)
    assert [f.name for f in files] == ["圖片.png"]          # 非圖片檔被排除
    assert np.array_equal(read_image(files[0]), img)


def test_controller_run_and_export(tmp_path, scene_separate):
    img, n = scene_separate
    write_image(tmp_path / "a.png", img)
    ctrl = DetectionController()
    ctrl.load_image(ctrl.open_folder(tmp_path)[0])
    result = ctrl.run()
    assert result.count == n

    out = tmp_path / "out"
    files = ctrl.export(out)
    assert files.binary.exists() and files.result.exists() and files.csv.exists()
    rows = files.csv.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("編號") and rows[-1] == f"總數,{n}"


def test_draw_single_dims_other_area(scene_separate):
    from conftest import synthetic_pipeline
    img, _ = scene_separate
    result = synthetic_pipeline().run(img)
    target = result.parts[0]
    single = draw_single(img, result.parts, target.id)
    full = draw_all(img, result.parts)
    assert single.shape == full.shape == img.shape
    # 角落（背景）在單選模式下應該變暗
    assert single[5, 5].mean() < img[5, 5].mean() * 0.5
    # 選中零件的重心保持原亮度
    cx, cy = map(int, target.centroid)
    assert abs(int(single[cy, cx].mean()) - int(img[cy, cx].mean())) < 40
