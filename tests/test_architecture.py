"""架構規則測試：淺色、深色兩組演算法彼此不可互相引用；演算法切換正常。"""
import ast
from pathlib import Path

import pytest

from metalpartidbetification.algorithms import ALGORITHMS
from metalpartidbetification.app import DetectionController

ALGO_DIR = Path(__file__).resolve().parents[1] / "src" / "metalpartidbetification" / "algorithms"


def _imported_modules(py_file: Path) -> list[str]:
    """列出檔案中所有 import 的模組名稱（含相對匯入的點號）。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names]
    return names


PKG_DIR = ALGO_DIR.parent


@pytest.mark.parametrize("package, forbidden", [
    ("algorithms/light", {"dark", "matching", "imaging"}),
    ("algorithms/dark", {"light", "matching", "imaging"}),
    # 鏡像配對演算法：只依賴 core；不可看到模擬參數（imaging），也不可使用辨識演算法、測試工具、GUI
    ("matching", {"algorithms", "light", "dark", "imaging", "shift_test", "gui"}),
    # 鏡像位移測試核心：不依賴 GUI
    ("shift_test", {"gui", "PySide6"}),
])
def test_algorithm_packages_are_independent(package, forbidden):
    for f in (PKG_DIR / package).glob("*.py"):
        for mod in _imported_modules(f):
            parts = set(mod.replace(".", " ").split())
            assert not parts & forbidden, f"{package}/{f.name} 引用了禁止的套件：{mod}"


def test_step_keys_do_not_collide():
    light = ALGORITHMS["light"].build()
    dark = ALGORITHMS["dark"].build()
    assert not {s.key for s in light.steps} & {s.key for s in dark.steps}


def test_controller_switches_algorithm_and_keeps_params():
    ctrl = DetectionController("light")
    ctrl.pipeline.steps[1].params["diameter"] = 7          # 改淺色組的參數
    ctrl.set_algorithm("dark")
    assert ctrl.pipeline.name == "深色背景"
    ctrl.set_algorithm("light")
    assert ctrl.pipeline.steps[1].params["diameter"] == 7  # 切回來參數仍保留
    with pytest.raises(KeyError):
        ctrl.set_algorithm("no_such")
