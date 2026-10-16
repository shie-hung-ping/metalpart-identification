"""插件註冊表：新增影像技術時只要寫一個類別並加上 @register_step，不需改動其他模組。"""
from __future__ import annotations

from typing import Any

from .step import PipelineStep

_REGISTRY: dict[str, type[PipelineStep]] = {}


def register_step(cls: type[PipelineStep]) -> type[PipelineStep]:
    """類別裝飾器：把步驟登記到註冊表。"""
    if cls.key in _REGISTRY and _REGISTRY[cls.key] is not cls:
        raise ValueError(f"步驟名稱重複：{cls.key}")
    _REGISTRY[cls.key] = cls
    return cls


def create_step(key: str, **params: Any) -> PipelineStep:
    """依名稱建立步驟實例。"""
    try:
        return _REGISTRY[key](**params)
    except KeyError as exc:
        if key not in _REGISTRY:
            raise KeyError(f"未註冊的步驟：{key}") from exc
        raise


def available_steps() -> dict[str, type[PipelineStep]]:
    """列出所有已註冊的步驟。"""
    return dict(_REGISTRY)
