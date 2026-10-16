"""核心層：資料模型、步驟介面、插件註冊與 Pipeline 引擎（不依賴 GUI）。"""
from .models import (
    FLIP_AXES, FLIP_AXIS_NAMES, LEVEL_GRAY, LEVEL_GREEN, LEVEL_NAMES, LEVEL_RED, LEVEL_YELLOW, NO_RELATION,
    ROI_LOST, ROI_ORIGINAL, ROI_PARTIAL, DetectedPart, FlipRelation, JudgeThresholds, MatchPair, MatchResult,
    PipelineContext, PipelineResult, Roi, RoiWarning, Snapshot, UnmatchedPart,
)
from .pipeline import Pipeline, PipelineError
from .registry import available_steps, create_step, register_step
from .step import ParamSpec, PipelineStep, Stage

__all__ = [
    "DetectedPart", "PipelineContext", "PipelineResult", "Roi", "Snapshot",
    "FLIP_AXES", "FLIP_AXIS_NAMES", "NO_RELATION", "FlipRelation", "MatchPair", "MatchResult",
    "JudgeThresholds", "LEVEL_GREEN", "LEVEL_YELLOW", "LEVEL_RED", "LEVEL_GRAY", "LEVEL_NAMES",
    "UnmatchedPart", "RoiWarning", "ROI_PARTIAL", "ROI_LOST", "ROI_ORIGINAL",
    "Pipeline", "PipelineError",
    "available_steps", "create_step", "register_step",
    "ParamSpec", "PipelineStep", "Stage",
]
