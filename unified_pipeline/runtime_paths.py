"""
统一流水线路径锚点。

本目录：ZK_Models/unified_pipeline（代码 + config）
ZK_MODELS_ROOT：上级 ZK_Models（ONNX 权重 pangu/fengwu/fuxi 等仍放于此）
GRAPH_CAST_ROOT：graphcast 示例工程根（conf/、graphcast_model/、chdir 目标）
"""
from __future__ import annotations

from pathlib import Path

UNIFIED_PIPELINE_ROOT = Path(__file__).resolve().parent
ZK_MODELS_ROOT = UNIFIED_PIPELINE_ROOT.parent
GRAPH_CAST_ROOT = ZK_MODELS_ROOT.parent
