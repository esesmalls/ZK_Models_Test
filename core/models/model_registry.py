"""
模型注册表：读取 config/models.yaml，按名称返回已加载的 WeatherModel 实例。

用法：
    registry = build_registry(cfg_path, device="auto")
    model = registry.get("pangu")           # 已 load()
    enabled = registry.list_enabled()       # ["pangu", "fengwu", ...]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base_model import WeatherModel


def _get_model_classes() -> Dict[str, type]:
    """延迟导入所有模型类（避免顶层导入 onnxruntime/torch 造成环境问题）。"""
    from .pangu_model import PanguModel
    from .fengwu_model import FengWuModel
    from .fuxi_model import FuXiModel
    from .graphcast_model import GraphCastModel
    return {
        "pangu": PanguModel,
        "fengwu": FengWuModel,
        "fuxi": FuXiModel,
        "graphcast": GraphCastModel,
    }


# 可由外部注册扩展的映射（顶层，但不含具体实现）
_MODEL_CLASSES: Dict[str, type] = {}

_ZK_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_ROOT = _ZK_ROOT.parent


def _expand_vars(s: str) -> str:
    """展开 ${ZK_ROOT} 和 ${GRAPH_ROOT} 占位符，以及普通环境变量。"""
    s = s.replace("${ZK_ROOT}", str(_ZK_ROOT))
    s = s.replace("${GRAPH_ROOT}", str(_GRAPH_ROOT))
    s = os.path.expandvars(s)
    return s


def _expand_cfg(obj: Any) -> Any:
    """递归展开 dict/list/str 中的路径占位符。"""
    if isinstance(obj, dict):
        return {k: _expand_cfg(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_cfg(v) for v in obj]
    if isinstance(obj, str):
        return _expand_vars(obj)
    return obj


class ModelRegistry:
    """已加载模型的注册表。"""

    def __init__(self):
        self._models: Dict[str, WeatherModel] = {}
        self._cfgs: Dict[str, Dict] = {}

    def register(self, name: str, model_cls: type) -> None:
        """注册自定义模型类。"""
        _MODEL_CLASSES[name] = model_cls

    def load_from_config(
        self,
        config_path: Path,
        device: Any = "auto",
        only: Optional[List[str]] = None,
        skip_disabled: bool = True,
    ) -> None:
        """
        从 YAML 配置文件加载所有（或指定）模型。

        Args:
            config_path:    config/models.yaml 路径
            device:         推理设备（auto | cpu | cuda | dcu）
            only:           仅加载指定模型名称列表（None=全部）
            skip_disabled:  跳过 enabled: false 的条目
        """
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        models_cfg: Dict[str, Dict] = raw.get("models", {})

        # 延迟加载具体模型类（此时才 import onnxruntime/torch）
        builtin_classes = _get_model_classes()
        all_classes = {**builtin_classes, **_MODEL_CLASSES}

        for name, cfg in models_cfg.items():
            if only is not None and name not in only:
                continue
            if skip_disabled and not cfg.get("enabled", True):
                continue
            if name not in all_classes:
                print(f"[ModelRegistry] 未知模型 '{name}'，跳过", flush=True)
                continue

            cfg_expanded = _expand_cfg(cfg)
            cls = all_classes[name]
            instance = cls()
            try:
                instance.load(cfg_expanded, device=device)
                self._models[name] = instance
                self._cfgs[name] = cfg_expanded
                print(f"[ModelRegistry] 已加载: {name}", flush=True)
            except Exception as e:
                print(f"[ModelRegistry] 加载 '{name}' 失败: {e}", flush=True)

    def get(self, name: str) -> WeatherModel:
        if name not in self._models:
            raise KeyError(f"模型 '{name}' 未加载，可用: {list(self._models)}")
        return self._models[name]

    def list_enabled(self) -> List[str]:
        return list(self._models.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __repr__(self) -> str:
        return f"ModelRegistry({list(self._models.keys())})"


def build_registry(
    config_path: Optional[Path] = None,
    device: Any = "auto",
    only: Optional[List[str]] = None,
) -> ModelRegistry:
    """
    便捷函数：读取配置并返回已加载的 ModelRegistry。

    config_path 默认为 ZK_Models/config/models.yaml。
    """
    if config_path is None:
        config_path = _ZK_ROOT / "config" / "models.yaml"
    reg = ModelRegistry()
    reg.load_from_config(Path(config_path), device=device, only=only)
    return reg
