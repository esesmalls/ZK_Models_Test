"""
抽象天气模型接口。

所有模型封装必须继承 WeatherModel 并实现：
  - load(cfg, device)         加载权重/会话
  - get_surface_var_names()   返回支持的地表变量名列表
  - get_pressure_var_names()  返回支持的气压层变量名列表
  - step(state, lead_hours)   推进一步，返回新的 ModelState

ModelState 是一个简单容器，封装模型自身的内部状态
（例如 FuXi 需要双帧历史，GraphCast 需要归一化 Tensor），
使流水线层不感知各模型的内部细节。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelState:
    """
    模型推理状态容器。
    data:   模型内部状态（ONNX 输入张量 / Torch Tensor 等），各模型自定义格式。
    blob:   最新步的物理量 blob dict（surface_* + pangu_* keys），流水线用来做评估和出图。
    lead:   当前已推进的预报时长（小时）。
    extra:  模型特有的额外信息（如 FengWu 的 pred69，GraphCast 的 latlon_torch 等）。
    """
    data: Any = None
    blob: Optional[Dict] = None
    lead: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class WeatherModel(ABC):
    """统一天气预报模型接口。"""

    MODEL_NAME: str = "base"

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, cfg: Dict, device: Any) -> None:
        """
        从 cfg（config/models.yaml 中该模型的条目）加载权重/ONNX Session。
        device: torch.device 或字符串（CPU/CUDA/ROCm）。
        """

    @abstractmethod
    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Any = None,
    ) -> ModelState:
        """
        根据初始 blob 构建模型初始状态。
        prev_blob: 前一时刻 blob（双帧模型如 FengWu/FuXi 需要）。
        init_dt:   初始预报时刻（datetime 对象，GraphCast 需要）。
        """

    @abstractmethod
    def step(self, state: ModelState) -> ModelState:
        """
        前进一步推理，返回新的 ModelState（含更新后的 blob 和 lead+step）。
        """

    @abstractmethod
    def get_surface_var_names(self) -> List[str]:
        """返回此模型支持的地表变量名（如 ['u10','v10','t2m','msl']）。"""

    def get_pressure_var_names(self) -> List[str]:
        """返回此模型支持的气压层变量名（如 ['z','q','t','u','v']）。默认空列表。"""
        return []

    def get_step_hours(self) -> int:
        """每步推理的时间步长（小时），默认 6。"""
        return 6

    # ------------------------------------------------------------------
    # 可选覆盖
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """
        释放模型占用的 GPU/CPU 内存（ONNX Session 或 PyTorch Module）。
        默认为空操作；各模型子类应覆盖此方法以实际释放资源。
        在多模型串行推理时，每个模型推理完毕后调用，避免 VRAM OOM。
        """

    def is_loaded(self) -> bool:
        return hasattr(self, "_loaded") and self._loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
