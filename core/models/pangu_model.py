"""
Pangu-Weather ONNX 模型封装。

支持 6h/24h 阶梯调度：
  - lead % 24 == 0 → 24h ONNX
  - 其余 → 6h ONNX
每次 step() 前进 6h（使用适当 ONNX）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel
from ..data.channel_mapper import blob_to_pangu_onnx, pangu_onnx_to_blob

_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from infer_cepri_onnx import create_session, pangu_one_step, pick_providers  # noqa: E402


class PanguModel(WeatherModel):
    MODEL_NAME = "pangu"

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._loaded = False
        self._step_h = 6

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        paths = cfg.get("paths", {})
        providers = pick_providers(str(device) if not isinstance(device, str) else device)
        for key, p_str in paths.items():
            p = Path(p_str)
            if p.is_file():
                self._sessions[key] = create_session(p, providers)
        if "6h" not in self._sessions:
            raise FileNotFoundError("Pangu: 找不到 pangu_weather_6.onnx，请检查 config/models.yaml 路径")
        self._loaded = True

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Any = None,
    ) -> ModelState:
        p, s = blob_to_pangu_onnx(init_blob)
        return ModelState(
            data={"p": p, "s": s},
            blob=init_blob,
            lead=0,
        )

    def step(self, state: ModelState) -> ModelState:
        next_lead = state.lead + self._step_h
        # 选择 ONNX：next_lead 能被 24 整除 → 24h 模型
        use_24 = (next_lead % 24 == 0) and ("24h" in self._sessions)
        sess = self._sessions["24h"] if use_24 else self._sessions["6h"]

        p_new, s_new = pangu_one_step(sess, state.data["p"], state.data["s"])
        blob_new = pangu_onnx_to_blob(p_new, s_new)
        blob_new["lat"] = state.blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        blob_new["lon"] = state.blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))
        return ModelState(
            data={"p": p_new, "s": s_new},
            blob=blob_new,
            lead=next_lead,
        )

    def get_surface_var_names(self) -> List[str]:
        return ["msl", "u10", "v10", "t2m"]

    def get_pressure_var_names(self) -> List[str]:
        return ["z", "q", "t", "u", "v"]

    def get_step_hours(self) -> int:
        return self._step_h
