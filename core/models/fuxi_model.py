"""
FuXi ONNX 模型封装。

使用 short.onnx（默认）或 medium.onnx，双帧 70 通道输入。
FuXi 输出布局自动检测：NTCHW 或 NCTHW。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel
from ..data.channel_mapper import blobs_to_fuxi_2frame, blob_to_fuxi_70ch

_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from infer_cepri_onnx import (  # noqa: E402
    create_session,
    fuxi_prepare_onnx_input,
    fuxi_temb,
    log_ort_session,
    pick_providers,
)


# FuXi 地表通道索引（70ch 帧）
_FUXI_SURFACE_IDX = {"t2m": 65, "u10": 66, "v10": 67, "msl": 68}


class FuXiModel(WeatherModel):
    MODEL_NAME = "fuxi"

    def __init__(self):
        self._sess = None
        self._layout: Optional[str] = None
        self._loaded = False
        self._step_h = 6

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        version = cfg.get("default_version", "short")
        paths = cfg.get("paths", {})
        onnx_path = Path(paths.get(version, paths.get("short", "")))
        if not onnx_path.is_file():
            raise FileNotFoundError(f"FuXi: 找不到 ONNX 文件 {onnx_path}")

        providers = pick_providers(str(device) if not isinstance(device, str) else device)
        self._sess = create_session(onnx_path, providers)
        log_ort_session("FuXi", self._sess)
        self._loaded = True

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Any = None,
    ) -> ModelState:
        if prev_blob is None:
            raise ValueError("FuXi 需要 prev_blob（t-6h 历史帧）")
        raw = blobs_to_fuxi_2frame(prev_blob, init_blob)  # (2, 70, H, W)
        x, layout = fuxi_prepare_onnx_input(raw, self._sess)
        self._layout = layout
        return ModelState(
            data={"cur": x.astype(np.float32)},
            blob=init_blob,
            lead=0,
            extra={"layout": layout},
        )

    def step(self, state: ModelState) -> ModelState:
        cur = state.data["cur"]
        layout = state.extra.get("layout", self._layout or "NTCHW")
        next_lead = state.lead + self._step_h

        feeds: Dict[str, np.ndarray] = {}
        for inp in self._sess.get_inputs():
            if "temb" in inp.name.lower():
                feeds[inp.name] = fuxi_temb(next_lead)
            else:
                feeds[inp.name] = cur
        y = self._sess.run(None, feeds)[0].astype(np.float32)

        if y.ndim == 5 and layout == "NTCHW":
            out_latest = y[0, -1]   # (70, H, W)
        elif y.ndim == 5 and layout == "NCTHW":
            out_latest = y[0, :, -1]
        elif y.ndim == 4:
            out_latest = y[0]
        else:
            raise RuntimeError(f"FuXi: 意外输出形状 {y.shape}")

        blob_new = {
            "surface_t2m": out_latest[_FUXI_SURFACE_IDX["t2m"]].astype(np.float32),
            "surface_u10": out_latest[_FUXI_SURFACE_IDX["u10"]].astype(np.float32),
            "surface_v10": out_latest[_FUXI_SURFACE_IDX["v10"]].astype(np.float32),
            "surface_msl": out_latest[_FUXI_SURFACE_IDX["msl"]].astype(np.float32),
            "lat": state.blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32)),
            "lon": state.blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32)),
        }

        return ModelState(
            data={"cur": y.astype(np.float32)},
            blob=blob_new,
            lead=next_lead,
            extra={"layout": layout},
        )

    def unload(self) -> None:
        """释放 ONNX Session，归还 GPU 显存。"""
        import gc
        self._sess = None
        self._loaded = False
        gc.collect()

    def get_surface_var_names(self) -> List[str]:
        return ["t2m", "u10", "v10", "msl"]

    def get_step_hours(self) -> int:
        return self._step_h
