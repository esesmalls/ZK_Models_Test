"""
FengWu ONNX 模型封装。

自动识别输入模式：
  - 138 通道 → 双帧输入（v2 默认，需要 t-6h 历史帧）
  - 189 通道 → 单帧 37 层（v1 可能）
  - 69 通道  → 单帧 13 层
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel
from ..data.channel_mapper import (
    blob_to_fengwu_69ch,
    blob_to_fengwu_138ch,
    fengwu_pred69_to_blob,
)

_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from infer_cepri_onnx import (  # noqa: E402
    create_session,
    fengwu_denorm_chw,
    fengwu_normalize_for_onnx,
    log_ort_session,
    pick_providers,
    unpack_fengwu_ort_outputs,
)


class FengWuModel(WeatherModel):
    MODEL_NAME = "fengwu"

    def __init__(self):
        self._sess = None
        self._stats_dir: Optional[Path] = None
        self._input_mode: Optional[str] = None  # '138ch' | '69ch' | '189ch'
        self._loaded = False
        self._step_h = 6

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        version = cfg.get("default_version", "v2")
        paths = cfg.get("paths", {})
        onnx_path = Path(paths.get(version, paths.get("v2", "")))
        if not onnx_path.is_file():
            raise FileNotFoundError(f"FengWu: 找不到 ONNX 文件 {onnx_path}")

        providers = pick_providers(str(device) if not isinstance(device, str) else device)
        self._sess = create_session(onnx_path, providers)
        log_ort_session("FengWu", self._sess)

        stats_dir = cfg.get("stats_dir")
        if stats_dir:
            p = Path(stats_dir)
            if p.is_dir():
                self._stats_dir = p

        # 自动检测输入模式
        inps = self._sess.get_inputs()
        c = inps[0].shape[1] if len(inps) == 1 else None
        if isinstance(c, int):
            if c == 138:
                self._input_mode = "138ch"
            elif c == 189:
                self._input_mode = "189ch"
            elif c == 69:
                self._input_mode = "69ch"
            else:
                self._input_mode = "unknown"
        else:
            self._input_mode = "138ch"  # 默认双帧
        self._loaded = True

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Any = None,
    ) -> ModelState:
        if self._input_mode == "138ch":
            if prev_blob is None:
                raise ValueError("FengWu 138ch 模式需要 prev_blob（t-6h 历史帧）")
            x = blob_to_fengwu_138ch(prev_blob, init_blob)
        else:
            x = blob_to_fengwu_69ch(init_blob)[np.newaxis]  # (1, 69, H, W)

        if self._stats_dir is not None:
            x = fengwu_normalize_for_onnx(x, self._stats_dir)

        return ModelState(
            data={"cur": x.astype(np.float32)},
            blob=init_blob,
            lead=0,
        )

    def step(self, state: ModelState) -> ModelState:
        cur = state.data["cur"]
        inps = self._sess.get_inputs()
        outs = self._sess.run(None, {inps[0].name: cur})

        surf_o, _, _, _, _, _ = unpack_fengwu_ort_outputs(outs)
        pred69: Optional[np.ndarray] = None
        if len(outs) == 1:
            yo = np.asarray(outs[0], dtype=np.float32)
            while yo.ndim > 3 and yo.shape[0] == 1:
                yo = yo[0]
            if yo.ndim == 3 and yo.shape[0] >= 69:
                pred69 = yo[:69]

        normalized = self._stats_dir is not None
        if pred69 is not None and normalized:
            pred69 = fengwu_denorm_chw(pred69, self._stats_dir)
            surf_o = pred69[:4]

        pred_phys = {
            "surface_u10": np.asarray(surf_o[0], dtype=np.float32),
            "surface_v10": np.asarray(surf_o[1], dtype=np.float32),
            "surface_t2m": np.asarray(surf_o[2], dtype=np.float32),
            "surface_msl": np.asarray(surf_o[3], dtype=np.float32),
        }
        if pred69 is not None:
            extra_blob = fengwu_pred69_to_blob(pred69)
            pred_phys.update({k: v for k, v in extra_blob.items() if k not in pred_phys})
        pred_phys["lat"] = state.blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        pred_phys["lon"] = state.blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))

        # 自回归更新 cur
        if self._input_mode == "138ch" and pred69 is not None:
            pred69_for_norm = pred69.copy()
            if normalized:
                pred69_for_norm = fengwu_normalize_for_onnx(
                    pred69_for_norm[np.newaxis], self._stats_dir
                )[0]
            new_cur = np.concatenate(
                [cur[:, 69:], pred69_for_norm[np.newaxis]], axis=1
            ).astype(np.float32)
        else:
            new_cur = np.asarray(outs[0], dtype=np.float32)
            if new_cur.shape != cur.shape:
                new_cur = cur  # 安全回退

        return ModelState(
            data={"cur": new_cur},
            blob=pred_phys,
            lead=state.lead + self._step_h,
        )

    def unload(self) -> None:
        """释放 ONNX Session，归还 GPU 显存。"""
        import gc
        self._sess = None
        self._loaded = False
        gc.collect()

    def get_surface_var_names(self) -> List[str]:
        return ["u10", "v10", "t2m", "msl"]

    def get_step_hours(self) -> int:
        return self._step_h
