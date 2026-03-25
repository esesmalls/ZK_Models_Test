"""
通道映射器：将统一 blob dict → 各模型所需输入张量。

这是解耦"数据格式"与"模型要求"的关键层。
任何 DataAdapter 生成的 blob，通过此模块均可映射为：
  - Pangu ONNX 输入 (p_tensor, s_tensor)
  - FengWu ONNX 输入（单帧/双帧）
  - FuXi ONNX 输入（双帧 70 通道）
  - GraphCast 归一化 state tensor

支持从不同 blob 中选取指定变量用于输出/评估。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from cepri_loader import (  # noqa: E402
    FUXI_LEVELS,
    PANGU_LEVELS,
    specific_humidity_to_relative_humidity,
)

# ------------------------------------------------------------------ #
# Pangu
# ------------------------------------------------------------------ #

def blob_to_pangu_onnx(blob: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      p_tensor: (1, 5, 13, H, W) float32  — z, q, t, u, v
      s_tensor: (1, 4, H, W)    float32  — msl, u10, v10, t2m
    """
    s = np.stack(
        [blob["surface_msl"], blob["surface_u10"], blob["surface_v10"], blob["surface_t2m"]],
        axis=0,
    )[np.newaxis].astype(np.float32)
    p = np.stack(
        [blob["pangu_z"], blob["pangu_q"], blob["pangu_t"], blob["pangu_u"], blob["pangu_v"]],
        axis=0,
    )[np.newaxis].astype(np.float32)
    return p, s


def pangu_onnx_to_blob(p_out: np.ndarray, s_out: np.ndarray) -> Dict[str, np.ndarray]:
    """
    ONNX output → blob dict（复用已有代码路径）。
    p_out: (1, 5, 13, H, W) or (5, 13, H, W)
    s_out: (1, 4, H, W)     or (4, H, W)
    """
    if p_out.ndim == 5:
        p_out = p_out[0]
    if s_out.ndim == 4:
        s_out = s_out[0]
    return {
        "surface_msl": s_out[0].astype(np.float32),
        "surface_u10": s_out[1].astype(np.float32),
        "surface_v10": s_out[2].astype(np.float32),
        "surface_t2m": s_out[3].astype(np.float32),
        "pangu_z": p_out[0].astype(np.float32),
        "pangu_q": p_out[1].astype(np.float32),
        "pangu_t": p_out[2].astype(np.float32),
        "pangu_u": p_out[3].astype(np.float32),
        "pangu_v": p_out[4].astype(np.float32),
    }

# ------------------------------------------------------------------ #
# FengWu
# ------------------------------------------------------------------ #

# 层序重排：PANGU_LEVELS (1000→50) → FUXI_LEVELS (50→1000)
_PANGU_TO_FUXI_ORDER = [
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50].index(int(lv))
    for lv in FUXI_LEVELS
]


def blob_to_fengwu_69ch(blob: Dict[str, np.ndarray]) -> np.ndarray:
    """
    69 通道单帧：[u10,v10,t2m,msl, z(50→1000), q(50→1000), u(50→1000), v(50→1000), t(50→1000)]
    shape: (69, H, W)
    """
    order = _PANGU_TO_FUXI_ORDER
    sfc = np.stack(
        [blob["surface_u10"], blob["surface_v10"], blob["surface_t2m"], blob["surface_msl"]],
        axis=0,
    ).astype(np.float32)
    z = blob["pangu_z"][order].astype(np.float32)
    q = blob["pangu_q"][order].astype(np.float32)
    u = blob["pangu_u"][order].astype(np.float32)
    v = blob["pangu_v"][order].astype(np.float32)
    t = blob["pangu_t"][order].astype(np.float32)
    return np.concatenate([sfc, z, q, u, v, t], axis=0)


def blob_to_fengwu_138ch(
    blob_prev: Dict[str, np.ndarray],
    blob_now: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    138 通道双帧：[frame(t-6h), frame(t0)]，每帧 69ch。
    shape: (1, 138, H, W)
    """
    f0 = blob_to_fengwu_69ch(blob_prev)
    f1 = blob_to_fengwu_69ch(blob_now)
    return np.concatenate([f0, f1], axis=0)[np.newaxis].astype(np.float32)


def fengwu_pred69_to_blob(pred69: np.ndarray) -> Dict[str, np.ndarray]:
    """
    69ch 帧反解 → blob dict（地表+气压）。
    FUXI_LEVELS 顺序 (50→1000) 转回 PANGU_LEVELS (1000→50)。
    """
    if pred69.ndim > 3:
        pred69 = pred69.squeeze()
    order_back = [FUXI_LEVELS.index(float(lv)) for lv in PANGU_LEVELS]
    blob: Dict[str, np.ndarray] = {
        "surface_u10": pred69[0].astype(np.float32),
        "surface_v10": pred69[1].astype(np.float32),
        "surface_t2m": pred69[2].astype(np.float32),
        "surface_msl": pred69[3].astype(np.float32),
        "pangu_z": pred69[4:17][order_back].astype(np.float32),
        "pangu_q": pred69[17:30][order_back].astype(np.float32),
        "pangu_u": pred69[30:43][order_back].astype(np.float32),
        "pangu_v": pred69[43:56][order_back].astype(np.float32),
        "pangu_t": pred69[56:69][order_back].astype(np.float32),
    }
    return blob


# ------------------------------------------------------------------ #
# FuXi
# ------------------------------------------------------------------ #

def blob_to_fuxi_70ch(blob: Dict[str, np.ndarray], tp_fill: float = 0.0) -> np.ndarray:
    """
    70 通道单帧（FuXi 物理顺序）：
      Z13,T13,U13,V13,R13（FUXI_LEVELS 50→1000），T2M,U10,V10,MSL,TP
    shape: (70, H, W)
    """
    order = _PANGU_TO_FUXI_ORDER
    z13 = blob["pangu_z"][order].astype(np.float32)
    t13 = blob["pangu_t"][order].astype(np.float32)
    u13 = blob["pangu_u"][order].astype(np.float32)
    v13 = blob["pangu_v"][order].astype(np.float32)
    q13 = blob["pangu_q"][order].astype(np.float32)
    r13 = np.empty_like(q13, dtype=np.float32)
    for i, lev in enumerate(FUXI_LEVELS):
        r13[i] = specific_humidity_to_relative_humidity(q13[i], t13[i], float(lev))
    s5 = np.stack(
        [
            blob["surface_t2m"],
            blob["surface_u10"],
            blob["surface_v10"],
            blob["surface_msl"],
            np.full_like(blob["surface_msl"], tp_fill, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    upper = np.concatenate([z13, t13, u13, v13, r13], axis=0)
    return np.concatenate([upper, s5], axis=0)


def blobs_to_fuxi_2frame(
    blob_prev: Dict[str, np.ndarray],
    blob_now: Dict[str, np.ndarray],
) -> np.ndarray:
    """shape: (2, 70, H, W) float32"""
    f0 = blob_to_fuxi_70ch(blob_prev)
    f1 = blob_to_fuxi_70ch(blob_now)
    return np.stack([f0, f1], axis=0).astype(np.float32)


# ------------------------------------------------------------------ #
# GraphCast
# ------------------------------------------------------------------ #

def blob_to_graphcast_norm(
    blob: Dict[str, np.ndarray],
    channels: List[str],
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """
    按 cfg.channels 顺序从 blob 组装，并归一化。
    返回: (C, H, W) float32，已归一化。
    """
    level_src = [int(lv) for lv in PANGU_LEVELS]  # [1000,925,...,50]

    def _lv_idx(hpa: int) -> int:
        return level_src.index(hpa)

    stack: List[np.ndarray] = []
    for ch in channels:
        if ch == "10m_u_component_of_wind":
            stack.append(blob["surface_u10"])
        elif ch == "10m_v_component_of_wind":
            stack.append(blob["surface_v10"])
        elif ch == "2m_temperature":
            stack.append(blob["surface_t2m"])
        elif ch == "mean_sea_level_pressure":
            stack.append(blob["surface_msl"])
        elif ch.startswith("geopotential_"):
            stack.append(blob["pangu_z"][_lv_idx(int(ch.split("_")[-1]))])
        elif ch.startswith("specific_humidity_"):
            stack.append(blob["pangu_q"][_lv_idx(int(ch.split("_")[-1]))])
        elif ch.startswith("temperature_"):
            stack.append(blob["pangu_t"][_lv_idx(int(ch.split("_")[-1]))])
        elif ch.startswith("u_component_of_wind_"):
            stack.append(blob["pangu_u"][_lv_idx(int(ch.split("_")[-1]))])
        elif ch.startswith("v_component_of_wind_"):
            stack.append(blob["pangu_v"][_lv_idx(int(ch.split("_")[-1]))])
        else:
            raise ValueError(f"不支持的 GraphCast 通道: {ch}")
    raw = np.stack(stack, axis=0).astype(np.float32)
    return (raw - mu[:, None, None]) / np.maximum(sd[:, None, None], 1e-6)


def graphcast_norm_to_blob(
    arr_norm: np.ndarray,
    channels: List[str],
    mu: np.ndarray,
    sd: np.ndarray,
) -> Dict[str, np.ndarray]:
    """归一化输出 → 物理单位 blob dict（地表变量）。"""
    level_src = [int(lv) for lv in PANGU_LEVELS]
    arr_phys = arr_norm * sd[:, None, None] + mu[:, None, None]
    blob: Dict[str, np.ndarray] = {}
    pz = np.zeros((13,) + arr_phys.shape[1:], dtype=np.float32)
    pq = np.zeros_like(pz)
    pt = np.zeros_like(pz)
    pu = np.zeros_like(pz)
    pv = np.zeros_like(pz)
    for i, ch in enumerate(channels):
        if ch == "10m_u_component_of_wind":
            blob["surface_u10"] = arr_phys[i]
        elif ch == "10m_v_component_of_wind":
            blob["surface_v10"] = arr_phys[i]
        elif ch == "2m_temperature":
            blob["surface_t2m"] = arr_phys[i]
        elif ch == "mean_sea_level_pressure":
            blob["surface_msl"] = arr_phys[i]
        elif ch.startswith("geopotential_"):
            pz[level_src.index(int(ch.split("_")[-1]))] = arr_phys[i]
        elif ch.startswith("specific_humidity_"):
            pq[level_src.index(int(ch.split("_")[-1]))] = arr_phys[i]
        elif ch.startswith("temperature_"):
            pt[level_src.index(int(ch.split("_")[-1]))] = arr_phys[i]
        elif ch.startswith("u_component_of_wind_"):
            pu[level_src.index(int(ch.split("_")[-1]))] = arr_phys[i]
        elif ch.startswith("v_component_of_wind_"):
            pv[level_src.index(int(ch.split("_")[-1]))] = arr_phys[i]
    blob.update({"pangu_z": pz, "pangu_q": pq, "pangu_t": pt, "pangu_u": pu, "pangu_v": pv})
    return blob


# ------------------------------------------------------------------ #
# 通用：从 blob 提取指定变量用于输出/评估
# ------------------------------------------------------------------ #

SURFACE_VAR_KEYS = {
    "u10": "surface_u10",
    "v10": "surface_v10",
    "t2m": "surface_t2m",
    "msl": "surface_msl",
}

PRESSURE_VAR_KEYS = {
    "z": "pangu_z",
    "q": "pangu_q",
    "t": "pangu_t",
    "u": "pangu_u",
    "v": "pangu_v",
}


def extract_surface_vars(
    blob: Dict[str, np.ndarray],
    var_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    从 blob 中提取地表变量。
    var_names=None 返回所有地表变量。
    """
    keys = var_names if var_names else list(SURFACE_VAR_KEYS.keys())
    result: Dict[str, np.ndarray] = {}
    for k in keys:
        bkey = SURFACE_VAR_KEYS.get(k, k)
        if bkey in blob:
            result[k] = blob[bkey]
    return result


def extract_pressure_var_at_level(
    blob: Dict[str, np.ndarray],
    var_name: str,
    level_hpa: int,
) -> np.ndarray:
    """
    提取 blob 中指定要素在指定气压层的二维数组。
    var_name: 'z'|'q'|'t'|'u'|'v'
    level_hpa: 整数，必须在 PANGU_LEVELS 中
    """
    bkey = PRESSURE_VAR_KEYS[var_name]
    idx = [int(lv) for lv in PANGU_LEVELS].index(level_hpa)
    return blob[bkey][idx]
