"""
地表变量单位对齐：预报与真值可比（用于出图与指标）。

ERA5/GunDong 常为 MSL 使用 Pa；部分模型输出 hPa。在对比前统一到 Pa。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def harmonize_surface_pair(
    var: str,
    pred: np.ndarray,
    truth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 pred、truth 转为可直接相减的同一单位（尽量不拷贝）。

    当前实现：msl 在「真值像 Pa、预报像 hPa」或反向时，将 hPa 一侧乘以 100。
    """
    if var != "msl":
        return pred, truth

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    mp = float(np.nanmedian(np.abs(p)))
    mt = float(np.nanmedian(np.abs(t)))
    if not np.isfinite(mp) or not np.isfinite(mt) or mp < 1e-6 or mt < 1e-6:
        return pred, truth

    # 典型：truth ~1e5 Pa，pred ~1e3 hPa
    if mt > 20000.0 and mp < 8000.0:
        p = (p * 100.0).astype(np.float32)
        return p, np.asarray(truth, dtype=np.float32)
    if mp > 20000.0 and mt < 8000.0:
        t = (t * 100.0).astype(np.float32)
        return np.asarray(pred, dtype=np.float32), t

    return pred, truth
