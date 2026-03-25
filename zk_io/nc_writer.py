"""
NetCDF 输出工具。

重构自 GunDong_Infer/io_plot_utils.py，扩展了 3D 气压场支持。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr


def write_step_nc(
    path: Path,
    *,
    model: str,
    init_time: str,
    lead_hours: int,
    valid_time: str,
    vars_2d: Dict[str, np.ndarray],
    vars_3d: Optional[Dict[str, np.ndarray]] = None,
    level_values: Optional[np.ndarray] = None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    """
    写入单步预报 NetCDF 文件。

    Args:
        path:          输出文件路径（父目录会自动创建）
        model:         模型名（写入 attrs）
        init_time:     起报时间 ISO 字符串
        lead_hours:    预报时效
        valid_time:    有效时间 ISO 字符串
        vars_2d:       {变量名: (H,W) ndarray}
        vars_3d:       {变量名: (n_levels, H, W) ndarray}（可选）
        level_values:  气压层值数组（vars_3d 不为空时必须提供）
        lat/lon:       坐标数组
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coords: Dict = {
        "latitude": np.asarray(lat, dtype=np.float32),
        "longitude": np.asarray(lon, dtype=np.float32),
    }
    data_vars: Dict = {}

    for k, v in vars_2d.items():
        data_vars[k] = (("latitude", "longitude"), np.asarray(v, dtype=np.float32))

    if vars_3d:
        if level_values is None:
            raise ValueError("vars_3d 不为空时 level_values 必须提供")
        coords["level"] = np.asarray(level_values, dtype=np.float32)
        for k, v in vars_3d.items():
            data_vars[k] = (("level", "latitude", "longitude"), np.asarray(v, dtype=np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "model": model,
            "init_time_utc": init_time,
            "lead_hours": int(lead_hours),
            "valid_time_utc": valid_time,
        },
    )
    ds.to_netcdf(str(path))


def write_diff_nc(
    path: Path,
    diff_array: np.ndarray,
    valid_times: list,
    lat: np.ndarray,
    lon: np.ndarray,
    var_name: str,
) -> None:
    """
    写入差值场 NC 文件（eval 阶段可选生成）。
    diff_array: (n_steps, H, W)
    """
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        np.asarray(diff_array, dtype=np.float32),
        coords={
            "time": [pd.Timestamp(t) for t in valid_times],
            "latitude": np.asarray(lat, dtype=np.float32),
            "longitude": np.asarray(lon, dtype=np.float32),
        },
        dims=["time", "latitude", "longitude"],
        name=f"{var_name}_diff",
    )
    da.to_netcdf(str(path))
