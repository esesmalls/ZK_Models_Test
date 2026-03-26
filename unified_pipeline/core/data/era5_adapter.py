"""
ERA5 平铺格式适配器（CEPRI / test_era5_data 布局）。

文件命名规则：
  {root}/YYYY_MM_DD_pressure.nc
  {root}/YYYY_MM_DD_surface_instant.nc
或月份子目录：
  {root}/YYYY_MM/YYYY_MM_DD_pressure.nc
  {root}/YYYY_MM/YYYY_MM_DD_surface_instant.nc

重构自 cepri_loader.py，通过 DataAdapter 接口统一输出 blob dict。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import DataAdapter

# 复用 ZK_Models 根目录的 cepri_loader 工具函数
_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from cepri_loader import (  # noqa: E402
    PANGU_LEVELS,
    _ensure_north_south_lat,
    _interp_levels,
    _pressure_coord_1d,
)

try:
    from netCDF4 import Dataset
except ImportError as e:
    raise ImportError("ERA5FlatAdapter requires netCDF4: pip install netcdf4") from e

_STD_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)
_STD_LON = np.arange(0.0, 360.0, 0.25, dtype=np.float32)


def _nc_path(root: Path, date_yyyymmdd: str, use_monthly: bool) -> Tuple[Path, Path]:
    y, m, d = date_yyyymmdd[:4], date_yyyymmdd[4:6], date_yyyymmdd[6:8]
    stem = f"{y}_{m}_{d}"
    if use_monthly:
        base = root / f"{y}_{m}"
    else:
        base = root
    return base / f"{stem}_pressure.nc", base / f"{stem}_surface_instant.nc"


class ERA5FlatAdapter(DataAdapter):
    """CEPRI 平铺/月份子目录 ERA5 适配器。"""

    FORMAT_NAME = "era5_flat"

    def __init__(self, root: Path, use_monthly_subdir: bool = False, **kwargs):
        super().__init__(root, **kwargs)
        self.use_monthly = use_monthly_subdir

    def _paths(self, date_yyyymmdd: str) -> Tuple[Path, Path]:
        return _nc_path(self.root, date_yyyymmdd, self.use_monthly)

    def load_blob(self, date_yyyymmdd: str, hour: int) -> Dict[str, np.ndarray]:
        p_nc, s_nc = self._paths(date_yyyymmdd)
        if not p_nc.is_file():
            raise FileNotFoundError(p_nc)
        if not s_nc.is_file():
            raise FileNotFoundError(s_nc)
        if not (0 <= hour < 24):
            raise ValueError(f"hour must be 0..23, got {hour}")

        dp = Dataset(str(p_nc))
        ds = Dataset(str(s_nc))
        try:
            lat_p = np.array(dp.variables["latitude"][:], dtype=np.float32)
            p_levels = _pressure_coord_1d(dp)

            def read_pres(name: str) -> np.ndarray:
                v = np.array(dp.variables[name][hour], dtype=np.float32)
                v, _ = _ensure_north_south_lat(v, lat_p)
                return v

            z_s = read_pres("z")
            q_s = read_pres("q")
            t_s = read_pres("t")
            u_s = read_pres("u")
            v_s = read_pres("v")

            lat_s = np.array(ds.variables["latitude"][:], dtype=np.float32)

            def read_sfc(name: str) -> np.ndarray:
                a = np.array(ds.variables[name][hour], dtype=np.float32)
                a, _ = _ensure_north_south_lat(a, lat_s)
                return a

            msl = read_sfc("msl")
            u10 = read_sfc("u10")
            v10 = read_sfc("v10")
            t2m = read_sfc("t2m")
        finally:
            dp.close()
            ds.close()

        tgt = np.array(PANGU_LEVELS, dtype=np.float64)
        z13 = _interp_levels(z_s, p_levels, tgt)
        q13 = _interp_levels(q_s, p_levels, tgt)
        t13 = _interp_levels(t_s, p_levels, tgt)
        u13 = _interp_levels(u_s, p_levels, tgt)
        v13 = _interp_levels(v_s, p_levels, tgt)

        return {
            "surface_msl": msl,
            "surface_u10": u10,
            "surface_v10": v10,
            "surface_t2m": t2m,
            "pangu_z": z13,
            "pangu_q": q13,
            "pangu_t": t13,
            "pangu_u": u13,
            "pangu_v": v13,
            "pressure_src": p_levels.astype(np.float32),
            "lat": _STD_LAT,
            "lon": _STD_LON,
        }

    def list_dates(self) -> List[str]:
        """扫描 pressure NC 文件，返回所有可用日期（yyyymmdd 列表）。"""
        if self.use_monthly:
            roots = sorted(self.root.glob("????_??"))
        else:
            roots = [self.root]

        dates: List[str] = []
        for d in roots:
            for p in sorted(d.glob("*_pressure.nc")):
                name = p.stem  # YYYY_MM_DD_pressure
                parts = name.split("_")
                if len(parts) >= 3:
                    try:
                        dates.append(f"{parts[0]}{parts[1]}{parts[2]}")
                    except IndexError:
                        pass
        return sorted(set(dates))
