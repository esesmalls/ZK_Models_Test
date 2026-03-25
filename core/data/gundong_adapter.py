"""
GunDong 20260324 格式适配器。

目录布局：
  {root}/pressure/pressure/YYYY_MM_DD_pressure.nc
  {root}/surface/YYYY_MM_DD_surface_instant.nc

重构自 GunDong_Infer/data_adapter_20260324.py。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import DataAdapter

try:
    from netCDF4 import Dataset
except ImportError as e:
    raise ImportError("GunDongAdapter requires netCDF4: pip install netcdf4") from e

PANGU_LEVELS: List[float] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

_STD_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)
_STD_LON = np.arange(0.0, 360.0, 0.25, dtype=np.float32)


def _dkey(date_yyyymmdd: str) -> str:
    return f"{date_yyyymmdd[:4]}_{date_yyyymmdd[4:6]}_{date_yyyymmdd[6:8]}"


def _day_paths(root: Path, date_yyyymmdd: str) -> Tuple[Path, Path]:
    stem = _dkey(date_yyyymmdd)
    p = root / "pressure" / "pressure" / f"{stem}_pressure.nc"
    s = root / "surface" / f"{stem}_surface_instant.nc"
    return p, s


def _ensure_ns_lat(data: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lat.size >= 2 and float(lat[0]) < float(lat[1]):
        return data[..., ::-1, :], lat[::-1].copy()
    return data, lat


def _find_hour_index(ts_sec: np.ndarray, hour: int) -> int:
    for i, v in enumerate(np.asarray(ts_sec).tolist()):
        if datetime.utcfromtimestamp(int(v)).hour == int(hour):
            return int(i)
    if len(ts_sec) > hour:
        return int(hour)
    raise ValueError(f"cannot locate hour={hour} in valid_time array")


def _interp_levels(x_src: np.ndarray, p_src: np.ndarray, p_tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(p_src, dtype=np.float64)
    tgt = np.asarray(p_tgt, dtype=np.float64)
    # fast path: already exact levels
    if src.shape == tgt.shape and np.allclose(np.sort(src), np.sort(tgt), atol=1e-6):
        idx = [int(np.where(np.isclose(src, lv, atol=1e-6))[0][0]) for lv in tgt]
        return np.asarray(x_src[idx], dtype=np.float32)
    order = np.argsort(np.log(np.clip(src, 1.0, 2000.0)))
    ps = src[order]
    xs = x_src[order]
    log_ps = np.log(np.clip(ps, 1.0, 2000.0))
    log_pt = np.log(np.clip(tgt, 1.0, 2000.0))
    out = np.empty((len(tgt),) + x_src.shape[1:], dtype=np.float32)
    flat_src = xs.reshape(xs.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    for i in range(flat_src.shape[1]):
        flat_out[:, i] = np.interp(log_pt, log_ps, flat_src[:, i]).astype(np.float32)
    return out


class GunDongAdapter(DataAdapter):
    """GunDong 20260324 格式 ERA5 适配器。"""

    FORMAT_NAME = "gundong_20260324"

    def load_blob(self, date_yyyymmdd: str, hour: int) -> Dict[str, np.ndarray]:
        p_nc, s_nc = _day_paths(self.root, date_yyyymmdd)
        if not p_nc.is_file():
            raise FileNotFoundError(p_nc)
        if not s_nc.is_file():
            raise FileNotFoundError(s_nc)

        dp = Dataset(str(p_nc))
        ds = Dataset(str(s_nc))
        try:
            p_levels = np.array(dp.variables["pressure_level"][:], dtype=np.float64)
            p_time = np.array(dp.variables["valid_time"][:], dtype=np.int64)
            s_time = np.array(ds.variables["valid_time"][:], dtype=np.int64)
            p_i = _find_hour_index(p_time, hour)
            s_i = _find_hour_index(s_time, hour)

            lat_p = np.array(dp.variables["latitude"][:], dtype=np.float32)
            lat_s = np.array(ds.variables["latitude"][:], dtype=np.float32)

            def r_p(name: str) -> np.ndarray:
                a = np.array(dp.variables[name][p_i], dtype=np.float32)
                a, _ = _ensure_ns_lat(a, lat_p)
                return a

            def r_s(name: str) -> np.ndarray:
                a = np.array(ds.variables[name][s_i], dtype=np.float32)
                a, _ = _ensure_ns_lat(a, lat_s)
                return a

            z_s = r_p("z")
            q_s = r_p("q")
            t_s = r_p("t")
            u_s = r_p("u")
            v_s = r_p("v")
            msl = r_s("msl")
            u10 = r_s("u10")
            v10 = r_s("v10")
            t2m = r_s("t2m")
        finally:
            dp.close()
            ds.close()

        tgt = np.asarray(PANGU_LEVELS, dtype=np.float64)
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
        pdir = self.root / "pressure" / "pressure"
        if not pdir.is_dir():
            return []
        out: List[str] = []
        for p in sorted(pdir.glob("*_pressure.nc")):
            name = p.name  # YYYY_MM_DD_pressure.nc
            parts = name.split("_")
            if len(parts) >= 3:
                try:
                    out.append(f"{parts[0]}{parts[1]}{parts[2]}")
                except IndexError:
                    pass
        return out
