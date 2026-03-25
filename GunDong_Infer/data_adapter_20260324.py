from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset

# Keep exactly aligned with ZK_Models/cepri_loader.py
PANGU_LEVELS: List[float] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


@dataclass
class DayPaths:
    pressure_nc: Path
    surface_nc: Path


def _dkey(date_yyyymmdd: str) -> str:
    return f"{date_yyyymmdd[:4]}_{date_yyyymmdd[4:6]}_{date_yyyymmdd[6:8]}"


def _day_paths(data_root: Path, date_yyyymmdd: str) -> DayPaths:
    stem = _dkey(date_yyyymmdd)
    p = data_root / "pressure" / "pressure" / f"{stem}_pressure.nc"
    s = data_root / "surface" / f"{stem}_surface_instant.nc"
    if not p.is_file():
        raise FileNotFoundError(p)
    if not s.is_file():
        raise FileNotFoundError(s)
    return DayPaths(pressure_nc=p, surface_nc=s)


def list_available_dates(data_root: Path) -> List[str]:
    pdir = data_root / "pressure" / "pressure"
    if not pdir.is_dir():
        return []
    out: List[str] = []
    for p in sorted(pdir.glob("*_pressure.nc")):
        name = p.name
        # YYYY_MM_DD_pressure.nc
        if len(name) >= 18 and name[4] == "_" and name[7] == "_":
            out.append(name[:4] + name[5:7] + name[8:10])
    return out


def _ensure_north_south_lat(data: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lat.size >= 2 and float(lat[0]) < float(lat[1]):
        return data[..., ::-1, :], lat[::-1].copy()
    return data, lat


def _find_time_index_by_hour(ts_sec: np.ndarray, hour: int) -> int:
    if not (0 <= int(hour) < 24):
        raise ValueError("hour must be in [0, 23]")
    # CEPRI-like files have exact 24 timestamps for a day.
    for i, v in enumerate(np.asarray(ts_sec).tolist()):
        h = datetime.utcfromtimestamp(int(v)).hour
        if h == int(hour):
            return int(i)
    # Fallback: assume regular hourly and index==hour.
    if len(ts_sec) > hour:
        return int(hour)
    raise ValueError(f"cannot locate hour={hour} in valid_time")


def _interp_levels(
    x_src: np.ndarray,
    p_src: np.ndarray,
    p_tgt: np.ndarray,
) -> np.ndarray:
    # Fast path: many ERA5 exports already provide exactly the target 13 levels.
    src = np.asarray(p_src, dtype=np.float64)
    tgt = np.asarray(p_tgt, dtype=np.float64)
    if src.shape == tgt.shape and np.allclose(np.sort(src), np.sort(tgt), atol=1e-6):
        idx = [int(np.where(np.isclose(src, lv, atol=1e-6))[0][0]) for lv in tgt]
        return np.asarray(x_src[idx], dtype=np.float32)

    p_src = np.asarray(p_src, dtype=np.float64)
    order = np.argsort(np.log(p_src))
    ps = p_src[order]
    xs = x_src[order]
    log_ps = np.log(np.clip(ps, 1.0, 2000.0))
    log_pt = np.log(np.clip(np.asarray(p_tgt, dtype=np.float64), 1.0, 2000.0))
    out = np.empty((len(p_tgt),) + x_src.shape[1:], dtype=np.float32)
    flat_src = xs.reshape(xs.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    for i in range(flat_src.shape[1]):
        flat_out[:, i] = np.interp(log_pt, log_ps, flat_src[:, i]).astype(np.float32)
    return out


def load_time_blob(data_root: Path, date_yyyymmdd: str, hour: int) -> Dict[str, np.ndarray]:
    """
    Load one analysis hour and convert to a CEPRI-compatible blob used by existing code.
    Returns surface_* and pangu_{z,q,t,u,v}.
    """
    paths = _day_paths(data_root, date_yyyymmdd)
    dp = Dataset(str(paths.pressure_nc))
    ds = Dataset(str(paths.surface_nc))
    try:
        p_levels = np.array(dp.variables["pressure_level"][:], dtype=np.float64)
        p_time = np.array(dp.variables["valid_time"][:], dtype=np.int64)
        s_time = np.array(ds.variables["valid_time"][:], dtype=np.int64)
        p_i = _find_time_index_by_hour(p_time, hour)
        s_i = _find_time_index_by_hour(s_time, hour)

        lat_p = np.array(dp.variables["latitude"][:], dtype=np.float32)
        lat_s = np.array(ds.variables["latitude"][:], dtype=np.float32)

        def r_p(name: str) -> np.ndarray:
            a = np.array(dp.variables[name][p_i], dtype=np.float32)
            a, _ = _ensure_north_south_lat(a, lat_p)
            return a

        def r_s(name: str) -> np.ndarray:
            a = np.array(ds.variables[name][s_i], dtype=np.float32)
            a, _ = _ensure_north_south_lat(a, lat_s)
            return a

        z_s = r_p("z")
        q_s = r_p("q")
        t_s = r_p("t")
        u_s = r_p("u")
        v_s = r_p("v")

        z13 = _interp_levels(z_s, p_levels, np.asarray(PANGU_LEVELS, dtype=np.float64))
        q13 = _interp_levels(q_s, p_levels, np.asarray(PANGU_LEVELS, dtype=np.float64))
        t13 = _interp_levels(t_s, p_levels, np.asarray(PANGU_LEVELS, dtype=np.float64))
        u13 = _interp_levels(u_s, p_levels, np.asarray(PANGU_LEVELS, dtype=np.float64))
        v13 = _interp_levels(v_s, p_levels, np.asarray(PANGU_LEVELS, dtype=np.float64))

        return {
            "surface_msl": r_s("msl"),
            "surface_u10": r_s("u10"),
            "surface_v10": r_s("v10"),
            "surface_t2m": r_s("t2m"),
            "pangu_z": z13,
            "pangu_q": q13,
            "pangu_t": t13,
            "pangu_u": u13,
            "pangu_v": v13,
            "pressure_src": p_levels.astype(np.float32),
        }
    finally:
        dp.close()
        ds.close()


def load_truth_blob_for_valid_time(data_root: Path, valid_dt_utc: datetime) -> Optional[Dict[str, np.ndarray]]:
    ds = valid_dt_utc.strftime("%Y%m%d")
    hr = int(valid_dt_utc.hour)
    try:
        return load_time_blob(data_root, ds, hr)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        return None

