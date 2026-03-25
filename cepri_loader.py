"""
Load CEPRI ERA5 NetCDF (daily files) and build model inputs compatible with ZK ONNX models.

Pressure files expose 11 levels: 1000,925,850,700,600,500,400,300,200,150,100 hPa.
Pangu / Fuxi (13-level) need extra levels (typically 250, 50) — filled by log-pressure interpolation.
Fengwu needs 37 ERA5-like levels — filled by interpolation (approximate; RH derived from q,t,p).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError as e:
    raise ImportError("cepri_loader requires netCDF4: pip install netcdf4") from e

# Same as example.py PANGU_DEC['l_ord']
PANGU_LEVELS: List[float] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Fuxi medium/short channel pressure order (examples/earth/fuxi/conf/config.yaml)
FUXI_LEVELS: List[float] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Fengwu onescience config: 37 levels (low to high hPa as stored in training HDF5)
FENGWU_LEVELS: List[float] = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]

# 部分 Fengwu ONNX（如 fengwu_v2）输入为 138 = 2×69，每帧 69 = 4(surface) + 13×5(z,r,u,v,t)，13 层与 PANGU_LEVELS 一致
_FENGWU_SUB13_IDX: Tuple[int, ...] = tuple(FENGWU_LEVELS.index(float(h)) for h in PANGU_LEVELS)


def _nc_path(era5_root: Path, date_yyyymmdd: str) -> Tuple[Path, Path]:
    """优先扁平目录（如 graphcast/test_era5_data/YYYY_MM_DD_*.nc），否则使用 YYYY_MM/ 子目录（CEPRI）。"""
    y, m, d = date_yyyymmdd[:4], date_yyyymmdd[4:6], date_yyyymmdd[6:8]
    stem = f"{y}_{m}_{d}"
    flat_p = era5_root / f"{stem}_pressure.nc"
    flat_s = era5_root / f"{stem}_surface_instant.nc"
    if flat_p.is_file() and flat_s.is_file():
        return flat_p, flat_s
    month_dir = era5_root / f"{y}_{m}"
    return month_dir / f"{stem}_pressure.nc", month_dir / f"{stem}_surface_instant.nc"


def _pressure_coord_1d(dp: Dataset) -> np.ndarray:
    """CEPRI daily files use `pressure`; CDS-style exports use `pressure_level`."""
    if "pressure" in dp.variables:
        return np.array(dp.variables["pressure"][:], dtype=np.float64)
    if "pressure_level" in dp.variables:
        return np.array(dp.variables["pressure_level"][:], dtype=np.float64)
    raise KeyError(
        "pressure NetCDF must define 'pressure' (CEPRI) or 'pressure_level' (CDS-style)"
    )


def _ensure_north_south_lat(data: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """If latitude increases southward, flip so index 0 is north (90°)."""
    if lat.size >= 2 and float(lat[0]) < float(lat[1]):
        return data[..., ::-1, :], lat[::-1].copy()
    return data, lat


def _interp_levels(
    x_src: np.ndarray,
    p_src: np.ndarray,
    p_tgt: np.ndarray,
) -> np.ndarray:
    """
    x_src: (n_src, H, W), p_src ascending or descending — we sort by log(p).
    Returns (n_tgt, H, W).
    """
    p_src = np.asarray(p_src, dtype=np.float64)
    order = np.argsort(np.log(p_src))
    ps = p_src[order]
    xs = x_src[order]
    log_ps = np.log(np.clip(ps, 1.0, 2000.0))
    log_pt = np.log(np.clip(np.asarray(p_tgt, dtype=np.float64), 1.0, 2000.0))
    out = np.empty((len(p_tgt),) + x_src.shape[1:], dtype=np.float32)
    flat = xs.reshape(xs.shape[0], -1)
    for i in range(flat.shape[1]):
        col = flat[:, i]
        out.reshape(out.shape[0], -1)[:, i] = np.interp(log_pt, log_ps, col).astype(np.float32)
    return out


def load_cepri_time(
    era5_root: str | Path,
    date_yyyymmdd: str,
    hour: int,
) -> Dict[str, np.ndarray]:
    """
    Read one hour from CEPRI daily NetCDF files.
    date_yyyymmdd: '20200101'
    Returns dict with:
      surface_msl, surface_u10, surface_v10, surface_t2m: (721, 1440) float32
      z,q,t,u,v: each (13, 721, 1440) on PANGU_LEVELS (after interp)
      pressure_src: original 1d levels from file
    """
    era5_root = Path(era5_root)
    p_nc, s_nc = _nc_path(era5_root, date_yyyymmdd)
    if not p_nc.is_file():
        raise FileNotFoundError(p_nc)
    if not s_nc.is_file():
        raise FileNotFoundError(s_nc)
    if not (0 <= hour < 24):
        raise ValueError("hour must be in [0, 23]")

    dp = Dataset(str(p_nc))
    ds = Dataset(str(s_nc))
    try:
        lat_p = dp.variables["latitude"][:]
        p_levels = _pressure_coord_1d(dp)

        def read_var(name: str) -> np.ndarray:
            v = np.array(dp.variables[name][hour], dtype=np.float32)  # (nlev, lat, lon)
            v, _ = _ensure_north_south_lat(v, lat_p)
            return v

        z_s = read_var("z")
        q_s = read_var("q")
        t_s = read_var("t")
        u_s = read_var("u")
        v_s = read_var("v")

        lat_s = ds.variables["latitude"][:]

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

    stacked = []
    for x in (z_s, q_s, t_s, u_s, v_s):
        stacked.append(_interp_levels(x, p_levels, np.array(PANGU_LEVELS, dtype=np.float64)))
    z13, q13, t13, u13, v13 = stacked

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
        "pressure_src": p_levels,
    }


def pack_pangu_onnx(
    blob: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match example.py / Microsoft Pangu-Weather ONNX:
      input_surface: (1, 4, 721, 1440) — msl, u10, v10, t2m
      input: (1, 5, 13, 721, 1440) — z, q, t, u, v
    """
    s = np.stack(
        [blob["surface_msl"], blob["surface_u10"], blob["surface_v10"], blob["surface_t2m"]],
        axis=0,
    )[np.newaxis, ...].astype(np.float32)
    p = np.stack(
        [
            blob["pangu_z"],
            blob["pangu_q"],
            blob["pangu_t"],
            blob["pangu_u"],
            blob["pangu_v"],
        ],
        axis=0,
    )[np.newaxis, ...].astype(np.float32)
    return p, s


def specific_humidity_to_relative_humidity(q: np.ndarray, t_k: np.ndarray, p_hpa: float) -> np.ndarray:
    """Approximate RH [0,1] from q (kg/kg), T (K), p (hPa)."""
    # Tetens saturation specific humidity
    es = 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65)) * 100.0  # Pa
    p_pa = p_hpa * 100.0
    qs = 0.622 * es / np.maximum(p_pa - 0.378 * es, 1.0)
    rh = q / np.maximum(qs, 1e-9)
    return np.clip(rh, 0.0, 1.0).astype(np.float32)


def load_cepri_fengwu_fields(
    era5_root: str | Path,
    date_yyyymmdd: str,
    hour: int,
) -> Dict[str, np.ndarray]:
    """z, r, u, v, t each (37,721,1440), surface (4,721,1440) u10,v10,t2m,msl."""
    era5_root = Path(era5_root)
    p_nc, s_nc = _nc_path(era5_root, date_yyyymmdd)
    dp = Dataset(str(p_nc))
    ds = Dataset(str(s_nc))
    try:
        lat_p = dp.variables["latitude"][:]
        p_levels = _pressure_coord_1d(dp)

        def read_var(name: str) -> np.ndarray:
            v = np.array(dp.variables[name][hour], dtype=np.float32)
            v, _ = _ensure_north_south_lat(v, lat_p)
            return v

        z_s = read_var("z")
        q_s = read_var("q")
        t_s = read_var("t")
        u_s = read_var("u")
        v_s = read_var("v")

        lat_s = ds.variables["latitude"][:]

        def read_sfc(name: str) -> np.ndarray:
            a = np.array(ds.variables[name][hour], dtype=np.float32)
            a, _ = _ensure_north_south_lat(a, lat_s)
            return a

        surface = np.stack([read_sfc("u10"), read_sfc("v10"), read_sfc("t2m"), read_sfc("msl")], axis=0)
    finally:
        dp.close()
        ds.close()

    p_tgt = np.array(FENGWU_LEVELS, dtype=np.float64)
    z37 = _interp_levels(z_s, p_levels, p_tgt)
    u37 = _interp_levels(u_s, p_levels, p_tgt)
    v37 = _interp_levels(v_s, p_levels, p_tgt)
    t37 = _interp_levels(t_s, p_levels, p_tgt)
    q37 = _interp_levels(q_s, p_levels, p_tgt)
    r37 = np.empty_like(q37, dtype=np.float32)
    for i, lev in enumerate(FENGWU_LEVELS):
        r37[i] = specific_humidity_to_relative_humidity(q37[i], t37[i], float(lev))
    return {
        "surface": surface.astype(np.float32),
        "z": z37,
        "r": r37,
        "u": u37,
        "v": v37,
        "t": t37,
    }


def fengwu_frame_69_from_37(fields: Dict[str, np.ndarray]) -> np.ndarray:
    """
    从 37 层 Fengwu 场抽一帧 69 通道 (4 + 13×5)，层序为 PANGU_LEVELS 在 FENGWU_LEVELS 中的位置。
    用于 138 通道 ONNX（两帧拼接）与完整 189 单帧的前缀布局一致。
    """
    # 必须用 ndarray/list 做高级索引；tuple 会被当成多维下标 (i0,...,i12) 导致 3D 数组报 13 个 index
    lev_ix = np.asarray(_FENGWU_SUB13_IDX, dtype=np.intp)
    surf = fields["surface"]
    parts = [
        surf,
        fields["z"][lev_ix],
        fields["r"][lev_ix],
        fields["u"][lev_ix],
        fields["v"][lev_ix],
        fields["t"][lev_ix],
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_cepri_fuxi_fields(
    era5_root: str | Path,
    date_yyyymmdd: str,
    hour0: int,
    hour1: int,
    tp_fill: float = 0.0,
) -> np.ndarray:
    """
    Two timesteps of 70 channels (Fuxi order), shape (2, 70, 721, 1440), physical units (not normalized).
    Channels (official FuXi): z,t,u,v,r × 13 levels (FUXI_LEVELS order: 50..1000),
    then surface [t2m, u10, v10, msl, tp].
    total_precipitation missing in CEPRI instant file → filled with tp_fill.
    """
    era5_root = Path(era5_root)
    p_nc, s_nc = _nc_path(era5_root, date_yyyymmdd)
    p_tgt = np.array(FUXI_LEVELS, dtype=np.float64)
    chans: List[np.ndarray] = []

    for h in (hour0, hour1):
        dp = Dataset(str(p_nc))
        ds = Dataset(str(s_nc))
        try:
            lat_p = dp.variables["latitude"][:]
            p_levels = _pressure_coord_1d(dp)

            def read_var(name: str) -> np.ndarray:
                v = np.array(dp.variables[name][h], dtype=np.float32)
                v, _ = _ensure_north_south_lat(v, lat_p)
                return v

            z_s = read_var("z")
            q_s = read_var("q")
            t_s = read_var("t")
            u_s = read_var("u")
            v_s = read_var("v")

            lat_s = ds.variables["latitude"][:]

            def read_sfc(name: str) -> np.ndarray:
                a = np.array(ds.variables[name][h], dtype=np.float32)
                a, _ = _ensure_north_south_lat(a, lat_s)
                return a

            u10 = read_sfc("u10")
            v10 = read_sfc("v10")
            t2m = read_sfc("t2m")
            msl = read_sfc("msl")
        finally:
            dp.close()
            ds.close()

        s5 = np.stack([t2m, u10, v10, msl, np.full_like(msl, tp_fill, dtype=np.float32)], axis=0)
        z13 = _interp_levels(z_s, p_levels, p_tgt)
        q13 = _interp_levels(q_s, p_levels, p_tgt)
        t13 = _interp_levels(t_s, p_levels, p_tgt)
        u13 = _interp_levels(u_s, p_levels, p_tgt)
        v13 = _interp_levels(v_s, p_levels, p_tgt)
        r13 = np.empty_like(z13, dtype=np.float32)
        for i, lev in enumerate(FUXI_LEVELS):
            r13[i] = specific_humidity_to_relative_humidity(q13[i], t13[i], float(lev))
        upper = np.concatenate([z13, t13, u13, v13, r13], axis=0)
        chans.append(np.concatenate([upper, s5], axis=0))

    return np.stack(chans, axis=0).astype(np.float32)
