#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def _lat_lon() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    return lat, lon


def _convert_lon_360_to_180(lon: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_lon = lon.copy()
    mask = new_lon > 180.0
    new_lon[mask] -= 360.0
    idx = np.argsort(new_lon)
    return new_lon[idx], data[..., idx]


def _subset_region(
    data: np.ndarray, lat: np.ndarray, lon: np.ndarray, lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("no points in requested region")
    return data[:, lat_idx][:, :, lon_idx], lat[lat_idx], lon[lon_idx]


def _interp_linear(data: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_new: np.ndarray, lon_new: np.ndarray) -> np.ndarray:
    # data: (step, lat, lon); pure numpy two-pass linear interpolation.
    lat_in = lat.astype(np.float64)
    lon_in = lon.astype(np.float64)
    arr = data.astype(np.float64)
    if lat_in[0] > lat_in[-1]:
        lat_in = lat_in[::-1]
        arr = arr[:, ::-1, :]
    if lon_in[0] > lon_in[-1]:
        lon_in = lon_in[::-1]
        arr = arr[:, :, ::-1]

    out = np.empty((arr.shape[0], lat_new.size, lon_new.size), dtype=np.float32)
    tmp = np.empty((arr.shape[0], arr.shape[1], lon_new.size), dtype=np.float64)
    for s in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            tmp[s, j] = np.interp(lon_new, lon_in, arr[s, j], left=np.nan, right=np.nan)
        for i in range(lon_new.size):
            out[s, :, i] = np.interp(lat_new, lat_in, tmp[s, :, i], left=np.nan, right=np.nan).astype(np.float32)
    return out


def _save_nc(path: Path, var_name: str, arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, init_time: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset(str(path), "w")
    try:
        ds.createDimension("step", arr.shape[0])
        ds.createDimension("latitude", arr.shape[1])
        ds.createDimension("longitude", arr.shape[2])
        v_step = ds.createVariable("step", "i4", ("step",))
        v_lat = ds.createVariable("latitude", "f4", ("latitude",))
        v_lon = ds.createVariable("longitude", "f4", ("longitude",))
        v = ds.createVariable(var_name, "f4", ("step", "latitude", "longitude"), zlib=True, complevel=1)
        v_step[:] = np.arange(1, arr.shape[0] + 1, dtype=np.int32)
        v_lat[:] = lat.astype(np.float32)
        v_lon[:] = lon.astype(np.float32)
        v[:] = arr.astype(np.float32)
        ds.source_init_time = init_time
    finally:
        ds.close()


def _plot_first_step(path: Path, arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    im = ax.imshow(arr[0], cmap="viridis", aspect="auto", origin="lower", extent=extent)
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_compare(
    path: Path,
    a: np.ndarray,
    alat: np.ndarray,
    alon: np.ndarray,
    b: np.ndarray,
    blat: np.ndarray,
    blon: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(min(np.nanmin(a[0]), np.nanmin(b[0])))
    vmax = float(max(np.nanmax(a[0]), np.nanmax(b[0])))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    ext_a = (float(alon.min()), float(alon.max()), float(alat.min()), float(alat.max()))
    ext_b = (float(blon.min()), float(blon.max()), float(blat.min()), float(blat.max()))
    im1 = axs[0].imshow(a[0], cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax, extent=ext_a)
    axs[0].set_title("01deg")
    axs[0].set_xlabel("Longitude (deg)")
    axs[0].set_ylabel("Latitude (deg)")
    plt.colorbar(im1, ax=axs[0], fraction=0.03, pad=0.02)
    im2 = axs[1].imshow(b[0], cmap="viridis", aspect="auto", origin="lower", vmin=vmin, vmax=vmax, extent=ext_b)
    axs[1].set_title("003deg")
    axs[1].set_xlabel("Longitude (deg)")
    axs[1].set_ylabel("Latitude (deg)")
    plt.colorbar(im2, ax=axs[1], fraction=0.03, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _load_pangu_surface(raw_dir: Path, num_steps: int) -> dict[str, np.ndarray]:
    t2m, u10, v10 = [], [], []
    for i in range(1, num_steps + 1):
        s = np.load(raw_dir / f"pangu_step{i:03d}_surface.npy")
        t2m.append(s[0, 3].astype(np.float32))
        u10.append(s[0, 1].astype(np.float32))
        v10.append(s[0, 2].astype(np.float32))
    return {"t2m": np.stack(t2m, axis=0), "u10": np.stack(u10, axis=0), "v10": np.stack(v10, axis=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--date", default="19790101")
    ap.add_argument("--hour", type=int, default=0)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent / "results" / f"pangu_interp_bundle_{ts}"
    nc01 = root / "interp_01deg_nc"
    nc003 = root / "interp_003deg_nc"
    plot01 = root / "plots" / "01deg"
    plot003 = root / "plots" / "003deg"
    plotcmp = root / "plots" / "compare_side_by_side"

    print("phase: build latlon", flush=True)
    lat, lon = _lat_lon()
    print("phase: load pangu npy", flush=True)
    vals = _load_pangu_surface(args.raw_dir, args.num_steps)
    init_time = f"{args.date}T{args.hour:02d}"

    # Match Test-interp-01deg.py
    v01: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print("phase: interp 01deg", flush=True)
    for k, arr in vals.items():
        sub, lat_s, lon_s = _subset_region(arr, lat, lon, 60, 140, 10, 60)
        lon2, sub2 = _convert_lon_360_to_180(lon_s, sub)
        new_lat = np.arange(10 - 0.05, 60 + 0.05 + 0.1, 0.1, dtype=np.float32)
        new_lon = np.arange(60 - 0.05, 140 + 0.05 + 0.1, 0.1, dtype=np.float32)
        itp = _interp_linear(sub2, lat_s, lon2, new_lat, new_lon)
        mask_lat = (new_lat >= 10) & (new_lat <= 60)
        mask_lon = (new_lon >= 60) & (new_lon <= 140)
        itp = itp[:, mask_lat][:, :, mask_lon]
        lat_o, lon_o = new_lat[mask_lat], new_lon[mask_lon]
        _save_nc(nc01 / f"{k}_{init_time}_interp_01deg.nc", k, itp, lat_o, lon_o, init_time)
        _plot_first_step(plot01 / f"{k}_{init_time}_interp_01deg.png", itp, lat_o, lon_o, f"01deg {k} step1")
        v01[k] = (itp, lat_o, lon_o)

    # Match Test-interp-003.py
    v003: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    print("phase: interp 003deg", flush=True)
    for k, arr in vals.items():
        sub, lat_s, lon_s = _subset_region(arr, lat, lon, 115, 135, 35, 55)
        lon2, sub2 = _convert_lon_360_to_180(lon_s, sub)
        new_lat = np.arange(35 - 0.015, 55 + 0.015 + 0.03, 0.03, dtype=np.float32)
        new_lon = np.arange(115 - 0.015, 135 + 0.015 + 0.03, 0.03, dtype=np.float32)
        itp = _interp_linear(sub2, lat_s, lon2, new_lat, new_lon)
        mask_lat = (new_lat >= 35) & (new_lat <= 55)
        mask_lon = (new_lon >= 115) & (new_lon <= 135)
        itp = itp[:, mask_lat][:, :, mask_lon]
        lat_o, lon_o = new_lat[mask_lat], new_lon[mask_lon]
        _save_nc(nc003 / f"{k}_{init_time}_interp_003deg.nc", k, itp, lat_o, lon_o, init_time)
        _plot_first_step(plot003 / f"{k}_{init_time}_interp_003deg.png", itp, lat_o, lon_o, f"003deg {k} step1")
        v003[k] = (itp, lat_o, lon_o)

    print("phase: side-by-side plots", flush=True)
    for k in sorted(v01):
        a, alat, alon = v01[k]
        b, blat, blon = v003[k]
        _plot_compare(plotcmp / f"{k}_01_vs_003.png", a, alat, alon, b, blat, blon, f"{k} interpolation comparison (step1)")

    print(f"RESULT_ROOT={root}")
    print(f"INTERP_01_NC={nc01}")
    print(f"INTERP_003_NC={nc003}")
    print(f"PLOTS={root / 'plots'}")


if __name__ == "__main__":
    main()
