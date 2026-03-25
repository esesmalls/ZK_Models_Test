#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


def main() -> None:
    root = Path(__file__).resolve().parent / "results" / "pangu_interp_bundle_20260324_155315"
    nc01 = root / "interp_01deg_nc" / "t2m_19790101T00_interp_01deg.nc"
    nc03 = root / "interp_003deg_nc" / "t2m_19790101T00_interp_003deg.nc"
    out = root / "plots" / "compare_side_by_side" / "t2m_003domain_native_01_vs_003.png"

    with Dataset(str(nc01), "r") as d1:
        v1 = np.asarray(d1.variables["t2m"][0], dtype=np.float64)
        lat1 = np.asarray(d1.variables["latitude"][:], dtype=np.float64)
        lon1 = np.asarray(d1.variables["longitude"][:], dtype=np.float64)

    with Dataset(str(nc03), "r") as d3:
        v3 = np.asarray(d3.variables["t2m"][0], dtype=np.float64)
        lat3 = np.asarray(d3.variables["latitude"][:], dtype=np.float64)
        lon3 = np.asarray(d3.variables["longitude"][:], dtype=np.float64)

    # 仅做区域裁切，不做重采样
    lat_min, lat_max = float(lat3.min()), float(lat3.max())
    lon_min, lon_max = float(lon3.min()), float(lon3.max())
    mlat = (lat1 >= lat_min) & (lat1 <= lat_max)
    mlon = (lon1 >= lon_min) & (lon1 <= lon_max)
    v1_sub = v1[np.ix_(mlat, mlon)]
    lat1_sub = lat1[mlat]
    lon1_sub = lon1[mlon]

    vmin = float(min(np.nanmin(v1_sub), np.nanmin(v3)))
    vmax = float(max(np.nanmax(v1_sub), np.nanmax(v3)))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    m0 = axs[0].pcolormesh(lon1_sub, lat1_sub, v1_sub, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("01deg native (cropped to 003 domain)")
    axs[0].set_xlabel("Longitude (deg)")
    axs[0].set_ylabel("Latitude (deg)")
    axs[0].set_xlim(lon_min, lon_max)
    axs[0].set_ylim(lat_min, lat_max)
    plt.colorbar(m0, ax=axs[0], fraction=0.03, pad=0.02)

    m1 = axs[1].pcolormesh(lon3, lat3, v3, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("003deg native")
    axs[1].set_xlabel("Longitude (deg)")
    axs[1].set_ylabel("Latitude (deg)")
    axs[1].set_xlim(lon_min, lon_max)
    axs[1].set_ylim(lat_min, lat_max)
    plt.colorbar(m1, ax=axs[1], fraction=0.03, pad=0.02)

    fig.suptitle("t2m native-grid comparison on 003deg domain (no resampling, step1)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
