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
    out = root / "plots" / "compare_side_by_side" / "t2m_003domain_aligned_01_vs_003.png"

    with Dataset(str(nc01), "r") as d1, Dataset(str(nc03), "r") as d3:
        v1 = np.asarray(d1.variables["t2m"][0], dtype=np.float64)
        lat1 = np.asarray(d1.variables["latitude"][:], dtype=np.float64)
        lon1 = np.asarray(d1.variables["longitude"][:], dtype=np.float64)

        v3 = np.asarray(d3.variables["t2m"][0], dtype=np.float64)
        lat3 = np.asarray(d3.variables["latitude"][:], dtype=np.float64)
        lon3 = np.asarray(d3.variables["longitude"][:], dtype=np.float64)

    if lat1[0] > lat1[-1]:
        lat1 = lat1[::-1]
        v1 = v1[::-1, :]
    if lon1[0] > lon1[-1]:
        lon1 = lon1[::-1]
        v1 = v1[:, ::-1]

    tmp = np.empty((v1.shape[0], lon3.size), dtype=np.float64)
    for j in range(v1.shape[0]):
        tmp[j, :] = np.interp(lon3, lon1, v1[j, :], left=np.nan, right=np.nan)

    v1_to_03 = np.empty((lat3.size, lon3.size), dtype=np.float64)
    for i in range(lon3.size):
        v1_to_03[:, i] = np.interp(lat3, lat1, tmp[:, i], left=np.nan, right=np.nan)

    diff = v1_to_03 - v3
    vmin = float(min(np.nanmin(v1_to_03), np.nanmin(v3)))
    vmax = float(max(np.nanmax(v1_to_03), np.nanmax(v3)))
    d = float(np.nanmax(np.abs(diff)))
    if not np.isfinite(d) or d < 1e-9:
        d = 1.0

    ext = (float(lon3.min()), float(lon3.max()), float(lat3.min()), float(lat3.max()))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    im0 = axs[0].imshow(v1_to_03, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("01deg remapped to 003deg grid")
    axs[0].set_xlabel("Longitude (deg)")
    axs[0].set_ylabel("Latitude (deg)")
    plt.colorbar(im0, ax=axs[0], fraction=0.03, pad=0.02)

    im1 = axs[1].imshow(v3, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("003deg native")
    axs[1].set_xlabel("Longitude (deg)")
    axs[1].set_ylabel("Latitude (deg)")
    plt.colorbar(im1, ax=axs[1], fraction=0.03, pad=0.02)

    im2 = axs[2].imshow(diff, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-d, vmax=d)
    axs[2].set_title("Difference (01->003 minus 003)")
    axs[2].set_xlabel("Longitude (deg)")
    axs[2].set_ylabel("Latitude (deg)")
    plt.colorbar(im2, ax=axs[2], fraction=0.03, pad=0.02)

    fig.suptitle("t2m comparison on 003deg domain (step1)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
