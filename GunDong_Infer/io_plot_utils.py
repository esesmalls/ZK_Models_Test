from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def write_step_nc(
    path: Path,
    *,
    model: str,
    init_time: str,
    lead_hours: int,
    valid_time: str,
    h1000_t: np.ndarray,
    v10: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "h1000_t": (("latitude", "longitude"), np.asarray(h1000_t, dtype=np.float32)),
            "v10": (("latitude", "longitude"), np.asarray(v10, dtype=np.float32)),
        },
        coords={
            "latitude": np.asarray(lat, dtype=np.float32),
            "longitude": np.asarray(lon, dtype=np.float32),
        },
        attrs={
            "model": model,
            "init_time_utc": init_time,
            "lead_hours": int(lead_hours),
            "valid_time_utc": valid_time,
        },
    )
    ds["h1000_t"].attrs["long_name"] = "temperature_at_1000hPa"
    ds["h1000_t"].attrs["units"] = "K"
    ds["v10"].attrs["long_name"] = "10m_v_component_of_wind"
    ds["v10"].attrs["units"] = "m s-1"
    ds.to_netcdf(path)


def _plot_single(path: Path, arr: np.ndarray, title: str, cmap: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="upper")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(
    path: Path,
    pred: np.ndarray,
    truth: Optional[np.ndarray],
    *,
    title: str,
    cmap: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(pred, dtype=np.float64)
    if truth is None:
        _plot_single(path, p.astype(np.float32), f"{title} | no ERA5 truth", cmap)
        return
    t = np.asarray(truth, dtype=np.float64)
    d = p - t
    vmin = float(min(np.nanmin(p), np.nanmin(t)))
    vmax = float(max(np.nanmax(p), np.nanmax(t)))
    dv = float(np.nanpercentile(np.abs(d), 99))
    if not np.isfinite(dv) or dv < 1e-9:
        dv = max(float(np.nanmax(np.abs(d))), 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axs[0].imshow(p, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[0].set_title("Forecast")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)
    im1 = axs[1].imshow(t, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[1].set_title("ERA5 analysis")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)
    im2 = axs[2].imshow(d, cmap="RdBu_r", aspect="auto", origin="upper", vmin=-dv, vmax=dv)
    axs[2].set_title("Forecast minus analysis")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

