from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
    vars_2d: Dict[str, np.ndarray],
    vars_3d: Optional[Dict[str, np.ndarray]] = None,
    level_values: Optional[np.ndarray] = None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = {
        "latitude": np.asarray(lat, dtype=np.float32),
        "longitude": np.asarray(lon, dtype=np.float32),
    }
    data_vars: Dict[str, tuple] = {}
    for k, v in vars_2d.items():
        data_vars[k] = (("latitude", "longitude"), np.asarray(v, dtype=np.float32))

    if vars_3d:
        if level_values is None:
            raise ValueError("level_values is required when vars_3d is provided")
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

