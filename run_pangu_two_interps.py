#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import importlib.util
import sys
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent


def _load_py(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lat_lon() -> tuple[np.ndarray, np.ndarray]:
    # ERA5 0.25deg global grid used in this project: 721 x 1440
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    return lat, lon


def _write_pangu_nc(raw_dir: Path, out_base_dir: Path, init_time: str, num_steps: int) -> Path:
    """
    Convert Pangu NPY outputs into files expected by the test-interp scripts:
    <base>/<YYYYMMDDTHH>/{t2m,u10,v10}.nc with dims (step, lat, lon).
    """
    lat, lon = _lat_lon()
    p_dir = out_base_dir / init_time
    p_dir.mkdir(parents=True, exist_ok=True)

    t2m_list: list[np.ndarray] = []
    u10_list: list[np.ndarray] = []
    v10_list: list[np.ndarray] = []
    for i in range(1, num_steps + 1):
        s = np.load(raw_dir / f"pangu_step{i:03d}_surface.npy")
        # s shape: (1,4,H,W)
        t2m_list.append(np.asarray(s[0, 3], dtype=np.float32))
        u10_list.append(np.asarray(s[0, 1], dtype=np.float32))
        v10_list.append(np.asarray(s[0, 2], dtype=np.float32))

    step = np.arange(1, num_steps + 1, dtype=np.int32)
    var_map = {
        "t2m": np.stack(t2m_list, axis=0),
        "u10": np.stack(u10_list, axis=0),
        "v10": np.stack(v10_list, axis=0),
    }
    for var, arr in var_map.items():
        ds = xr.Dataset(
            data_vars={var: (("step", "lat", "lon"), arr)},
            coords={"step": step, "lat": lat, "lon": lon},
            attrs={"source_model": "pangu", "init_time": init_time},
        )
        ds[var].attrs["init_time"] = init_time
        ds.to_netcdf(p_dir / f"{var}.nc")
    return p_dir


def _run_interp_with_module(mod, input_folder: Path, output_dir: Path) -> list[Path]:
    # Apply the same logic with module-specific region/resolution constants.
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    init_time = input_folder.name
    for var_key in mod.TARGET_VARS:
        file_path = mod.find_var_file(str(input_folder), var_key)
        if not file_path:
            continue
        ds = xr.open_dataset(file_path)
        try:
            actual_var = var_key
            if actual_var not in ds.variables:
                candidates = [v for v in ds.variables if var_key in v.lower()]
                if not candidates:
                    continue
                actual_var = candidates[0]
            data_var = ds[actual_var]
            lon_vals = ds["lon"].values if "lon" in ds else ds["longitude"].values
            lat_vals = ds["lat"].values if "lat" in ds else ds["latitude"].values

            lat_idx, lon_idx = mod.get_region_indices(
                lon_vals,
                lat_vals,
                mod.REGION_LON_MIN,
                mod.REGION_LON_MAX,
                mod.REGION_LAT_MIN,
                mod.REGION_LAT_MAX,
            )

            if "init_time" in data_var.dims:
                subset = data_var.isel(init_time=0)
            elif "time" in data_var.dims:
                subset = data_var.isel(time=0)
            else:
                subset = data_var

            region_data = subset.isel({"lat": lat_idx, "lon": lon_idx})
            raw_vals = np.asarray(region_data.values)
            proc_vals = raw_vals
            if var_key == "t2m" and np.nanmax(raw_vals) > 1000:
                proc_vals = raw_vals / 100.0

            reg_lat = lat_vals[lat_idx]
            reg_lon = lon_vals[lon_idx]
            p_lon, p_data = reg_lon, proc_vals
            if np.max(reg_lon) > 180:
                p_lon, p_data = mod.convert_lon_360_to_180_numpy(reg_lon, proc_vals)

            interp_val, interp_lon, interp_lat = mod.interpolate_to_grid_numpy(
                p_data, p_lon, reg_lat, mod.TARGET_RESOLUTION
            )
            res_str = str(mod.TARGET_RESOLUTION).replace(".", "")
            out_path = output_dir / f"{var_key}_{init_time}_interp_{res_str}deg.nc"
            _save_nc_simple(
                data_val=interp_val,
                lon=interp_lon,
                lat=interp_lat,
                output_path=out_path,
                var_name=actual_var,
                init_time_str=init_time,
                original_ds=region_data,
                region_name=f"{mod.REGION_LON_MIN}-{mod.REGION_LON_MAX}E_{mod.REGION_LAT_MIN}-{mod.REGION_LAT_MAX}N",
                res=mod.TARGET_RESOLUTION,
            )
            out_paths.append(out_path)
        finally:
            ds.close()
    return out_paths


def _save_nc_simple(
    data_val: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    output_path: Path,
    var_name: str,
    init_time_str: str,
    original_ds: xr.DataArray,
    region_name: str,
    res: float,
) -> None:
    coords = {"latitude": lat, "longitude": lon}
    if data_val.ndim == 3:
        if "step" in original_ds.coords:
            coords["step"] = np.asarray(original_ds.coords["step"].values)
        else:
            coords["step"] = np.arange(data_val.shape[0], dtype=np.int32)
        da = xr.DataArray(data_val, dims=("step", "latitude", "longitude"), coords=coords, name=var_name)
    else:
        da = xr.DataArray(data_val, dims=("latitude", "longitude"), coords=coords, name=var_name)
    da.attrs["init_time"] = init_time_str
    da.attrs["interp_resolution_deg"] = float(res)
    ds = da.to_dataset()
    ds.attrs["source_init_time"] = init_time_str
    ds.attrs["region"] = region_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid netCDF4 zlib path to improve runtime stability in mixed HPC envs.
    ds.to_netcdf(output_path, engine="scipy")


def _plot_from_nc(nc_path: Path, out_png: Path, title: str) -> None:
    ds = xr.open_dataset(nc_path)
    try:
        var = list(ds.data_vars.keys())[0]
        arr = ds[var].values
        if arr.ndim == 3:
            img = arr[0]
        else:
            img = arr
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(img, cmap="viridis", aspect="auto", origin="upper")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
    finally:
        ds.close()


def _plot_side_by_side(nc_01: Path, nc_003: Path, out_png: Path, title: str) -> None:
    ds1 = xr.open_dataset(nc_01)
    ds2 = xr.open_dataset(nc_003)
    try:
        v1 = list(ds1.data_vars.keys())[0]
        v2 = list(ds2.data_vars.keys())[0]
        a1 = ds1[v1].values[0] if ds1[v1].values.ndim == 3 else ds1[v1].values
        a2 = ds2[v2].values[0] if ds2[v2].values.ndim == 3 else ds2[v2].values
        vmin = float(min(np.nanmin(a1), np.nanmin(a2)))
        vmax = float(max(np.nanmax(a1), np.nanmax(a2)))

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        im1 = axs[0].imshow(a1, cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        axs[0].set_title("01deg")
        plt.colorbar(im1, ax=axs[0], fraction=0.03, pad=0.02)
        im2 = axs[1].imshow(a2, cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        axs[1].set_title("003deg")
        plt.colorbar(im2, ax=axs[1], fraction=0.03, pad=0.02)
        fig.suptitle(title)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
    finally:
        ds1.close()
        ds2.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="19790101")
    ap.add_argument("--hour", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--era5-root", type=Path, default=Path("/public/share/aciwgvx1jd/CEPRI_ERA5"))
    ap.add_argument(
        "--reuse-raw-dir",
        type=Path,
        default=None,
        help="已有 pangu_stepXXX_{pressure,surface}.npy 的目录；提供后将跳过 run_pangu",
    )
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = HERE / "results" / f"pangu_interp_bundle_{ts}"
    raw_dir = result_root / "pangu_raw"
    input_base = result_root / "pangu_for_interp_input"
    interp01_dir = result_root / "interp_01deg_nc"
    interp003_dir = result_root / "interp_003deg_nc"
    plot_dir = result_root / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_raw_dir is None:
        from infer_cepri_onnx import pick_providers, run_pangu

        providers = pick_providers("dcu")
        run_pangu(
            era5_root=args.era5_root,
            date_yyyymmdd=args.date,
            hour=args.hour,
            num_steps=args.num_steps,
            providers=providers,
            out_dir=raw_dir,
            save_pngs=False,
        )
        used_raw_dir = raw_dir
    else:
        used_raw_dir = args.reuse_raw_dir
        if not used_raw_dir.is_dir():
            raise FileNotFoundError(used_raw_dir)

    init_time = f"{args.date}T{args.hour:02d}"
    inp_folder = _write_pangu_nc(used_raw_dir, input_base, init_time, args.num_steps)

    m01 = _load_py(HERE / "Test-interp-01deg.py", "test_interp_01")
    m003 = _load_py(HERE / "Test-interp-003.py", "test_interp_003")

    out01 = _run_interp_with_module(m01, inp_folder, interp01_dir)
    out003 = _run_interp_with_module(m003, inp_folder, interp003_dir)

    # per-script visualization
    for p in out01:
        _plot_from_nc(p, plot_dir / "01deg" / f"{p.stem}.png", f"01deg {p.stem}")
    for p in out003:
        _plot_from_nc(p, plot_dir / "003deg" / f"{p.stem}.png", f"003deg {p.stem}")

    # side-by-side comparison
    map01 = {p.name.split("_")[0]: p for p in out01}
    map003 = {p.name.split("_")[0]: p for p in out003}
    for var in sorted(set(map01).intersection(map003)):
        _plot_side_by_side(
            map01[var],
            map003[var],
            plot_dir / "compare_side_by_side" / f"{var}_01_vs_003.png",
            f"{var} interpolation comparison (step1)",
        )

    print(f"RESULT_ROOT={result_root}")
    print(f"RAW_PANGU={used_raw_dir}")
    print(f"INTERP_01_NC={interp01_dir}")
    print(f"INTERP_003_NC={interp003_dir}")
    print(f"PLOTS={plot_dir}")


if __name__ == "__main__":
    sys.exit(main())
