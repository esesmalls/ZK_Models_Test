#!/usr/bin/env python3
"""
Dedicated FengWu/FuXi inference and plotting pipeline for ERA5 test data.

Design notes:
- Keep this script standalone for FengWu/FuXi workflows only.
- Reuse existing adapters from infer_cepri_onnx.py and cepri_loader.py.
- Follow official channel/time conventions from FuXi/FengWu readme.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import xarray as xr
except Exception:
    xr = None

import pytz

from cepri_loader import (
    FUXI_LEVELS,
    PANGU_LEVELS,
    load_cepri_fuxi_fields,
    load_cepri_time,
    specific_humidity_to_relative_humidity,
)
from infer_cepri_onnx import (
    ZK_ROOT,
    build_fengwu_onnx_combo_input,
    create_session,
    fengwu_denorm_chw,
    fengwu_normalize_for_onnx,
    fuxi_normalize_for_layout,
    fuxi_prepare_onnx_input,
    fuxi_temb,
    pick_providers,
    unpack_fengwu_ort_outputs,
)


GRAPH_ROOT = ZK_ROOT.parent


@dataclass(frozen=True)
class VarSpec:
    key: str
    title: str
    cmap: str


VAR_SPECS: Dict[str, VarSpec] = {
    "u10": VarSpec("u10", "10m u-wind (m/s)", "RdBu_r"),
    "v10": VarSpec("v10", "10m v-wind (m/s)", "RdBu_r"),
    "t2m": VarSpec("t2m", "2m temperature (K)", "viridis"),
    "msl": VarSpec("msl", "MSL pressure (Pa)", "viridis"),
    "z500": VarSpec("z500", "500 hPa geopotential", "viridis"),
    "t850": VarSpec("t850", "850 hPa temperature (K)", "viridis"),
    # For FuXi/FengWu the humidity channel is RH (R*) not specific humidity (Q*).
    # Keep key `q850` for request compatibility, and compare against ERA5 RH converted from q,t.
    "q850": VarSpec("q850", "850 hPa humidity (RH proxy)", "viridis"),
}


def _progress(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [fw_fx] {msg}", flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_triplet(path: Path, pred: np.ndarray, truth: np.ndarray, title: str, cmap: str) -> None:
    _ensure_dir(path.parent)
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    diff = pred - truth

    vmin = float(min(np.nanmin(pred), np.nanmin(truth)))
    vmax = float(max(np.nanmax(pred), np.nanmax(truth)))
    dv = float(np.nanpercentile(np.abs(diff), 99.0))
    if not np.isfinite(dv) or dv < 1e-8:
        dv = max(float(np.nanmax(np.abs(diff))), 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axs[0].imshow(pred, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[0].set_title("Forecast")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)

    im1 = axs[1].imshow(truth, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[1].set_title("ERA5")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)

    im2 = axs[2].imshow(diff, cmap="RdBu_r", aspect="auto", origin="upper", vmin=-dv, vmax=dv)
    axs[2].set_title("Forecast - ERA5")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_family_compare(
    path: Path,
    truth: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    title: str,
    a_name: str,
    b_name: str,
    cmap: str,
) -> None:
    _ensure_dir(path.parent)
    truth = np.asarray(truth, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    vmin = float(min(np.nanmin(truth), np.nanmin(a), np.nanmin(b)))
    vmax = float(max(np.nanmax(truth), np.nanmax(a), np.nanmax(b)))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axs[0].imshow(truth, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[0].set_title("ERA5")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)

    im1 = axs[1].imshow(a, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[1].set_title(a_name)
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)

    im2 = axs[2].imshow(b, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[2].set_title(b_name)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _dt_from_args(date: str, hour: int) -> datetime:
    return datetime(
        int(date[:4]),
        int(date[4:6]),
        int(date[6:8]),
        int(hour),
        tzinfo=pytz.utc,
    )


def _truth_blob(era5_root: Path, valid_time: datetime) -> dict:
    valid_time = valid_time.astimezone(pytz.UTC)
    ds = valid_time.strftime("%Y%m%d")
    hh = int(valid_time.hour)
    return load_cepri_time(era5_root, ds, hh)


def _truth_field(blob: dict, key: str) -> np.ndarray:
    i500 = PANGU_LEVELS.index(500)
    i850 = PANGU_LEVELS.index(850)
    if key == "u10":
        return np.asarray(blob["surface_u10"], dtype=np.float32)
    if key == "v10":
        return np.asarray(blob["surface_v10"], dtype=np.float32)
    if key == "t2m":
        return np.asarray(blob["surface_t2m"], dtype=np.float32)
    if key == "msl":
        return np.asarray(blob["surface_msl"], dtype=np.float32)
    if key == "z500":
        return np.asarray(blob["pangu_z"][i500], dtype=np.float32)
    if key == "t850":
        return np.asarray(blob["pangu_t"][i850], dtype=np.float32)
    if key == "q850":
        rh = specific_humidity_to_relative_humidity(
            blob["pangu_q"][i850],
            blob["pangu_t"][i850],
            850.0,
        )
        return np.asarray(rh, dtype=np.float32)
    raise ValueError(f"unsupported truth key: {key}")


def _fengwu_field(pred69: np.ndarray, key: str) -> np.ndarray:
    i500 = FUXI_LEVELS.index(500)
    i850 = FUXI_LEVELS.index(850)
    if key == "u10":
        return pred69[0]
    if key == "v10":
        return pred69[1]
    if key == "t2m":
        return pred69[2]
    if key == "msl":
        return pred69[3]
    if key == "z500":
        return pred69[4 + i500]
    if key == "q850":
        return pred69[17 + i850]
    if key == "t850":
        return pred69[56 + i850]
    raise ValueError(f"unsupported FengWu key: {key}")


def _fuxi_field(pred70: np.ndarray, key: str) -> np.ndarray:
    i500 = FUXI_LEVELS.index(500)
    i850 = FUXI_LEVELS.index(850)
    if key == "u10":
        return pred70[66]
    if key == "v10":
        return pred70[67]
    if key == "t2m":
        return pred70[65]
    if key == "msl":
        return pred70[68]
    if key == "z500":
        return pred70[i500]
    if key == "t850":
        return pred70[13 + i850]
    if key == "q850":
        return pred70[52 + i850]
    raise ValueError(f"unsupported FuXi key: {key}")


def _load_fuxi_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    gm = stats_dir / "global_means.npy"
    gs = stats_dir / "global_stds.npy"
    if not gm.is_file() or not gs.is_file():
        raise FileNotFoundError(f"FuXi stats not found in {stats_dir}")
    mu = np.load(gm).astype(np.float32)[:, :70, :, :]
    sd = np.load(gs).astype(np.float32)[:, :70, :, :]
    return mu, sd


def run_fengwu_variant(
    variant: str,
    era5_root: Path,
    date: str,
    hour: int,
    num_steps: int,
    providers,
    out_root: Path,
    fw_stats_dir: Path,
    vars_to_plot: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    onnx_path = ZK_ROOT / "fengwu" / f"fengwu_{variant}.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    sess = create_session(onnx_path, providers)
    inps = sess.get_inputs()
    if len(inps) != 1:
        raise RuntimeError(f"FengWu {variant} expects single input")

    cur = build_fengwu_onnx_combo_input(era5_root, date, hour, sess)
    c_in = int(cur.shape[1])
    cur = fengwu_normalize_for_onnx(cur, fw_stats_dir)

    raw_dir = out_root / f"fengwu_{variant}" / "raw"
    _ensure_dir(raw_dir)
    out: Dict[int, Dict[str, np.ndarray]] = {}

    for step in range(1, int(num_steps) + 1):
        _progress(f"FengWu {variant}: step {step}/{num_steps}")
        outs = sess.run(None, {inps[0].name: cur})
        surf, z, r, u, v, t = unpack_fengwu_ort_outputs(outs)
        pred69 = np.concatenate([surf, z, r, u, v, t], axis=0).astype(np.float32)
        pred69 = fengwu_denorm_chw(pred69, fw_stats_dir)
        np.save(raw_dir / f"step{step:03d}_pred69.npy", pred69)
        out[step] = {k: _fengwu_field(pred69, k).astype(np.float32) for k in vars_to_plot}

        if c_in == 138:
            pred69_norm = fengwu_normalize_for_onnx(pred69[np.newaxis, ...], fw_stats_dir)[0]
            cur = np.concatenate([cur[:, 69:], pred69_norm[np.newaxis, ...]], axis=1).astype(np.float32)
        else:
            if step < int(num_steps):
                _progress(
                    f"FengWu {variant}: input channels={c_in}, auto-regressive rollout stops at step {step}"
                )
            break
    return out


def run_fuxi_variant(
    model_key: str,
    era5_root: Path,
    date: str,
    hour: int,
    num_steps: int,
    providers,
    out_root: Path,
    fuxi_stats_dir: Path,
    vars_to_plot: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    rel = "short.onnx" if model_key == "fuxi_short" else "medium.onnx"
    onnx_path = ZK_ROOT / "fuxi" / rel
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    mu, sd = _load_fuxi_stats(fuxi_stats_dir)
    sess = create_session(onnx_path, providers)

    h0 = int(hour)
    h1 = int(min(h0 + 1, 23))
    raw = load_cepri_fuxi_fields(era5_root, date, h0, h1)
    x, layout = fuxi_prepare_onnx_input(raw, sess)
    x = fuxi_normalize_for_layout(x, fuxi_stats_dir, layout)

    raw_dir = out_root / model_key / "raw"
    _ensure_dir(raw_dir)
    out: Dict[int, Dict[str, np.ndarray]] = {}

    cur = x.astype(np.float32)
    inps = sess.get_inputs()
    for step in range(1, int(num_steps) + 1):
        _progress(f"{model_key}: step {step}/{num_steps}")
        feeds = {}
        for inp in inps:
            if "temb" in inp.name.lower():
                feeds[inp.name] = fuxi_temb(6 * step)
            else:
                feeds[inp.name] = cur
        y = sess.run(None, feeds)[0]
        cur = y.astype(np.float32)

        if y.ndim == 5 and layout == "NTCHW":
            latest = y[:, -1]
        elif y.ndim == 5 and layout == "NCTHW":
            latest = y[:, :, -1]
        elif y.ndim == 4:
            latest = y
        else:
            raise RuntimeError(f"{model_key}: unexpected output shape {y.shape}")

        latest = np.asarray(latest, dtype=np.float32)
        if latest.ndim == 4 and latest.shape[0] == 1:
            latest = latest[0]
        denorm = latest * sd[0] + mu[0]
        np.save(raw_dir / f"step{step:03d}_latest_raw.npy", latest)
        np.save(raw_dir / f"step{step:03d}_latest_denorm.npy", denorm)
        out[step] = {k: _fuxi_field(denorm, k).astype(np.float32) for k in vars_to_plot}
    return out


def _write_netcdf(
    out_path: Path,
    model_name: str,
    preds: Dict[int, Dict[str, np.ndarray]],
    start_time: datetime,
) -> None:
    if xr is None:
        _progress(f"netcdf skipped for {model_name}: xarray unavailable")
        return
    if not preds:
        return

    steps = sorted(preds.keys())
    first = preds[steps[0]]
    any_field = next(iter(first.values()))
    nlat, nlon = int(any_field.shape[0]), int(any_field.shape[1])

    lat = np.linspace(90.0, -90.0, nlat, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float32)
    valid_times = np.array(
        [(start_time + timedelta(hours=6 * s)).replace(tzinfo=None) for s in steps],
        dtype="datetime64[ns]",
    )

    data_vars = {}
    for key in first.keys():
        arr = np.stack([preds[s][key] for s in steps], axis=0).astype(np.float32)
        data_vars[key] = (("step", "lat", "lon"), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "step": np.asarray(steps, dtype=np.int32),
            "valid_time": ("step", valid_times),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "model": model_name,
            "description": "FengWu/FuXi inference outputs on ERA5 test data",
        },
    )
    _ensure_dir(out_path.parent)
    ds.to_netcdf(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dedicated FengWu/FuXi inference + plotting")
    ap.add_argument("--test-data", type=Path, default=GRAPH_ROOT / "test_era5_data")
    ap.add_argument("--date", type=str, default="20260301")
    ap.add_argument("--hour", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "dcu", "cuda", "cpu"])
    ap.add_argument(
        "--models",
        type=str,
        default="fengwu_v1,fengwu_v2,fuxi_short,fuxi_medium",
        help="comma separated from: fengwu_v1,fengwu_v2,fuxi_short,fuxi_medium",
    )
    ap.add_argument(
        "--variables",
        type=str,
        default="u10,v10,t2m,msl,z500,t850,q850",
        help="comma separated: u10,v10,t2m,msl,z500,t850,q850",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ZK_ROOT / "results_fengwu_fuxi",
    )
    ap.add_argument(
        "--fuxi-stats-dir",
        type=Path,
        default=Path("/public/home/aciwgvx1jd/newh5/stats"),
    )
    ap.add_argument(
        "--fengwu-stats-dir",
        type=Path,
        default=ZK_ROOT / "fengwu",
    )
    ap.add_argument("--no-netcdf", action="store_true", help="disable netcdf output")
    args = ap.parse_args()

    os.chdir(GRAPH_ROOT)
    _ensure_dir(args.output_root)

    vars_to_plot = [v.strip() for v in args.variables.split(",") if v.strip()]
    for v in vars_to_plot:
        if v not in VAR_SPECS:
            raise ValueError(f"unsupported variable: {v}")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    valid_models = {"fengwu_v1", "fengwu_v2", "fuxi_short", "fuxi_medium"}
    bad = [m for m in model_list if m not in valid_models]
    if bad:
        raise ValueError(f"unsupported models: {bad}")

    providers = pick_providers(args.device)
    _progress(f"providers={providers}")
    _progress(f"run models={model_list}, vars={vars_to_plot}, steps={args.num_steps}")

    start_t = _dt_from_args(args.date, args.hour)

    all_preds: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    if "fengwu_v1" in model_list:
        all_preds["fengwu_v1"] = run_fengwu_variant(
            "v1",
            args.test_data,
            args.date,
            args.hour,
            args.num_steps,
            providers,
            args.output_root,
            args.fengwu_stats_dir,
            vars_to_plot,
        )
    if "fengwu_v2" in model_list:
        all_preds["fengwu_v2"] = run_fengwu_variant(
            "v2",
            args.test_data,
            args.date,
            args.hour,
            args.num_steps,
            providers,
            args.output_root,
            args.fengwu_stats_dir,
            vars_to_plot,
        )
    if "fuxi_short" in model_list:
        all_preds["fuxi_short"] = run_fuxi_variant(
            "fuxi_short",
            args.test_data,
            args.date,
            args.hour,
            args.num_steps,
            providers,
            args.output_root,
            args.fuxi_stats_dir,
            vars_to_plot,
        )
    if "fuxi_medium" in model_list:
        all_preds["fuxi_medium"] = run_fuxi_variant(
            "fuxi_medium",
            args.test_data,
            args.date,
            args.hour,
            args.num_steps,
            providers,
            args.output_root,
            args.fuxi_stats_dir,
            vars_to_plot,
        )

    # Build truth per valid time/step once.
    truth_by_step: Dict[int, Dict[str, np.ndarray]] = {}
    max_step = max((max(d.keys()) for d in all_preds.values() if d), default=0)
    for step in range(1, int(max_step) + 1):
        valid_t = start_t + timedelta(hours=6 * step)
        blob = _truth_blob(args.test_data, valid_t)
        truth_by_step[step] = {k: _truth_field(blob, k) for k in vars_to_plot}

    # Per-model triplet plots.
    for model_name, pred_steps in all_preds.items():
        for step, fields in pred_steps.items():
            for key, pred_arr in fields.items():
                spec = VAR_SPECS[key]
                truth_arr = truth_by_step[step][key]
                title = (
                    f"{model_name} +{6 * step}h | {spec.title} | "
                    f"init={args.date} {args.hour:02d}Z valid={(start_t + timedelta(hours=6 * step)).strftime('%Y-%m-%d %H:%MZ')}"
                )
                out_png = (
                    args.output_root
                    / "plots"
                    / "triplet"
                    / model_name
                    / f"step{step:02d}_{key}_triplet.png"
                )
                _plot_triplet(out_png, pred_arr, truth_arr, title, spec.cmap)

    # Family compare plots: truth + two variants.
    families = [
        ("fengwu", "fengwu_v1", "fengwu_v2"),
        ("fuxi", "fuxi_short", "fuxi_medium"),
    ]
    for fam, a, b in families:
        if a not in all_preds or b not in all_preds:
            continue
        common_steps = sorted(set(all_preds[a].keys()) & set(all_preds[b].keys()))
        for step in common_steps:
            for key in vars_to_plot:
                spec = VAR_SPECS[key]
                title = (
                    f"{fam} compare +{6 * step}h | {spec.title} | "
                    f"init={args.date} {args.hour:02d}Z"
                )
                out_png = (
                    args.output_root
                    / "plots"
                    / "family_compare"
                    / fam
                    / f"step{step:02d}_{key}_compare.png"
                )
                _plot_family_compare(
                    out_png,
                    truth_by_step[step][key],
                    all_preds[a][step][key],
                    all_preds[b][step][key],
                    title,
                    a,
                    b,
                    spec.cmap,
                )

    if not args.no_netcdf:
        for model_name, pred_steps in all_preds.items():
            out_nc = args.output_root / "netcdf" / f"{model_name}.nc"
            _write_netcdf(out_nc, model_name, pred_steps, start_t)
        _write_netcdf(args.output_root / "netcdf" / "era5_truth.nc", "era5_truth", truth_by_step, start_t)

    _progress(f"done, outputs in {args.output_root}")


if __name__ == "__main__":
    main()
