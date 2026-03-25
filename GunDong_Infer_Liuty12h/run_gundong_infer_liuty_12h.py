#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytz

# Must set per-process visible device BEFORE importing torch.
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["ROCR_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["HIP_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["HSA_VISIBLE_DEVICES"] = str(_local_rank)

torch = None  # lazy import (only needed for GraphCast export)


THIS_DIR = Path(__file__).resolve().parent
ZK_ROOT = THIS_DIR.parent
GRAPH_ROOT = ZK_ROOT.parent

# Allow importing local adapter/util by relative name.
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ZK_ROOT))
os.chdir(GRAPH_ROOT)

from data_adapter_20260324 import PANGU_LEVELS, load_time_blob
from infer_cepri_onnx import create_session, pangu_one_step, pick_providers
from cepri_loader import pack_pangu_onnx

from ruamel.yaml.scalarfloat import ScalarFloat


def _progress(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def parse_start_datetime(s: str) -> datetime:
    # Expected: YYYYMMDDTHH, e.g. 20260308T12
    if "T" not in s:
        raise ValueError(f"invalid --start-datetime: {s} (expected YYYYMMDDTHH)")
    date_part, hour_part = s.split("T", 1)
    if len(date_part) != 8 or len(hour_part) != 2:
        raise ValueError(f"invalid --start-datetime: {s} (expected YYYYMMDDTHH)")
    year = int(date_part[0:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])
    hour = int(hour_part)
    return datetime(year, month, day, hour, tzinfo=pytz.UTC)


@dataclass(frozen=True)
class MemmapSpec:
    prefix: str
    path: Path


class NpyMemmapWriter:
    """
    Write many .npy files as float32 memmaps: (lead, lat, lon).
    """

    def __init__(
        self,
        specs: Iterable[MemmapSpec],
        *,
        shape: Tuple[int, int, int],
    ) -> None:
        self.shape = shape
        self.memmaps: Dict[str, np.memmap] = {}
        for spec in specs:
            spec.path.parent.mkdir(parents=True, exist_ok=True)
            # Overwrite existing outputs for idempotent reruns.
            mm = np.lib.format.open_memmap(
                spec.path,
                mode="w+",
                dtype=np.float32,
                shape=self.shape,
            )
            self.memmaps[spec.prefix] = mm

    def write_slice(self, prefix: str, lead_idx: int, arr_2d: np.ndarray) -> None:
        self.memmaps[prefix][lead_idx, :, :] = np.asarray(arr_2d, dtype=np.float32)

    def fill_all(self, prefix: str, value: float) -> None:
        self.memmaps[prefix][:, :, :] = np.float32(value)

    def close(self) -> None:
        for mm in self.memmaps.values():
            mm.flush()


def _pangu_specs(init_tag: str, *, surface_only: bool) -> List[MemmapSpec]:
    # Liuty naming (PanGu/ERA5_6H):
    #   msl_surface, t2m_surface, u10_surface, v10_surface
    #   z/q/t/u/v_{level} for levels in PANGU_LEVELS
    levels = [int(x) for x in PANGU_LEVELS]
    surface = ["msl_surface", "t2m_surface", "u10_surface", "v10_surface"]
    vars3 = ["z", "q", "t", "u", "v"]

    out: List[MemmapSpec] = []
    base_dir = Path(
        "/public/share/aciwgvx1jd/GunDong_Infer_result_12h"
    )  # placeholder, caller will rewrite path if needed
    # We'll rewrite paths in main by replacing parent dirs.
    for s in surface:
        out.append(MemmapSpec(prefix=s, path=base_dir / "PanGu" / "ERA5_6H" / f"{s}_{init_tag}.npy"))
    if surface_only:
        return out
    for v in vars3:
        for lv in levels:
            out.append(
                MemmapSpec(
                    prefix=f"{v}_{lv}",
                    path=base_dir / "PanGu" / "ERA5_6H" / f"{v}_{lv}_{init_tag}.npy",
                )
            )
    return out


def _graphcast_specs(init_tag: str, *, surface_only: bool) -> List[MemmapSpec]:
    # Liuty naming (GraphCast/ERA5_6H):
    #   msl, t2m, u10, v10
    #   z/q/t/u/v_{level}
    levels = [int(x) for x in PANGU_LEVELS]
    surface = ["msl", "t2m", "u10", "v10"]
    vars3 = ["z", "q", "t", "u", "v"]

    out: List[MemmapSpec] = []
    base_dir = Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h")  # placeholder
    for s in surface:
        out.append(MemmapSpec(prefix=s, path=base_dir / "GraphCast" / "ERA5_6H" / f"{s}_{init_tag}.npy"))
    if surface_only:
        return out
    for v in vars3:
        for lv in levels:
            out.append(
                MemmapSpec(
                    prefix=f"{v}_{lv}",
                    path=base_dir / "GraphCast" / "ERA5_6H" / f"{v}_{lv}_{init_tag}.npy",
                )
            )
    return out


def _rewrite_specs_path(specs: List[MemmapSpec], model_dir: Path) -> List[MemmapSpec]:
    """
    Replace MemmapSpec.path base with target model_dir while keeping the filename.
    """
    out: List[MemmapSpec] = []
    for sp in specs:
        out.append(MemmapSpec(prefix=sp.prefix, path=model_dir / sp.path.name))
    return out


def _extract_mu_sd_for_graphcast(cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    mu/sd in the same channel order as cfg.channels.
    """
    meta_path = Path(cfg.data_dir) / "metadata.json"
    variables = json.load(open(meta_path, encoding="utf-8"))["variables"]
    channel_indices = [variables.index(c) for c in cfg.channels]
    mu_full = np.load(Path(cfg.stats_dir) / "global_means.npy")
    sd_full = np.load(Path(cfg.stats_dir) / "global_stds.npy")
    mu = mu_full[0, channel_indices, 0, 0].astype(np.float32)
    sd = sd_full[0, channel_indices, 0, 0].astype(np.float32)
    return mu, sd


def _blob_to_graphcast_norm(blob: dict, cfg, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    # Match run_gundong_infer.py logic.
    stack: List[np.ndarray] = []
    for ch in cfg.channels:
        if ch == "10m_u_component_of_wind":
            stack.append(blob["surface_u10"])
        elif ch == "10m_v_component_of_wind":
            stack.append(blob["surface_v10"])
        elif ch == "2m_temperature":
            stack.append(blob["surface_t2m"])
        elif ch == "mean_sea_level_pressure":
            stack.append(blob["surface_msl"])
        elif ch.startswith("geopotential_"):
            lv = int(ch.split("_")[-1])
            stack.append(blob["pangu_z"][PANGU_LEVELS.index(float(lv))])
        elif ch.startswith("specific_humidity_"):
            lv = int(ch.split("_")[-1])
            stack.append(blob["pangu_q"][PANGU_LEVELS.index(float(lv))])
        elif ch.startswith("temperature_"):
            lv = int(ch.split("_")[-1])
            stack.append(blob["pangu_t"][PANGU_LEVELS.index(float(lv))])
        elif ch.startswith("u_component_of_wind_"):
            lv = int(ch.split("_")[-1])
            stack.append(blob["pangu_u"][PANGU_LEVELS.index(float(lv))])
        elif ch.startswith("v_component_of_wind_"):
            lv = int(ch.split("_")[-1])
            stack.append(blob["pangu_v"][PANGU_LEVELS.index(float(lv))])
        else:
            raise ValueError(f"unsupported graphcast channel: {ch}")

    raw = np.stack(stack, axis=0).astype(np.float32)
    return (raw - mu[:, None, None]) / np.maximum(sd[:, None, None], 1e-6)


def _build_graphcast_invar(
    norm_climate: torch.Tensor,
    cfg,
    static_data: torch.Tensor,
    latlon_torch: torch.Tensor,
    forecast_time: datetime,
    device: torch.device,
) -> torch.Tensor:
    import torch as _torch
    from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle

    ts = _torch.tensor([forecast_time.timestamp()], dtype=_torch.float32, device=device)
    cz = cos_zenith_angle(ts, latlon=latlon_torch.to(device)).float()
    cz = _torch.squeeze(cz, dim=2)
    cz = _torch.clamp(cz, min=0.0) - 1.0 / _torch.pi

    doy = float(forecast_time.timetuple().tm_yday)
    tod = forecast_time.hour + forecast_time.minute / 60.0 + forecast_time.second / 3600.0
    ndy = _torch.tensor((doy / 365.0) * (np.pi / 2), dtype=_torch.float32, device=device)
    ntd = _torch.tensor((tod / (24.0 - cfg.dt)) * (np.pi / 2), dtype=_torch.float32, device=device)
    sin_dy = _torch.sin(ndy).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    cos_dy = _torch.cos(ndy).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    sin_td = _torch.sin(ntd).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    cos_td = _torch.cos(ntd).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    return _torch.cat((norm_climate, cz, static_data, sin_dy, cos_dy, sin_td, cos_td), dim=1)


def _gc_channel_to_prefix(ch: str) -> Optional[str]:
    # GraphCast mapping to liuty prefixes.
    if ch == "mean_sea_level_pressure":
        return "msl"
    if ch == "2m_temperature":
        return "t2m"
    if ch == "10m_u_component_of_wind":
        return "u10"
    if ch == "10m_v_component_of_wind":
        return "v10"
    if ch.startswith("geopotential_"):
        lv = ch.split("_")[-1]
        return f"z_{lv}"
    if ch.startswith("specific_humidity_"):
        lv = ch.split("_")[-1]
        return f"q_{lv}"
    if ch.startswith("temperature_"):
        lv = ch.split("_")[-1]
        return f"t_{lv}"
    if ch.startswith("u_component_of_wind_"):
        lv = ch.split("_")[-1]
        return f"u_{lv}"
    if ch.startswith("v_component_of_wind_"):
        lv = ch.split("_")[-1]
        return f"v_{lv}"
    return None


def run_pangu_liuty_export(
    *,
    start_dt: datetime,
    max_lead_hours: int,
    input_root: Path,
    output_root: Path,
    providers,
    surface_only: bool,
) -> None:
    assert max_lead_hours % 6 == 0
    n_leads = max_lead_hours // 6  # horizons: +6..+max_lead
    init_tag = start_dt.strftime("%Y%m%dT%H")

    # Prepare memmaps.
    target_model_dir = output_root / "PanGu" / "ERA5_6H"
    specs = _rewrite_specs_path(_pangu_specs(init_tag, surface_only=surface_only), target_model_dir)
    writer = NpyMemmapWriter(specs, shape=(n_leads, 721, 1440))

    # Load ONNX session: only use 6h model for rollouts.
    onnx6 = ZK_ROOT / "pangu" / "pangu_weather_6.onnx"
    if not onnx6.is_file():
        raise FileNotFoundError(f"Missing required pangu ONNX model: {onnx6}")
    session6 = create_session(onnx6, providers)

    # Init state.
    date_yyyymmdd = start_dt.strftime("%Y%m%d")
    hour0 = int(start_dt.hour)
    blob = load_time_blob(input_root, date_yyyymmdd, hour0)
    p_cur, s_cur = pack_pangu_onnx(blob)
    c6p, c6s = p_cur.copy(), s_cur.copy()

    # Autoregressive rollout: input at T -> predict T+6 -> feed back -> predict T+12 ...
    # lead_idx=0 corresponds to +6h (e.g. 12h init -> 18h output).
    for lead_idx in range(n_leads):
        op, os_ = pangu_one_step(session6, c6p, c6s)
        pr = op[0]  # (5,13,H,W) -> z,q,t,u,v
        s = os_[0]  # (4,H,W) -> msl,u10,v10,t2m

        # Feed predicted state back for the next 6h step.
        c6p, c6s = op, os_

        # Surface.
        if "msl_surface" in writer.memmaps:
            writer.write_slice("msl_surface", lead_idx, s[0])
            writer.write_slice("u10_surface", lead_idx, s[1])
            writer.write_slice("v10_surface", lead_idx, s[2])
            writer.write_slice("t2m_surface", lead_idx, s[3])

        if not surface_only:
            # Pressure levels: order in PANGU_LEVELS matches cepri_loader.PANGU_LEVELS.
            for lv in PANGU_LEVELS:
                li = int(PANGU_LEVELS.index(lv))
                lv_i = int(lv)
                prefix_z = f"z_{lv_i}"
                prefix_q = f"q_{lv_i}"
                prefix_t = f"t_{lv_i}"
                prefix_u = f"u_{lv_i}"
                prefix_v = f"v_{lv_i}"
                if prefix_z not in writer.memmaps:
                    continue
                writer.write_slice(prefix_z, lead_idx, pr[0, li])
                writer.write_slice(prefix_q, lead_idx, pr[1, li])
                writer.write_slice(prefix_t, lead_idx, pr[2, li])
                writer.write_slice(prefix_u, lead_idx, pr[3, li])
                writer.write_slice(prefix_v, lead_idx, pr[4, li])

        if (lead_idx + 1) % 8 == 0 or lead_idx == 0:
            cur_valid = start_dt + timedelta(hours=(lead_idx + 1) * 6)
            _progress(f"[pangu] wrote lead={lead_idx}/{n_leads-1} valid={cur_valid.strftime('%Y-%m-%d %H')} UTC")

    writer.close()
    _progress(f"[pangu] done: wrote {n_leads} leads to {target_model_dir}")


def run_graphcast_liuty_export(
    *,
    start_dt: datetime,
    max_lead_hours: int,
    input_root: Path,
    output_root: Path,
    device: torch.device,
    surface_only: bool,
) -> None:
    import torch as _torch
    from onescience.datapipes.climate.utils.invariant import latlon_grid
    from onescience.models.graphcast.graph_cast_net import GraphCastNet
    from onescience.utils.fcn.YParams import YParams
    from onescience.utils.graphcast.data_utils import StaticData

    assert max_lead_hours % 6 == 0
    n_leads = max_lead_hours // 6
    init_tag = start_dt.strftime("%Y%m%dT%H")

    cfg = YParams(str(GRAPH_ROOT / "conf/config.yaml"), "model")
    mu, sd = _extract_mu_sd_for_graphcast(cfg)

    # Build model.
    ckpt_path = Path(cfg.checkpoint_dir) / "graphcast_finetune.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing GraphCast checkpoint: {ckpt_path}")

    _progress("[graphcast] build GraphCastNet")
    model_dtype = _torch.float32
    input_dim_grid_nodes = (len(cfg.channels) + cfg.use_cos_zenith + 4 * cfg.use_time_of_year_index) * (
        cfg.num_history + 1
    ) + cfg.num_channels_static
    model = GraphCastNet(
        mesh_level=cfg.mesh_level,
        multimesh=cfg.multimesh,
        input_res=tuple(cfg.img_size),
        input_dim_grid_nodes=input_dim_grid_nodes,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=len(cfg.channels),
        processor_type=cfg.processor_type,
        khop_neighbors=cfg.khop_neighbors,
        num_attention_heads=cfg.num_attention_heads,
        processor_layers=cfg.processor_layers,
        hidden_dim=cfg.hidden_dim,
        norm_type=cfg.norm_type,
        do_concat_trick=cfg.concat_trick,
        recompute_activation=cfg.recompute_activation,
    )
    model.set_checkpoint_encoder(cfg.checkpoint_encoder)
    model.set_checkpoint_decoder(cfg.checkpoint_decoder)
    ckpt = _torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dtype=model_dtype).to(device)
    model.eval()

    static_data = StaticData(cfg.static_dir, model.latitudes, model.longitudes).get().to(
        device=device, dtype=_torch.float32
    )
    latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=cfg.img_size[-2:])
    latlon_torch = _torch.tensor(np.stack(latlon, axis=0), dtype=_torch.float32)

    # memmaps for GraphCast:
    target_model_dir = output_root / "GraphCast" / "ERA5_6H"
    specs = _rewrite_specs_path(_graphcast_specs(init_tag, surface_only=surface_only), target_model_dir)
    writer = NpyMemmapWriter(specs, shape=(n_leads, 721, 1440))

    date_yyyymmdd = start_dt.strftime("%Y%m%d")
    hour0 = int(start_dt.hour)
    blob0 = load_time_blob(input_root, date_yyyymmdd, hour0)

    state_norm = _torch.from_numpy(_blob_to_graphcast_norm(blob0, cfg, mu, sd)).unsqueeze(0).to(
        device=device, dtype=_torch.float32
    )
    t = start_dt
    dt = int(cfg.dt)
    if dt <= 0:
        raise ValueError(f"invalid cfg.dt={cfg.dt}")
    if max_lead_hours % dt != 0:
        raise ValueError(f"max_lead_hours={max_lead_hours} not divisible by cfg.dt={dt}")
    n_steps = max_lead_hours // dt

    # Map which memmap prefix each channel writes.
    channel_prefixes: List[Optional[str]] = [_gc_channel_to_prefix(ch) for ch in cfg.channels]

    _progress(f"[graphcast] rollout: steps={n_steps}, dt={dt}, writing leads={n_leads}")
    with _torch.no_grad():
        for step in range(1, n_steps + 1):
            invar = _build_graphcast_invar(state_norm, cfg, static_data, latlon_torch, t, device)
            pred = model(invar)
            state_norm = pred
            t = t + timedelta(hours=dt)

            lead_idx = step - 1
            arr = pred.float().cpu().numpy()[0]  # normalized outputs (C,H,W)

            for i, prefix in enumerate(channel_prefixes):
                if prefix is None:
                    continue
                if prefix not in writer.memmaps:
                    continue
                # Denorm per channel.
                den = arr[i] * sd[i] + mu[i]
                writer.write_slice(prefix, lead_idx, den)

            if step % 10 == 0:
                _progress(f"[graphcast] progress step={step}/{n_steps} (lead_idx_written={lead_idx})")

    writer.close()
    _progress(f"[graphcast] done: wrote {n_leads} leads to {target_model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/public/share/aciwgvx1jd/20260324"))
    ap.add_argument("--output-root", type=Path, default=Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h"))
    ap.add_argument("--start-datetime", type=str, default="20260308T12")
    ap.add_argument("--max-lead-hours", type=int, default=240)
    ap.add_argument("--only-models", type=str, default="pangu,graphcast", help="comma: pangu,graphcast")
    ap.add_argument("--surface-only", action="store_true", help="only write surface variables to npy")
    ap.add_argument("--validate", action="store_true", help="validate saved .npy shapes")
    args = ap.parse_args()

    start_dt = parse_start_datetime(args.start_datetime)
    start_list = [start_dt]

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    assigned = [sp for i, sp in enumerate(start_list) if i % world == rank]
    if not assigned:
        _progress(f"[rank={rank}] no assigned start-datetime, exit early.")
        return
    sp = assigned[0]

    only = {x.strip().lower() for x in args.only_models.split(",") if x.strip()}
    run_pangu = ("pangu" in only) or (not only)
    run_graphcast = ("graphcast" in only) or (not only)

    providers = pick_providers("dcu" if run_pangu else "cpu")
    device = None
    if run_graphcast:
        import torch as _torch

        device = _torch.device("cuda:0" if _torch.cuda.is_available() else "cpu")
        # Fix pickle safety for ScalarFloat models.
        _torch.serialization.add_safe_globals([ScalarFloat])

    _progress(f"start={sp.strftime('%Y-%m-%d %H:%M')} UTC only_models={only} providers={providers} device={device}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if run_pangu:
        run_pangu_liuty_export(
            start_dt=sp,
            max_lead_hours=args.max_lead_hours,
            input_root=args.input_root,
            output_root=args.output_root,
            providers=providers,
            surface_only=args.surface_only,
        )
    if run_graphcast:
        run_graphcast_liuty_export(
            start_dt=sp,
            max_lead_hours=args.max_lead_hours,
            input_root=args.input_root,
            output_root=args.output_root,
            device=device,
            surface_only=args.surface_only,
        )

    if args.validate:
        _progress("[validate] validating npy shapes (this may take a while)...")
        n_leads = args.max_lead_hours // 6
        expected = (n_leads, 721, 1440)

        def _check_dir(d: Path) -> None:
            for f in d.glob("*.npy"):
                a = np.load(f, mmap_mode="r")
                if a.shape != expected:
                    raise RuntimeError(f"bad shape: {f} got {a.shape}, expected {expected}")

        if run_pangu:
            _check_dir(args.output_root / "PanGu" / "ERA5_6H")
        if run_graphcast:
            _check_dir(args.output_root / "GraphCast" / "ERA5_6H")
        _progress("[validate] ok")


if __name__ == "__main__":
    main()

