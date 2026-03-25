#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pytz

# Must set per-process visible device BEFORE importing torch.
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    os.environ["ROCR_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["HIP_VISIBLE_DEVICES"] = str(_local_rank)
    os.environ["HSA_VISIBLE_DEVICES"] = str(_local_rank)

import torch

THIS_DIR = Path(__file__).resolve().parent
ZK_ROOT = THIS_DIR.parent
GRAPH_ROOT = ZK_ROOT.parent
sys.path.insert(0, str(ZK_ROOT))
os.chdir(GRAPH_ROOT)

from cepri_loader import PANGU_LEVELS, pack_pangu_onnx
from infer_cepri_onnx import create_session, pangu_one_step, pick_providers
from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.fcn.YParams import YParams
from onescience.utils.graphcast.data_utils import StaticData
from ruamel.yaml.scalarfloat import ScalarFloat

from data_adapter_20260324 import (
    list_available_dates,
    load_time_blob,
    load_truth_blob_for_valid_time,
)
from io_plot_utils import plot_compare, write_step_nc

torch.serialization.add_safe_globals([ScalarFloat])


def _progress(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _level_index(hpa: int) -> int:
    return int(PANGU_LEVELS.index(int(hpa)))


L1000 = _level_index(1000)


def _extract_truth_fields(tb: Dict[str, np.ndarray] | None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if tb is None:
        return None, None
    return tb["pangu_t"][L1000], tb["surface_v10"]


def _build_graphcast_model(cfg, device: torch.device) -> GraphCastNet:
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
    ckpt_path = Path(cfg.checkpoint_dir) / "graphcast_finetune.pth"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dtype=torch.float32).to(device)
    model.eval()
    return model


def _blob_to_graphcast_norm(blob: Dict[str, np.ndarray], cfg, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
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
            stack.append(blob["pangu_z"][_level_index(int(ch.split("_")[-1]))])
        elif ch.startswith("specific_humidity_"):
            stack.append(blob["pangu_q"][_level_index(int(ch.split("_")[-1]))])
        elif ch.startswith("temperature_"):
            stack.append(blob["pangu_t"][_level_index(int(ch.split("_")[-1]))])
        elif ch.startswith("u_component_of_wind_"):
            stack.append(blob["pangu_u"][_level_index(int(ch.split("_")[-1]))])
        elif ch.startswith("v_component_of_wind_"):
            stack.append(blob["pangu_v"][_level_index(int(ch.split("_")[-1]))])
        else:
            raise ValueError(f"unsupported graphcast channel: {ch}")
    raw = np.stack(stack, axis=0).astype(np.float32)
    return (raw - mu[:, None, None]) / np.maximum(sd[:, None, None], 1e-6)


def _build_graphcast_invar(
    norm_state: torch.Tensor,
    cfg,
    static_data: torch.Tensor,
    latlon_torch: torch.Tensor,
    forecast_time: datetime,
    device: torch.device,
) -> torch.Tensor:
    ts = torch.tensor([forecast_time.timestamp()], dtype=torch.float32, device=device)
    cz = cos_zenith_angle(ts, latlon=latlon_torch.to(device)).float()
    cz = torch.squeeze(cz, dim=2)
    cz = torch.clamp(cz, min=0.0) - 1.0 / torch.pi

    doy = float(forecast_time.timetuple().tm_yday)
    tod = forecast_time.hour + forecast_time.minute / 60.0 + forecast_time.second / 3600.0
    ndy = torch.tensor((doy / 365.0) * (np.pi / 2), dtype=torch.float32, device=device)
    ntd = torch.tensor((tod / (24.0 - cfg.dt)) * (np.pi / 2), dtype=torch.float32, device=device)
    sin_dy = torch.sin(ndy).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    cos_dy = torch.cos(ndy).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    sin_td = torch.sin(ntd).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    cos_td = torch.cos(ntd).expand(1, 1, cfg.img_size[0], cfg.img_size[1])
    return torch.cat((norm_state, cz, static_data, sin_dy, cos_dy, sin_td, cos_td), dim=1)


def _shard_dates(dates: List[str]) -> List[str]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return [d for i, d in enumerate(dates) if i % world == rank]


def _set_local_visible_device() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    os.environ["ROCR_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["HIP_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["HSA_VISIBLE_DEVICES"] = str(local_rank)


def _run_one_date_pangu(
    *,
    date: str,
    hour0: int,
    lead_hours: Iterable[int],
    data_root: Path,
    out_dir: Path,
    providers,
) -> None:
    paths = {
        "6h": ZK_ROOT / "pangu" / "pangu_weather_6.onnx",
        "24h": ZK_ROOT / "pangu" / "pangu_weather_24.onnx",
    }
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    sessions = {k: create_session(p, providers) for k, p in paths.items()}

    init_blob = load_time_blob(data_root, date, hour0)
    p_cur, s_cur = pack_pangu_onnx(init_blob)
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.UTC)
    init_s = init_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for lead in lead_hours:
        sess = sessions["24h"] if (lead % 24 == 0) else sessions["6h"]
        p_cur, s_cur = pangu_one_step(sess, p_cur, s_cur)
        valid_dt = init_dt + timedelta(hours=int(lead))
        valid_s = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        h1000_t = p_cur[0, 2, L1000]
        v10 = s_cur[0, 2]
        tb = load_truth_blob_for_valid_time(data_root, valid_dt)
        t_true, v_true = _extract_truth_fields(tb)

        nc_path = out_dir / "nc" / "pangu" / f"lead_{lead:03d}.nc"
        write_step_nc(
            nc_path,
            model="pangu",
            init_time=init_s,
            lead_hours=lead,
            valid_time=valid_s,
            h1000_t=h1000_t,
            v10=v10,
            lat=lat,
            lon=lon,
        )
        plot_compare(
            out_dir / "plots" / "pangu" / f"h1000_t_compare_lead{lead:03d}.png",
            h1000_t,
            t_true,
            title=f"Pangu +{lead}h h1000_t",
            cmap="viridis",
        )
        plot_compare(
            out_dir / "plots" / "pangu" / f"v10_compare_lead{lead:03d}.png",
            v10,
            v_true,
            title=f"Pangu +{lead}h v10",
            cmap="RdBu_r",
        )


def _run_one_date_graphcast(
    *,
    date: str,
    hour0: int,
    lead_hours: Iterable[int],
    data_root: Path,
    out_dir: Path,
    cfg,
    mu: np.ndarray,
    sd: np.ndarray,
    device: torch.device,
) -> None:
    blob0 = load_time_blob(data_root, date, hour0)
    init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.UTC)
    init_s = init_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)

    model = _build_graphcast_model(cfg, device)
    static_data = StaticData(cfg.static_dir, model.latitudes, model.longitudes).get().to(
        device=device, dtype=torch.float32
    )
    latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=cfg.img_size[-2:])
    latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    idx_v10 = cfg.channels.index("10m_v_component_of_wind")
    idx_t1000 = cfg.channels.index("temperature_1000")
    target_set = set(int(x) for x in lead_hours)

    state = torch.from_numpy(_blob_to_graphcast_norm(blob0, cfg, mu, sd)).unsqueeze(0).to(device=device, dtype=torch.float32)
    dt = int(cfg.dt)
    max_lead = max(target_set)
    n_steps = max_lead // dt
    t = init_dt

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            invar = _build_graphcast_invar(state, cfg, static_data, latlon_torch, t, device)
            pred = model(invar)
            state = pred
            t = t + timedelta(hours=dt)
            lead = step * dt
            if lead not in target_set:
                continue

            arr = pred.float().cpu().numpy()[0]
            h1000_t = arr[idx_t1000] * sd[idx_t1000] + mu[idx_t1000]
            v10 = arr[idx_v10] * sd[idx_v10] + mu[idx_v10]
            valid_dt = init_dt + timedelta(hours=int(lead))
            valid_s = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            tb = load_truth_blob_for_valid_time(data_root, valid_dt)
            t_true, v_true = _extract_truth_fields(tb)

            nc_path = out_dir / "nc" / "graphcast" / f"lead_{lead:03d}.nc"
            write_step_nc(
                nc_path,
                model="graphcast",
                init_time=init_s,
                lead_hours=lead,
                valid_time=valid_s,
                h1000_t=h1000_t,
                v10=v10,
                lat=lat,
                lon=lon,
            )
            plot_compare(
                out_dir / "plots" / "graphcast" / f"h1000_t_compare_lead{lead:03d}.png",
                h1000_t,
                t_true,
                title=f"GraphCast +{lead}h h1000_t",
                cmap="viridis",
            )
            plot_compare(
                out_dir / "plots" / "graphcast" / f"v10_compare_lead{lead:03d}.png",
                v10,
                v_true,
                title=f"GraphCast +{lead}h v10",
                cmap="RdBu_r",
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/public/share/aciwgvx1jd/20260324"))
    ap.add_argument("--output-root", type=Path, default=Path("/public/share/aciwgvx1jd/GunDong_Infer_result"))
    ap.add_argument("--start-hour", type=int, default=0)
    ap.add_argument("--max-lead-hours", type=int, default=240)
    ap.add_argument("--lead-step-hours", type=int, default=6)
    ap.add_argument("--device", default="auto", choices=["auto", "dcu", "cuda", "cpu"])
    ap.add_argument("--only-models", default="", help="comma list: pangu,graphcast")
    ap.add_argument("--date-filter", default="", help="comma list dates yyyymmdd")
    args = ap.parse_args()

    _set_local_visible_device()
    dates = list_available_dates(args.input_root)
    if args.date_filter.strip():
        want = {x.strip() for x in args.date_filter.split(",") if x.strip()}
        dates = [d for d in dates if d in want]
    dates = _shard_dates(dates)
    if not dates:
        _progress("no assigned dates, exit")
        return

    leads = list(range(args.lead_step_hours, args.max_lead_hours + 1, args.lead_step_hours))
    only = {x.strip().lower() for x in args.only_models.split(",") if x.strip()}
    run_pangu = (not only) or ("pangu" in only)
    run_graphcast = (not only) or ("graphcast" in only)

    providers = pick_providers(args.device)
    _progress(f"providers={providers}")
    _progress(f"assigned_dates={dates}")

    cfg = YParams(str(GRAPH_ROOT / "conf/config.yaml"), "model")
    meta_path = Path(cfg.data_dir) / "metadata.json"
    variables = json.load(open(meta_path, encoding="utf-8"))["variables"]
    channel_indices = [variables.index(c) for c in cfg.channels]
    mu_full = np.load(Path(cfg.stats_dir) / "global_means.npy")
    sd_full = np.load(Path(cfg.stats_dir) / "global_stds.npy")
    mu = mu_full[0, channel_indices, 0, 0].astype(np.float32)
    sd = sd_full[0, channel_indices, 0, 0].astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for d in dates:
        init_tag = f"{d}T{args.start_hour:02d}"
        out_dir = args.output_root / init_tag
        _progress(f"start {init_tag}")
        if run_pangu:
            _run_one_date_pangu(
                date=d,
                hour0=args.start_hour,
                lead_hours=leads,
                data_root=args.input_root,
                out_dir=out_dir,
                providers=providers,
            )
            _progress(f"{init_tag} pangu done")
        if run_graphcast:
            _run_one_date_graphcast(
                date=d,
                hour0=args.start_hour,
                lead_hours=leads,
                data_root=args.input_root,
                out_dir=out_dir,
                cfg=cfg,
                mu=mu,
                sd=sd,
                device=device,
            )
            _progress(f"{init_tag} graphcast done")
        _progress(f"finished {init_tag}")


if __name__ == "__main__":
    main()

