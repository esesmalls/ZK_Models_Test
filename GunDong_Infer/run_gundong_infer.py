#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
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

import torch

THIS_DIR = Path(__file__).resolve().parent
ZK_ROOT = THIS_DIR.parent
GRAPH_ROOT = ZK_ROOT.parent
sys.path.insert(0, str(ZK_ROOT))
os.chdir(GRAPH_ROOT)

from cepri_loader import FUXI_LEVELS, PANGU_LEVELS, specific_humidity_to_relative_humidity, pack_pangu_onnx
from infer_cepri_onnx import (
    build_fengwu_onnx_combo_input,
    create_session,
    fengwu_denorm_chw,
    fengwu_normalize_for_onnx,
    fuxi_prepare_onnx_input,
    fuxi_temb,
    pangu_one_step,
    pick_providers,
    unpack_fengwu_ort_outputs,
)
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


def _parse_start_datetime(s: str) -> datetime:
    if "T" not in s:
        raise ValueError(f"invalid start datetime {s}, expected YYYYMMDDTHH")
    d, h = s.split("T", 1)
    if len(d) != 8 or len(h) != 2:
        raise ValueError(f"invalid start datetime {s}, expected YYYYMMDDTHH")
    return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]), int(h), tzinfo=pytz.UTC)


def _init_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H")


def _fwfx_surface_writer_paths(output_root: Path, model_dir_name: str, init_dt: datetime) -> Dict[str, Path]:
    base = output_root / model_dir_name / "ERA5_6H"
    tag = _init_tag(init_dt)
    return {
        "msl": base / f"msl_{tag}.npy",
        "t2m": base / f"t2m_{tag}.npy",
        "u10": base / f"u10_{tag}.npy",
        "v10": base / f"v10_{tag}.npy",
    }


def _save_surface_stacks(paths: Dict[str, Path], stacks: Dict[str, List[np.ndarray]]) -> None:
    for k, p in paths.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(stacks[k], axis=0).astype(np.float32)
        np.save(p, arr)


def _plot_surface_compares(
    *,
    model_name: str,
    out_dir: Path,
    init_dt: datetime,
    lead: int,
    pred: Dict[str, np.ndarray],
    truth_blob: Dict[str, np.ndarray] | None,
) -> None:
    tag = _init_tag(init_dt)
    plot_dir = out_dir / "plots" / model_name / tag
    plot_compare(
        plot_dir / f"msl_compare_lead{lead:03d}.png",
        pred["msl"],
        None if truth_blob is None else truth_blob["surface_msl"],
        title=f"{model_name} +{lead}h msl",
        cmap="viridis",
    )
    plot_compare(
        plot_dir / f"t2m_compare_lead{lead:03d}.png",
        pred["t2m"],
        None if truth_blob is None else truth_blob["surface_t2m"],
        title=f"{model_name} +{lead}h t2m",
        cmap="viridis",
    )
    plot_compare(
        plot_dir / f"u10_compare_lead{lead:03d}.png",
        pred["u10"],
        None if truth_blob is None else truth_blob["surface_u10"],
        title=f"{model_name} +{lead}h u10",
        cmap="RdBu_r",
    )
    plot_compare(
        plot_dir / f"v10_compare_lead{lead:03d}.png",
        pred["v10"],
        None if truth_blob is None else truth_blob["surface_v10"],
        title=f"{model_name} +{lead}h v10",
        cmap="RdBu_r",
    )


def _fengwu_69_from_blob_q_order(blob: Dict[str, np.ndarray]) -> np.ndarray:
    # [u10,v10,t2m,msl,z(50..1000),q(50..1000),u(50..1000),v(50..1000),t(50..1000)]
    levels_src = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    order = [levels_src.index(int(lv)) for lv in FUXI_LEVELS]  # 50..1000
    sfc = np.stack(
        [blob["surface_u10"], blob["surface_v10"], blob["surface_t2m"], blob["surface_msl"]],
        axis=0,
    ).astype(np.float32)
    z = blob["pangu_z"][order].astype(np.float32)
    q = blob["pangu_q"][order].astype(np.float32)
    u = blob["pangu_u"][order].astype(np.float32)
    v = blob["pangu_v"][order].astype(np.float32)
    t = blob["pangu_t"][order].astype(np.float32)
    return np.concatenate([sfc, z, q, u, v, t], axis=0).astype(np.float32)


def _fuxi_frame70_from_blob(blob: Dict[str, np.ndarray], *, tp_fill: float = 0.0) -> np.ndarray:
    # FuXi order: Z13, T13, U13, V13, R13, T2M, U10, V10, MSL, TP
    levels_src = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    order = [levels_src.index(int(lv)) for lv in FUXI_LEVELS]  # 50..1000
    z13 = blob["pangu_z"][order].astype(np.float32)
    t13 = blob["pangu_t"][order].astype(np.float32)
    u13 = blob["pangu_u"][order].astype(np.float32)
    v13 = blob["pangu_v"][order].astype(np.float32)
    q13 = blob["pangu_q"][order].astype(np.float32)
    r13 = np.empty_like(q13, dtype=np.float32)
    for i, lev in enumerate(FUXI_LEVELS):
        r13[i] = specific_humidity_to_relative_humidity(q13[i], t13[i], float(lev))
    s5 = np.stack(
        [
            blob["surface_t2m"],
            blob["surface_u10"],
            blob["surface_v10"],
            blob["surface_msl"],
            np.full_like(blob["surface_msl"], tp_fill, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    upper = np.concatenate([z13, t13, u13, v13, r13], axis=0)
    return np.concatenate([upper, s5], axis=0).astype(np.float32)


def _run_one_date_fengwu(
    *,
    date: str,
    hour0: int,
    lead_hours: Iterable[int],
    data_root: Path,
    output_root: Path,
    providers,
    skip_plots: bool,
    model_version: str,
    stats_dir: Optional[Path],
) -> None:
    onnx_p = ZK_ROOT / "fengwu" / f"fengwu_{model_version}.onnx"
    if not onnx_p.is_file():
        raise FileNotFoundError(onnx_p)
    sess = create_session(onnx_p, providers)
    inps = sess.get_inputs()
    if len(inps) != 1:
        raise RuntimeError("fengwu expects single tensor input")

    start_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.UTC)
    tag = _init_tag(start_dt)
    n_leads = len(list(lead_hours))
    surfaces: Dict[str, List[np.ndarray]] = {k: [] for k in ("msl", "t2m", "u10", "v10")}

    # Build semantic +6 first lead input by using (t-6h, t0) for 138-ch models.
    exp_c = sess.get_inputs()[0].shape[1]
    if isinstance(exp_c, int) and exp_c == 138:
        prev_dt = start_dt - timedelta(hours=6)
        b_prev = load_time_blob(data_root, prev_dt.strftime("%Y%m%d"), prev_dt.hour)
        b_now = load_time_blob(data_root, date, hour0)
        x = np.concatenate([_fengwu_69_from_blob_q_order(b_prev), _fengwu_69_from_blob_q_order(b_now)], axis=0)[
            np.newaxis, ...
        ].astype(np.float32)
    else:
        # 37-level single-frame path is not available in data_adapter_20260324 layout.
        x = build_fengwu_onnx_combo_input(data_root, date, hour0, sess)

    normalized = False
    if stats_dir is not None:
        x = fengwu_normalize_for_onnx(x, stats_dir)
        normalized = True
    cur = x.astype(np.float32)

    for i, lead in enumerate(lead_hours, start=1):
        outs = sess.run(None, {inps[0].name: cur})
        surf_o, _, _, _, _, _ = unpack_fengwu_ort_outputs(outs)
        pred69 = None
        if len(outs) == 1:
            yo = np.asarray(outs[0], dtype=np.float32)
            while yo.ndim > 3 and yo.shape[0] == 1:
                yo = yo[0]
            if yo.ndim == 3 and yo.shape[0] >= 69:
                pred69 = yo[:69]
        if pred69 is not None and normalized and stats_dir is not None:
            pred69 = fengwu_denorm_chw(pred69, stats_dir)
            surf_o = pred69[:4]

        pred = {
            "u10": np.asarray(surf_o[0], dtype=np.float32),
            "v10": np.asarray(surf_o[1], dtype=np.float32),
            "t2m": np.asarray(surf_o[2], dtype=np.float32),
            "msl": np.asarray(surf_o[3], dtype=np.float32),
        }
        for k in surfaces:
            surfaces[k].append(pred[k])

        valid_dt = start_dt + timedelta(hours=int(lead))
        truth = load_truth_blob_for_valid_time(data_root, valid_dt)
        if not skip_plots:
            _plot_surface_compares(
                model_name="fengwu",
                out_dir=output_root,
                init_dt=start_dt,
                lead=int(lead),
                pred=pred,
                truth_blob=truth,
            )

        # autoregressive update
        if cur.shape[1] == 138 and pred69 is not None:
            if normalized and stats_dir is not None:
                pred69_norm = fengwu_normalize_for_onnx(pred69[np.newaxis, ...], stats_dir)[0]
            else:
                pred69_norm = pred69
            cur = np.concatenate([cur[:, 69:], pred69_norm[np.newaxis, ...]], axis=1).astype(np.float32)
        elif len(outs) == 1 and isinstance(np.asarray(outs[0]).shape[1] if np.asarray(outs[0]).ndim > 1 else None, int):
            ny = np.asarray(outs[0], dtype=np.float32)
            if ny.shape == cur.shape:
                cur = ny.astype(np.float32)
        if i % 8 == 0 or i == 1:
            _progress(f"[fengwu] {tag} lead={lead}h done")

    paths = _fwfx_surface_writer_paths(output_root, "FengWu", start_dt)
    _save_surface_stacks(paths, surfaces)


def _run_one_date_fuxi(
    *,
    date: str,
    hour0: int,
    lead_hours: Iterable[int],
    data_root: Path,
    output_root: Path,
    providers,
    skip_plots: bool,
) -> None:
    onnx_p = ZK_ROOT / "fuxi" / "short.onnx"
    if not onnx_p.is_file():
        raise FileNotFoundError(onnx_p)
    sess = create_session(onnx_p, providers)

    # Match zforecast semantics: two frames are t-6h and t0.
    prev_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.UTC) - timedelta(hours=6)
    b_prev = load_time_blob(data_root, prev_dt.strftime("%Y%m%d"), prev_dt.hour)
    b_now = load_time_blob(data_root, date, hour0)
    raw = np.stack([_fuxi_frame70_from_blob(b_prev), _fuxi_frame70_from_blob(b_now)], axis=0).astype(np.float32)
    x, layout = fuxi_prepare_onnx_input(raw, sess)
    cur = x.astype(np.float32)

    start_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.UTC)
    tag = _init_tag(start_dt)
    surfaces: Dict[str, List[np.ndarray]] = {k: [] for k in ("msl", "t2m", "u10", "v10")}

    for i, lead in enumerate(lead_hours, start=1):
        feeds = {}
        for inp in sess.get_inputs():
            if "temb" in inp.name.lower():
                feeds[inp.name] = fuxi_temb(int(lead))
            else:
                feeds[inp.name] = cur
        y = sess.run(None, feeds)[0]
        cur = y.astype(np.float32)
        if y.ndim == 5 and layout == "NTCHW":
            out_latest = y[0, -1]
        elif y.ndim == 5 and layout == "NCTHW":
            out_latest = y[0, :, -1]
        elif y.ndim == 4:
            out_latest = y[0]
        else:
            raise RuntimeError(f"unexpected fuxi output shape: {y.shape}")

        # FuXi channel order: ... T2M(65), U10(66), V10(67), MSL(68), TP(69)
        pred = {
            "t2m": np.asarray(out_latest[65], dtype=np.float32),
            "u10": np.asarray(out_latest[66], dtype=np.float32),
            "v10": np.asarray(out_latest[67], dtype=np.float32),
            "msl": np.asarray(out_latest[68], dtype=np.float32),
        }
        for k in surfaces:
            surfaces[k].append(pred[k])

        valid_dt = start_dt + timedelta(hours=int(lead))
        truth = load_truth_blob_for_valid_time(data_root, valid_dt)
        if not skip_plots:
            _plot_surface_compares(
                model_name="fuxi",
                out_dir=output_root,
                init_dt=start_dt,
                lead=int(lead),
                pred=pred,
                truth_blob=truth,
            )
        if i % 8 == 0 or i == 1:
            _progress(f"[fuxi] {tag} lead={lead}h done")

    paths = _fwfx_surface_writer_paths(output_root, "FuXi", start_dt)
    _save_surface_stacks(paths, surfaces)


def _run_one_date_pangu(
    *,
    date: str,
    hour0: int,
    lead_hours: Iterable[int],
    data_root: Path,
    out_dir: Path,
    providers,
    skip_plots: bool,
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
        vars_2d = {
            "surface_msl": s_cur[0, 0],
            "surface_u10": s_cur[0, 1],
            "surface_v10": s_cur[0, 2],
            "surface_t2m": s_cur[0, 3],
        }
        vars_3d = {
            "pressure_z": p_cur[0, 0],
            "pressure_q": p_cur[0, 1],
            "pressure_t": p_cur[0, 2],
            "pressure_u": p_cur[0, 3],
            "pressure_v": p_cur[0, 4],
        }
        write_step_nc(
            nc_path,
            model="pangu",
            init_time=init_s,
            lead_hours=lead,
            valid_time=valid_s,
            vars_2d=vars_2d,
            vars_3d=vars_3d,
            level_values=np.asarray(PANGU_LEVELS, dtype=np.float32),
            lat=lat,
            lon=lon,
        )
        if not skip_plots:
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
    skip_plots: bool,
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
            vars_2d = {}
            for i, ch in enumerate(cfg.channels):
                k = ch.lower().replace("-", "_").replace(" ", "_")
                vars_2d[f"gc_{k}"] = arr[i] * sd[i] + mu[i]
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
                vars_2d=vars_2d,
                lat=lat,
                lon=lon,
            )
            if not skip_plots:
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
    ap.add_argument("--only-models", default="", help="comma list: pangu,graphcast,fengwu,fuxi")
    ap.add_argument("--date-filter", default="", help="comma list dates yyyymmdd")
    ap.add_argument("--skip-plots", action="store_true", help="only write nc, do not generate plots")
    ap.add_argument("--single-start-datetime", default="", help="run one init: YYYYMMDDTHH (overrides date-filter/start-hour)")
    ap.add_argument("--fengwu-model-version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--fengwu-stats-dir", type=Path, default=ZK_ROOT / "fengwu")
    args = ap.parse_args()

    _set_local_visible_device()
    if args.single_start_datetime.strip():
        sdt = _parse_start_datetime(args.single_start_datetime.strip())
        dates = [sdt.strftime("%Y%m%d")]
        args.start_hour = int(sdt.hour)
    else:
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
    run_fengwu = (not only) or ("fengwu" in only)
    run_fuxi = (not only) or ("fuxi" in only)

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
                skip_plots=args.skip_plots,
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
                skip_plots=args.skip_plots,
            )
            _progress(f"{init_tag} graphcast done")
        if run_fengwu:
            _run_one_date_fengwu(
                date=d,
                hour0=args.start_hour,
                lead_hours=leads,
                data_root=args.input_root,
                output_root=Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h"),
                providers=providers,
                skip_plots=args.skip_plots,
                model_version=args.fengwu_model_version,
                stats_dir=args.fengwu_stats_dir,
            )
            _progress(f"{init_tag} fengwu done")
        if run_fuxi:
            _run_one_date_fuxi(
                date=d,
                hour0=args.start_hour,
                lead_hours=leads,
                data_root=args.input_root,
                output_root=Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h"),
                providers=providers,
                skip_plots=args.skip_plots,
            )
            _progress(f"{init_tag} fuxi done")
        _progress(f"finished {init_tag}")


if __name__ == "__main__":
    main()

