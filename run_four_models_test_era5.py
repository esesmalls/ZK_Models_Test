#!/usr/bin/env python3
"""
使用 graphcast/test_era5_data 下的日尺度 NetCDF，对 GraphCast、Pangu、Fengwu、Fuxi 做短时多步推理并保存 PNG。

请在 graphcast 目录下运行（保证 conf、result 路径正确）:
  cd .../graphcast
  python ZK_Models/run_four_models_test_era5.py --num-steps 3

出图变量：地面 u10,v10,t2m,msl + 1000hPa 的 z,q,u,v,t；对比图 `*_compare.png` 三联（英文子图标题避免乱码）。
Fengwu/Fuxi 在 1000hPa 用相对湿度 r（非比湿 q）；对比时与 ERA5 由 q,T 推算的 RH 对齐。
GraphCast/Pangu 仍按 `--hour` 起报。Fengwu：189 通道取当小时分析场；138 双帧为 **(h,h+1)**。
Fuxi short 两帧为 **(h,h+1)**，时间嵌入为简易 `fuxi_temb`。
对比图：`--fengwu-compare-lead-hours` / `--fuxi-compare-lead-hours`（默认 6）相对 `--hour` 起报时刻取同期 ERA5。
`--no-compare` 可关闭对比图。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _progress(phase: str, detail: str = "") -> None:
    """Slurm/重定向下用 flush，便于 tail -f 日志看到执行到哪一步。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [progress] {phase}"
    if detail:
        line += f" | {detail}"
    print(line, flush=True)

import numpy as np

ZK_ROOT = Path(__file__).resolve().parent
GRAPH_ROOT = ZK_ROOT.parent
sys.path.insert(0, str(ZK_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 避免标题中文缺字显示为方块；并尽量选常见 CJK 字体
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]

import pytz
import torch

from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.fcn.YParams import YParams
from onescience.utils.graphcast.data_utils import StaticData
from ruamel.yaml.scalarfloat import ScalarFloat

torch.serialization.add_safe_globals([ScalarFloat])

from cepri_loader import (
    FENGWU_LEVELS,
    FUXI_LEVELS,
    PANGU_LEVELS,
    load_cepri_fuxi_fields,
    load_cepri_time,
    pack_pangu_onnx,
    specific_humidity_to_relative_humidity,
)
from infer_cepri_onnx import (
    build_fengwu_onnx_combo_input,
    create_session,
    fengwu_denorm_chw,
    fengwu_normalize_for_onnx,
    fuxi_normalize_for_layout,
    fuxi_prepare_onnx_input,
    pangu_one_step,
    pick_providers,
    unpack_fengwu_ort_outputs,
)


def _plot_map(path: Path, arr: np.ndarray, title: str, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="upper")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fuxi_temb_like_zforecast(start_time_utc: datetime, step_idx_1based: int, freq_hours: int) -> np.ndarray:
    """
    Match zforecast.py style:
      times = init_time + [i-1, i, i+1] * freq_hours
      features = [(day_of_year/366, hour/24)] * 3
      temb = concat(sin(features), cos(features)) -> shape (1, 12)
    """
    i = int(step_idx_1based)
    times = [start_time_utc + timedelta(hours=freq_hours * k) for k in (i - 1, i, i + 1)]
    feats: list[float] = []
    for t in times:
        feats.extend([float(t.timetuple().tm_yday) / 366.0, float(t.hour) / 24.0])
    x = np.asarray(feats, dtype=np.float32)
    return np.concatenate([np.sin(x), np.cos(x)], axis=0)[np.newaxis, :]


def _field_diag(tag: str, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    n_total = int(a.size)
    n_finite = int(np.count_nonzero(finite))
    if n_finite == 0:
        print(f"[{tag}] {name}: all values are non-finite", flush=True)
        return
    af = a[finite]
    print(
        f"[{tag}] {name}: min={float(np.min(af)):.3e} max={float(np.max(af)):.3e} "
        f"mean={float(np.mean(af)):.3e} std={float(np.std(af)):.3e} "
        f"nan_or_inf={n_total - n_finite}",
        flush=True,
    )


def load_era5_truth_blob(era5_root: Path, when_utc: datetime) -> dict | None:
    """读取与预报有效时刻同一小时（整点）的 ERA5 分析场；缺文件则返回 None。"""
    when_utc = when_utc.astimezone(pytz.UTC)
    ds = when_utc.strftime("%Y%m%d")
    hr = int(when_utc.hour)
    try:
        return load_cepri_time(era5_root, ds, hr)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        print(f"[compare] 无同期 ERA5: {ds} {hr:02d}Z — {e}", flush=True)
        return None


def _level_index(hpa: int) -> int:
    return PANGU_LEVELS.index(int(hpa))


# 与 PANGU_LEVELS 一致：1000 hPa 最接近地表，在 13 层栈中为下标 0
L1000 = _level_index(1000)
FUXI_I1000 = FUXI_LEVELS.index(1000)


def _truth_from_era5_blob(blob: dict, field_id: str) -> np.ndarray:
    """与 `load_cepri_time` 的 pangu_* / surface_* 对齐，用于对比图。"""
    if field_id == "sfc_u10":
        return np.asarray(blob["surface_u10"], dtype=np.float64)
    if field_id == "sfc_v10":
        return np.asarray(blob["surface_v10"], dtype=np.float64)
    if field_id == "sfc_t2m":
        return np.asarray(blob["surface_t2m"], dtype=np.float64)
    if field_id == "sfc_msl":
        return np.asarray(blob["surface_msl"], dtype=np.float64)
    if field_id == "h1000_z":
        return np.asarray(blob["pangu_z"][L1000], dtype=np.float64)
    if field_id == "h1000_q":
        return np.asarray(blob["pangu_q"][L1000], dtype=np.float64)
    if field_id == "h1000_t":
        return np.asarray(blob["pangu_t"][L1000], dtype=np.float64)
    if field_id == "h1000_u":
        return np.asarray(blob["pangu_u"][L1000], dtype=np.float64)
    if field_id == "h1000_v":
        return np.asarray(blob["pangu_v"][L1000], dtype=np.float64)
    if field_id == "h1000_rh":
        return np.asarray(
            specific_humidity_to_relative_humidity(
                blob["pangu_q"][L1000], blob["pangu_t"][L1000], 1000.0
            ),
            dtype=np.float64,
        )
    raise ValueError(f"unknown field_id: {field_id}")


def _plot_model_vs_era5(
    path: Path,
    pred: np.ndarray,
    truth_blob: dict | None,
    suptitle_en: str,
    *,
    field_id: str,
) -> None:
    """3-panel: forecast | ERA5 analysis | difference. Titles in ASCII to avoid font tofu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred, dtype=np.float64)
    if truth_blob is None:
        _plot_map(path, pred.astype(np.float32), f"{suptitle_en} (no ERA5 analysis)")
        return
    truth = _truth_from_era5_blob(truth_blob, field_id)
    diff = pred - truth
    vmin = float(min(pred.min(), truth.min()))
    vmax = float(max(pred.max(), truth.max()))
    dv = float(np.percentile(np.abs(diff), 99))
    if not np.isfinite(dv) or dv < 1e-9:
        dv = max(float(np.abs(diff).max()), 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axs[0].imshow(pred, cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[0].set_title("Forecast")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)
    im1 = axs[1].imshow(truth, cmap="viridis", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[1].set_title("ERA5 analysis (same time)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)
    im2 = axs[2].imshow(diff, cmap="RdBu_r", aspect="auto", origin="upper", vmin=-dv, vmax=dv)
    axs[2].set_title("Forecast minus analysis")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)
    fig.suptitle(suptitle_en)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fengwu_level_index_1000(n_levels: int) -> int:
    if n_levels == len(FENGWU_LEVELS):
        return int(FENGWU_LEVELS.index(1000))
    if n_levels == len(PANGU_LEVELS):
        return L1000
    return 0


def _plot_input_standard_fields(blob: dict, out_plot: Path, tag: str) -> None:
    """起报时刻：sfc u10,v10,t2m,msl + 1000hPa z,q,u,v,t"""
    fields: list[tuple[str, np.ndarray, str]] = [
        ("sfc_u10", blob["surface_u10"], "10m u-wind (m/s)"),
        ("sfc_v10", blob["surface_v10"], "10m v-wind (m/s)"),
        ("sfc_t2m", blob["surface_t2m"], "2m temperature (K)"),
        ("sfc_msl", blob["surface_msl"], "mean sea level pressure (Pa)"),
        ("h1000_z", blob["pangu_z"][L1000], "1000 hPa geopotential (m2/s2)"),
        ("h1000_q", blob["pangu_q"][L1000], "1000 hPa specific humidity (kg/kg)"),
        ("h1000_u", blob["pangu_u"][L1000], "1000 hPa u-wind (m/s)"),
        ("h1000_v", blob["pangu_v"][L1000], "1000 hPa v-wind (m/s)"),
        ("h1000_t", blob["pangu_t"][L1000], "1000 hPa temperature (K)"),
    ]
    for key, arr, label in fields:
        _plot_map(
            out_plot / f"{tag}_{key}.png",
            arr,
            f"{tag} analysis {label}",
            cmap="RdBu_r" if "u-wind" in label or "v-wind" in label else "viridis",
        )


def blob_to_graphcast_norm(blob: dict, cfg, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    stack: list[np.ndarray] = []
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
            raise ValueError(f"未实现的 GraphCast 通道: {ch}")
    raw = np.stack(stack, axis=0).astype(np.float32)
    return (raw - mu[:, np.newaxis, np.newaxis]) / np.maximum(sd[:, np.newaxis, np.newaxis], 1e-6)


def build_graphcast_invar(
    norm_climate: torch.Tensor,
    cfg,
    static_data: torch.Tensor,
    latlon_torch: torch.Tensor,
    forecast_time: datetime,
    device: torch.device,
    model_dtype: torch.dtype,
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

    return torch.cat(
        (norm_climate.to(device=device, dtype=model_dtype), cz, static_data, sin_dy, cos_dy, sin_td, cos_td),
        dim=1,
    )


def run_graphcast_rollout(
    cfg,
    blob0: dict,
    mu: np.ndarray,
    sd: np.ndarray,
    num_steps: int,
    start_time: datetime,
    out_plot: Path,
    device: torch.device,
    era5_root: Path,
    do_compare: bool,
) -> None:
    ckpt_path = Path(cfg.checkpoint_dir) / "graphcast_finetune.pth"
    if not ckpt_path.is_file():
        print(f"[graphcast] 跳过：无权重 {ckpt_path}")
        return

    _progress("GraphCast", "构建 GraphCastNet …")
    model_dtype = torch.float32
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
    _progress("GraphCast", f"加载权重 {ckpt_path.name} …")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dtype=model_dtype).to(device)
    model.eval()

    _progress("GraphCast", "StaticData(lsm 等) …")
    static_data = StaticData(cfg.static_dir, model.latitudes, model.longitudes).get().to(
        device=device, dtype=model_dtype
    )

    latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=cfg.img_size[-2:])
    latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    gc_plot_idx: dict[str, int] = {}
    for short, full in (
        ("u10", "10m_u_component_of_wind"),
        ("v10", "10m_v_component_of_wind"),
        ("t2m", "2m_temperature"),
        ("msl", "mean_sea_level_pressure"),
        ("z1000", "geopotential_1000"),
        ("q1000", "specific_humidity_1000"),
        ("t1000", "temperature_1000"),
        ("u1000", "u_component_of_wind_1000"),
        ("v1000", "v_component_of_wind_1000"),
    ):
        if full in cfg.channels:
            gc_plot_idx[short] = cfg.channels.index(full)
        else:
            print(f"[graphcast] skip plot (not in cfg.channels): {full}", flush=True)

    gc_plot_rows: list[tuple[str, str, str, str]] = [
        ("sfc_u10", "u10", "10m u-wind (m/s)", "RdBu_r"),
        ("sfc_v10", "v10", "10m v-wind (m/s)", "RdBu_r"),
        ("sfc_t2m", "t2m", "2m temperature (K)", "viridis"),
        ("sfc_msl", "msl", "MSL pressure (Pa)", "viridis"),
        ("h1000_z", "z1000", "1000 hPa geopotential (m2/s2)", "viridis"),
        ("h1000_q", "q1000", "1000 hPa specific humidity (kg/kg)", "viridis"),
        ("h1000_t", "t1000", "1000 hPa temperature (K)", "viridis"),
        ("h1000_u", "u1000", "1000 hPa u-wind (m/s)", "RdBu_r"),
        ("h1000_v", "v1000", "1000 hPa v-wind (m/s)", "RdBu_r"),
    ]

    state_norm = torch.from_numpy(blob_to_graphcast_norm(blob0, cfg, mu, sd)).unsqueeze(0).to(
        device=device, dtype=model_dtype
    )
    t = start_time
    dt_hours = int(cfg.dt)

    _progress("GraphCast", f"开始 autoregressive rollout，共 {num_steps} 步 (dt={dt_hours}h)")
    with torch.no_grad():
        for step in range(1, num_steps + 1):
            _progress("GraphCast", f"步 {step}/{num_steps}：拼 invar + forward …")
            invar = build_graphcast_invar(state_norm, cfg, static_data, latlon_torch, t, device, model_dtype)
            pred = model(invar)
            state_norm = pred
            t = t + timedelta(hours=dt_hours)

            pr = pred.float().cpu().numpy()[0]
            h = step * dt_hours
            valid_t = start_time + timedelta(hours=h)
            tb = load_era5_truth_blob(era5_root, valid_t) if do_compare else None
            for fid, idxkey, label, cmap in gc_plot_rows:
                if idxkey not in gc_plot_idx:
                    continue
                i = gc_plot_idx[idxkey]
                arr = pr[i] * sd[i] + mu[i]
                _plot_map(
                    out_plot / f"graphcast_step{step:02d}_{fid}.png",
                    arr,
                    f"GraphCast +{h}h forecast: {label}",
                    cmap=cmap,
                )
                if do_compare:
                    _plot_model_vs_era5(
                        out_plot / f"graphcast_step{step:02d}_{fid}_compare.png",
                        arr,
                        tb,
                        f"GraphCast +{h}h vs ERA5 {valid_t.strftime('%Y-%m-%d %H')}Z | {label}",
                        field_id=fid,
                    )
            _progress("GraphCast", f"步 {step}/{num_steps} 完成（已写 PNG）")
    print(f"[graphcast] 已写 {num_steps} 步图 -> {out_plot}")


def run_pangu_rollout(
    era5_root: Path,
    date_yyyymmdd: str,
    hour0: int,
    num_steps: int,
    providers,
    out_plot: Path,
    do_compare: bool,
) -> None:
    paths = {
        "1h": ZK_ROOT / "pangu" / "pangu_weather_1.onnx",
        "6h": ZK_ROOT / "pangu" / "pangu_weather_6.onnx",
        "24h": ZK_ROOT / "pangu" / "pangu_weather_24.onnx",
    }
    if not all(p.is_file() for p in paths.values()):
        print("[pangu] 跳过：ONNX 不全")
        return
    has_3h = (ZK_ROOT / "pangu" / "pangu_weather_3.onnx").is_file()
    if has_3h:
        paths["3h"] = ZK_ROOT / "pangu" / "pangu_weather_3.onnx"
    _progress("Pangu", f"加载 ONNX 会话 {list(paths.keys())}（可能较慢、ORT 优化图）…")
    sessions = {k: create_session(p, providers) for k, p in paths.items()}
    _progress("Pangu", "ONNX 会话就绪")

    blob = load_cepri_time(era5_root, date_yyyymmdd, hour0)
    p_in, s_in = pack_pangu_onnx(blob)
    cur_p, cur_s = p_in.copy(), s_in.copy()
    c1p, c1s = cur_p.copy(), cur_s.copy()
    c3p, c3s = cur_p.copy(), cur_s.copy()
    c6p, c6s = cur_p.copy(), cur_s.copy()
    c24p, c24s = cur_p.copy(), cur_s.copy()

    pangu_plot_ids = [
        "sfc_u10",
        "sfc_v10",
        "sfc_t2m",
        "sfc_msl",
        "h1000_z",
        "h1000_q",
        "h1000_t",
        "h1000_u",
        "h1000_v",
    ]
    base_t = datetime(
        int(date_yyyymmdd[:4]),
        int(date_yyyymmdd[4:6]),
        int(date_yyyymmdd[6:8]),
        hour0,
        tzinfo=pytz.utc,
    )
    _progress("Pangu", f"开始 {num_steps} 步积分 …")
    for step in range(1, num_steps + 1):
        _progress("Pangu", f"步 {step}/{num_steps}：选子模型并 sess.run …")
        use_24h = step % 24 == 0
        use_6h = (not use_24h) and (step % 6 == 0)
        use_3h = (not use_24h) and (not use_6h) and (step % 3 == 0)
        if use_24h:
            op, os_ = pangu_one_step(sessions["24h"], c24p, c24s)
        elif use_6h:
            op, os_ = pangu_one_step(sessions["6h"], c6p, c6s)
        elif use_3h and has_3h:
            op, os_ = pangu_one_step(sessions["3h"], c3p, c3s)
        elif use_3h:
            op, os_ = pangu_one_step(sessions["1h"], c1p, c1s)
        else:
            op, os_ = pangu_one_step(sessions["1h"], c1p, c1s)

        s = os_[0]
        pr = op[0]
        valid_t = base_t + timedelta(hours=step)
        tb = load_era5_truth_blob(era5_root, valid_t) if do_compare else None
        pangu_arr = {
            "sfc_u10": s[1],
            "sfc_v10": s[2],
            "sfc_t2m": s[3],
            "sfc_msl": s[0],
            "h1000_z": pr[0, L1000],
            "h1000_q": pr[1, L1000],
            "h1000_t": pr[2, L1000],
            "h1000_u": pr[3, L1000],
            "h1000_v": pr[4, L1000],
        }
        labels = {
            "sfc_u10": ("10m u-wind (m/s)", "RdBu_r"),
            "sfc_v10": ("10m v-wind (m/s)", "RdBu_r"),
            "sfc_t2m": ("2m temperature (K)", "viridis"),
            "sfc_msl": ("MSL pressure (Pa)", "viridis"),
            "h1000_z": ("1000 hPa geopotential (m2/s2)", "viridis"),
            "h1000_q": ("1000 hPa specific humidity (kg/kg)", "viridis"),
            "h1000_t": ("1000 hPa temperature (K)", "viridis"),
            "h1000_u": ("1000 hPa u-wind (m/s)", "RdBu_r"),
            "h1000_v": ("1000 hPa v-wind (m/s)", "RdBu_r"),
        }
        for fid in pangu_plot_ids:
            arr = pangu_arr[fid]
            lab, cmap = labels[fid]
            _plot_map(
                out_plot / f"pangu_step{step:02d}_{fid}.png",
                arr,
                f"Pangu step {step} (+{step}h assumed): {lab}",
                cmap=cmap,
            )
            if do_compare:
                _plot_model_vs_era5(
                    out_plot / f"pangu_step{step:02d}_{fid}_compare.png",
                    arr,
                    tb,
                    f"Pangu step {step} vs ERA5 {valid_t.strftime('%Y-%m-%d %H')}Z | {lab}",
                    field_id=fid,
                )
        _progress("Pangu", f"步 {step}/{num_steps} 完成")

        cur_p, cur_s = op, os_
        # Keep step-specific streams aligned with the model actually used at this step.
        if use_24h:
            c1p, c1s = cur_p.copy(), cur_s.copy()
            c3p, c3s = cur_p.copy(), cur_s.copy()
            c6p, c6s = cur_p.copy(), cur_s.copy()
            c24p, c24s = cur_p.copy(), cur_s.copy()
        elif use_6h:
            c1p, c1s = cur_p.copy(), cur_s.copy()
            c3p, c3s = cur_p.copy(), cur_s.copy()
            c6p, c6s = cur_p.copy(), cur_s.copy()
        elif use_3h:
            c1p, c1s = cur_p.copy(), cur_s.copy()
            c3p, c3s = cur_p.copy(), cur_s.copy()
        else:
            c1p, c1s = cur_p.copy(), cur_s.copy()
    print(f"[pangu] 已写 {num_steps} 步图 -> {out_plot}")


def run_fengwu_once(
    era5_root: Path,
    date_yyyymmdd: str,
    hour0: int,
    providers,
    out_plot: Path,
    do_compare: bool,
    compare_lead_hours: int,
    num_steps: int,
    model_version: str,
    stats_dir: Path | None = None,
) -> None:
    onnx_p = ZK_ROOT / "fengwu" / f"fengwu_{model_version}.onnx"
    if not onnx_p.is_file():
        print("[fengwu] 跳过：无 ONNX")
        return
    _progress("Fengwu", "加载 ONNX …")
    sess = create_session(onnx_p, providers)
    _progress("Fengwu", "读 NetCDF + 按 ONNX 通道数组输入（189 单帧或 138 双帧×13 层）…")
    inps = sess.get_inputs()
    if len(inps) != 1:
        raise RuntimeError("多输入 Fengwu 需扩展 feeds 映射")
    x = build_fengwu_onnx_combo_input(era5_root, date_yyyymmdd, hour0, sess)
    if stats_dir is not None:
        try:
            x = fengwu_normalize_for_onnx(x, stats_dir)
            print(f"[fengwu] input normalized with stats: {stats_dir}", flush=True)
        except Exception as e:
            print(f"[fengwu] stats normalize skipped: {e}", flush=True)
    _progress("Fengwu", f"sess.run 自回归推理（num_steps={num_steps}）…")
    cur = x.astype(np.float32)
    last_denorm: np.ndarray | None = None
    for step in range(1, int(num_steps) + 1):
        outs = sess.run(None, {inps[0].name: cur})
        surf_o, z_o, r_o, u_o, v_o, t_o = unpack_fengwu_ort_outputs(outs)
        pred69 = np.concatenate([surf_o, z_o, r_o, u_o, v_o, t_o], axis=0)
        if stats_dir is not None:
            try:
                pred69 = fengwu_denorm_chw(pred69, stats_dir)
            except Exception as e:
                print(f"[fengwu] stats denorm skipped at step {step}: {e}", flush=True)
        last_denorm = pred69.astype(np.float32)
        np.save(out_plot / f"fengwu_step{step:02d}_pred69.npy", pred69)
        if cur.shape[1] == 138 and stats_dir is not None:
            try:
                pred69_norm = fengwu_normalize_for_onnx(pred69[np.newaxis, ...], stats_dir)[0]
                cur = np.concatenate([cur[:, 69:], pred69_norm[np.newaxis, ...]], axis=1).astype(np.float32)
            except Exception as e:
                # If stats/channel mismatch happens here, we cannot continue autoregressive rollout safely.
                # Still keep the last_denorm outputs (step1 plots) and stop the loop.
                print(f"[fengwu] warning: failed to normalize for autoregressive update: {e}", flush=True)
                break
        else:
            break

    if last_denorm is None:
        raise RuntimeError("Fengwu produced no outputs")
    surf_o, z_o, r_o, u_o, v_o, t_o = (
        last_denorm[:4],
        last_denorm[4:17],
        last_denorm[17:30],
        last_denorm[30:43],
        last_denorm[43:56],
        last_denorm[56:69],
    )
    iz = _fengwu_level_index_1000(z_o.shape[0])
    print(
        f"[fengwu] output stats: "
        f"sfc_u10 std={float(np.nanstd(surf_o[0])):.3e}, "
        f"h1000_z std={float(np.nanstd(z_o[iz])):.3e}, "
        f"h1000_rh std={float(np.nanstd(r_o[iz])):.3e}",
        flush=True,
    )

    base_t = datetime(
        int(date_yyyymmdd[:4]),
        int(date_yyyymmdd[4:6]),
        int(date_yyyymmdd[6:8]),
        int(hour0),
        tzinfo=pytz.utc,
    )
    valid_t = base_t + timedelta(hours=compare_lead_hours)
    print(f"[fengwu] compare valid_time={valid_t.strftime('%Y-%m-%d %H:%M:%SZ')}", flush=True)
    tb = load_era5_truth_blob(era5_root, valid_t) if do_compare else None

    fw_fields: list[tuple[str, np.ndarray, str, str]] = [
        ("sfc_u10", surf_o[0], "10m u-wind (m/s)", "RdBu_r"),
        ("sfc_v10", surf_o[1], "10m v-wind (m/s)", "RdBu_r"),
        ("sfc_t2m", surf_o[2], "2m temperature (K)", "viridis"),
        ("sfc_msl", surf_o[3], "MSL (raw ch; check units)", "viridis"),
        ("h1000_z", z_o[iz], "1000 hPa geopotential", "viridis"),
        ("h1000_rh", r_o[iz], "1000 hPa relative humidity", "viridis"),
        ("h1000_u", u_o[iz], "1000 hPa u-wind (m/s)", "RdBu_r"),
        ("h1000_v", v_o[iz], "1000 hPa v-wind (m/s)", "RdBu_r"),
        ("h1000_t", t_o[iz], "1000 hPa temperature (K)", "viridis"),
    ]

    for fid, arr, label, cmap in fw_fields:
        _field_diag("fengwu", fid, arr)
        _plot_map(
            out_plot / f"fengwu_step01_{fid}.png",
            arr,
            f"Fengwu +{compare_lead_hours}h: {label} (physical units not guaranteed)",
            cmap=cmap,
        )
        if do_compare:
            tid = fid
            _plot_model_vs_era5(
                out_plot / f"fengwu_step01_{fid}_compare.png",
                arr,
                tb,
                f"Fengwu +{compare_lead_hours}h vs ERA5 {valid_t.strftime('%Y-%m-%d %H')}Z | {label}",
                field_id=tid,
            )

    print(f"[fengwu] 已写单步图 -> {out_plot}")


def run_fuxi_once(
    era5_root: Path,
    date_yyyymmdd: str,
    hour0: int,
    providers,
    stats_dir: Path | None,
    out_plot: Path,
    do_compare: bool,
    compare_lead_hours: int,
    num_steps: int,
    plot_step: int = -1,
) -> None:
    onnx_p = ZK_ROOT / "fuxi" / "short.onnx"
    if not onnx_p.is_file():
        print("[fuxi] 跳过：无 short.onnx")
        return
    _progress("Fuxi", "加载 short.onnx …")
    sess = create_session(onnx_p, providers)
    h_prev, h_curr = hour0, min(hour0 + 1, 23)
    _progress("Fuxi", "组装两时次 70 通道 …")
    raw = load_cepri_fuxi_fields(era5_root, date_yyyymmdd, h_prev, h_curr)
    x, fuxi_layout = fuxi_prepare_onnx_input(raw, sess)
    print(f"[fuxi] input layout={fuxi_layout}, x.shape={x.shape}", flush=True)
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    if stats_dir is not None and (stats_dir / "global_means.npy").is_file():
        mu = np.load(stats_dir / "global_means.npy")[:, :70, :, :]
        sd = np.load(stats_dir / "global_stds.npy")[:, :70, :, :]
        x = fuxi_normalize_for_layout(x, Path(stats_dir), fuxi_layout)
    x = x.astype(np.float32)
    _progress("Fuxi", f"sess.run（zforecast风格temb + (h,h+1) 双帧，num_steps={num_steps}）…")
    start_time_utc = datetime(
        int(date_yyyymmdd[:4]),
        int(date_yyyymmdd[4:6]),
        int(date_yyyymmdd[6:8]),
        int(hour0),
        tzinfo=pytz.utc,
    )
    cur = x
    out = None
    for step in range(1, int(num_steps) + 1):
        feeds = {}
        for inp in sess.get_inputs():
            if "temb" in inp.name.lower():
                feeds[inp.name] = _fuxi_temb_like_zforecast(start_time_utc, step, int(compare_lead_hours))
            else:
                feeds[inp.name] = cur
        y = sess.run(None, feeds)[0]
        cur = y.astype(np.float32)
        # Official output: latest time slot
        if y.ndim == 5 and fuxi_layout == "NTCHW":
            out_step = y[0, -1]
        elif y.ndim == 5 and fuxi_layout == "NCTHW":
            out_step = y[0, :, -1]
        elif y.ndim == 4:
            out_step = y[0]
        else:
            raise RuntimeError(f"Fuxi 输出 shape 非预期: {y.shape} (ndim={y.ndim})")
        np.save(out_plot / f"fuxi_step{step:02d}_latest_raw.npy", out_step)
        if step == (int(num_steps) if int(plot_step) < 0 else int(plot_step)):
            out = out_step
    if out is None:
        out = out_step

    def _fuxi_plot_ch(ch: int) -> np.ndarray:
        a = np.asarray(out[ch], dtype=np.float64)
        if mu is not None and sd is not None:
            m = np.asfarray(mu[0, ch])
            s = np.asfarray(sd[0, ch])
            a = a * s + m
        return a.astype(np.float32)

    has_stats = mu is not None
    stats_note = "" if has_stats else " (raw net output, denorm skipped)"

    # FuXi official channel order:
    # Z(0..12), T(13..25), U(26..38), V(39..51), R(52..64), T2M(65), U10(66), V10(67), MSL(68), TP(69)
    fuxi_ch = {
        "sfc_u10": 66,
        "sfc_v10": 67,
        "sfc_t2m": 65,
        "sfc_msl": 68,
        "sfc_tp": 69,
        "h1000_z": FUXI_I1000,
        "h1000_t": 13 + FUXI_I1000,
        "h1000_u": 26 + FUXI_I1000,
        "h1000_v": 39 + FUXI_I1000,
        "h1000_r": 52 + FUXI_I1000,
    }
    fuxi_labels = {
        "sfc_u10": ("10m u-wind (m/s)", "RdBu_r", "sfc_u10"),
        "sfc_v10": ("10m v-wind (m/s)", "RdBu_r", "sfc_v10"),
        "sfc_t2m": ("2m temperature (K)", "viridis", "sfc_t2m"),
        "sfc_msl": ("MSL pressure (Pa)", "viridis", "sfc_msl"),
        # CEPRI test_era5_data 无官方对齐 TP 真值，故仅画预报图不做 compare。
        "sfc_tp": ("total precipitation (model channel)", "viridis", None),
        "h1000_z": ("1000 hPa geopotential", "viridis", "h1000_z"),
        "h1000_u": ("1000 hPa u-wind (m/s)", "RdBu_r", "h1000_u"),
        "h1000_v": ("1000 hPa v-wind (m/s)", "RdBu_r", "h1000_v"),
        "h1000_t": ("1000 hPa temperature (K)", "viridis", "h1000_t"),
        "h1000_r": ("1000 hPa RH (FuXi channel)", "viridis", "h1000_rh"),
    }

    base_t = start_time_utc
    # Compare with valid time of the plotted step.
    step_for_plot = int(num_steps) if int(plot_step) < 0 else int(plot_step)
    lead_for_plot = int(compare_lead_hours) * step_for_plot
    valid_t = base_t + timedelta(hours=lead_for_plot)
    print(f"[fuxi] compare valid_time={valid_t.strftime('%Y-%m-%d %H:%M:%SZ')}", flush=True)
    tb = load_era5_truth_blob(era5_root, valid_t) if do_compare else None

    for fid, chi in fuxi_ch.items():
        lab, cmap, tid = fuxi_labels[fid]
        arr = _fuxi_plot_ch(chi)
        _field_diag("fuxi", fid, arr)
        arr_std = float(np.std(arr))
        if not np.isfinite(arr_std) or arr_std < 1e-8:
            print(f"[fuxi] warning: near-constant field {fid}, std={arr_std:.3e}, ch={chi}", flush=True)
        _plot_map(
            out_plot / f"fuxi_step01_{fid}.png",
            arr,
            f"Fuxi short +{lead_for_plot}h: {lab}{stats_note}",
            cmap=cmap,
        )
        if do_compare and tid is not None:
            _plot_model_vs_era5(
                out_plot / f"fuxi_step01_{fid}_compare.png",
                arr,
                tb,
                f"Fuxi short +{lead_for_plot}h vs ERA5 {valid_t.strftime('%Y-%m-%d %H')}Z | {lab}",
                field_id=tid,
            )
    print(f"[fuxi] 已写单步图 -> {out_plot}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", type=Path, default=GRAPH_ROOT / "test_era5_data")
    ap.add_argument("--date", default="20260301")
    ap.add_argument(
        "--hour",
        type=int,
        default=0,
        help="GraphCast/Pangu / Fengwu(189) / 对比锚点起报整点；Fuxi 与 Fengwu(138) 第二帧为 min(h+1,23)",
    )
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument(
        "--fengwu-model-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="默认 v2（旧工作流）；ERA5 严格评估可改 v1",
    )
    ap.add_argument("--fuxi-plot-step", type=int, default=-1, help="FuXi绘图使用第几步（-1表示最后一步）")
    ap.add_argument("--device", default="auto", choices=["auto", "dcu", "cuda", "cpu"])
    ap.add_argument("--stats-for-fuxi", type=Path, default=None, help="FuXi stats目录（需含global_means/global_stds）")
    ap.add_argument(
        "--no-compare",
        action="store_true",
        help="不生成与同期 ERA5 分析场的对比三联图",
    )
    ap.add_argument(
        "--fengwu-compare-lead-hours",
        type=int,
        default=6,
        help="Fengwu 单步预报相对起报时刻的有效提前量（小时），用于取同期 ERA5",
    )
    ap.add_argument(
        "--fuxi-compare-lead-hours",
        type=int,
        default=6,
        help="Fuxi short 相对起报的有效提前量（小时），用于取同期 ERA5（建议与 temb 提前量一致）",
    )
    ap.add_argument(
        "--stats-for-fengwu",
        type=Path,
        default=None,
        help="可选：Fengwu stats（global_means/global_stds；或 data_mean/data_std）",
    )
    ap.add_argument(
        "--only-models",
        type=str,
        default="",
        help="逗号分隔，只跑子集：graphcast,pangu,fengwu,fuxi。例：fengwu,fuxi（跳过 GraphCast/Pangu 与输入场 PNG）",
    )
    args = ap.parse_args()

    do_compare = not args.no_compare
    only_raw = [x.strip().lower() for x in args.only_models.split(",") if x.strip()]
    only_set = set(only_raw) if only_raw else None
    run_graphcast = only_set is None or "graphcast" in only_set
    run_pangu = only_set is None or "pangu" in only_set
    run_fengwu = only_set is None or "fengwu" in only_set
    run_fuxi = only_set is None or "fuxi" in only_set

    _progress(
        "启动",
        f"cwd→graphcast, date={args.date} hour={args.hour} num_steps={args.num_steps} device={args.device} "
        f"compare={do_compare} only_models={only_set or 'all'}",
    )
    os.chdir(GRAPH_ROOT)
    _progress("配置", "读取 conf/config.yaml …")
    cfg = YParams(str(GRAPH_ROOT / "conf/config.yaml"), "model")

    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    if run_graphcast:
        meta_path = Path(cfg.data_dir) / "metadata.json"
        variables = json.load(open(meta_path, encoding="utf-8"))["variables"]
        channel_indices = [variables.index(c) for c in cfg.channels]
        mu_full = np.load(Path(cfg.stats_dir) / "global_means.npy")
        sd_full = np.load(Path(cfg.stats_dir) / "global_stds.npy")
        mu = mu_full[0, channel_indices, 0, 0].astype(np.float32)
        sd = sd_full[0, channel_indices, 0, 0].astype(np.float32)
        _progress("配置", "metadata + stats 已就绪（GraphCast）")
    else:
        _progress("配置", "跳过 GraphCast metadata（--only-models 未含 graphcast）")

    out_plot = GRAPH_ROOT / "result" / "four_models_test_era5" / "plots"
    out_plot.mkdir(parents=True, exist_ok=True)

    era5_root = args.test_data
    date = args.date
    hour0 = args.hour

    blob0 = None
    if run_graphcast or run_pangu:
        _progress("数据", f"读 ERA5 test_era5_data + 画输入场 …")
        blob0 = load_cepri_time(era5_root, date, hour0)
        _plot_input_standard_fields(blob0, out_plot, "input_t0")
        _progress("数据", "输入场 PNG (sfc + 1000hPa) 已写")
    else:
        _progress("数据", "跳过输入场 PNG（未跑 GraphCast/Pangu）")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device in ("cuda", "dcu"):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print(f"[device] {args.device} requested but CUDA/ROCm unavailable; fallback to CPU", flush=True)
            device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if run_graphcast and blob0 is not None and mu is not None and sd is not None:
        try:
            start_time = datetime(
                int(date[:4]), int(date[4:6]), int(date[6:8]), hour0, tzinfo=pytz.utc
            )
            _progress("GraphCast", "进入 GraphCast rollout …")
            run_graphcast_rollout(
                cfg,
                blob0,
                mu,
                sd,
                args.num_steps,
                start_time,
                out_plot,
                device,
                era5_root,
                do_compare,
            )
        except Exception as e:
            print(f"[graphcast] 失败: {e}")
    elif run_graphcast:
        print("[graphcast] 跳过：缺少 blob0 或 stats（不应发生）", flush=True)

    _progress("阶段", "ONNX 段（Pangu / Fengwu / Fuxi）按 only-models 选择执行")
    providers = pick_providers(args.device)
    print("ZK ORT providers:", providers, flush=True)

    if run_pangu:
        try:
            run_pangu_rollout(era5_root, date, hour0, args.num_steps, providers, out_plot, do_compare)
        except Exception as e:
            print(f"[pangu] 失败: {e}")
    else:
        print("[pangu] 跳过（--only-models）", flush=True)

    if run_fengwu:
        try:
            if args.stats_for_fengwu is not None:
                fw_stats = args.stats_for_fengwu
            else:
                fw_default = ZK_ROOT / "fengwu"
                has_fw_stats = (
                    (fw_default / "data_mean.npy").is_file()
                    and (fw_default / "data_std.npy").is_file()
                ) or (
                    (fw_default / "global_means.npy").is_file()
                    and (fw_default / "global_stds.npy").is_file()
                )
                fw_stats = fw_default if has_fw_stats else None
                if fw_stats is None:
                    print("[fengwu] 未找到官方 stats（data_mean/data_std），将跳过归一化直接推理", flush=True)
            run_fengwu_once(
                era5_root,
                date,
                hour0,
                providers,
                out_plot,
                do_compare,
                args.fengwu_compare_lead_hours,
                args.num_steps,
                args.fengwu_model_version,
                fw_stats,
            )
        except Exception as e:
            print(f"[fengwu] 失败: {e}")
    else:
        print("[fengwu] 跳过（--only-models）", flush=True)

    # FuXi: allow running without stats (aligned with zforecast.py workflow).
    stats_fuxi = args.stats_for_fuxi
    if run_fuxi:
        if stats_fuxi is None:
            print("[fuxi] 未提供 --stats-for-fuxi：按 raw 输入/输出推理", flush=True)
        else:
            has_mu = (stats_fuxi / "global_means.npy").is_file()
            has_sd = (stats_fuxi / "global_stds.npy").is_file()
            if not (has_mu and has_sd):
                print(
                    f"[fuxi] stats 不完整（{stats_fuxi}），将按 raw 输入/输出推理："
                    f"global_means={has_mu}, global_stds={has_sd}",
                    flush=True,
                )
                stats_fuxi = None
    if run_fuxi:
        try:
            run_fuxi_once(
                era5_root,
                date,
                hour0,
                providers,
                stats_fuxi,
                out_plot,
                do_compare,
                args.fuxi_compare_lead_hours,
                args.num_steps,
                args.fuxi_plot_step,
            )
        except Exception as e:
            print(f"[fuxi] 失败: {e}")
    else:
        print("[fuxi] 跳过（--only-models）", flush=True)

    _progress("结束", f"全部阶段跑完，图目录: {out_plot}")
    print(f"完成。图目录: {out_plot}", flush=True)


if __name__ == "__main__":
    main()
