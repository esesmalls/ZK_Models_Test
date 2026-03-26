#!/usr/bin/env python3
"""
功能二（仅评估）：从已有 NPY 预报栈 + ERA5 适配器重算 W-MAE/W-RMSE，
输出合并 CSV 与时序图（与滚动推理内嵌评估一致）。

不加载 ONNX/GraphCast，适合推理已完成、仅需重出评估结果或改指标的场景。

示例：
  cd /path/to/graphcast
  python ZK_Models/run_eval_npy.py \\
    --data-source gundong_20260324 \\
    --date-range 20260310 \\
    --init-hour 12 --max-lead 240 --lead-step 6 \\
    --models pangu fengwu fuxi graphcast \\
    --output-root /public/share/aciwgvx1jd/GunDong_Infer_result_12h
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from runtime_paths import UNIFIED_PIPELINE_ROOT, GRAPH_CAST_ROOT

_ZK_ROOT = UNIFIED_PIPELINE_ROOT
_GRAPH_ROOT = GRAPH_CAST_ROOT
sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

from core.data.detector import get_adapter
from core.data.channel_mapper import extract_surface_vars
from core.data.surface_units import harmonize_surface_pair
from core.evaluation.metrics import MetricsAccumulator
from pipelines.rolling_pipeline import _load_data_source, _parse_date_range
from zk_io.plot_utils import plot_compare

_EVAL_MODEL_ORDER = ["PanGu", "FengWu", "FuXi", "GraphCast", "GraphCast_CS"]

_SLUG_TO_DISPLAY = {
    "pangu": "PanGu",
    "fengwu": "FengWu",
    "fuxi": "FuXi",
    "graphcast": "GraphCast",
    "graphcast_cs": "GraphCast_CS",
}


def _npy_path(
    output_root: Path,
    display_name: str,
    slug: str,
    var: str,
    init_tag: str,
) -> Path:
    if slug == "pangu":
        fname = f"{var}_surface_{init_tag}.npy"
    else:
        fname = f"{var}_{init_tag}.npy"
    return output_root / display_name / "ERA5_6H" / fname


def run_eval_from_npy(
    *,
    output_root: Path,
    data_source: str,
    date_range: str,
    init_hour: int,
    lead_step: int,
    max_lead: int,
    model_slugs: List[str],
    variables: List[str],
    metrics: List[str],
    save_diff: bool,
    save_diff_nc: bool,
    data_cfg_path: Path,
    spatial_plots: bool,
) -> None:
    n_steps = max_lead // lead_step
    leads = list(range(lead_step, max_lead + 1, lead_step))
    if len(leads) != n_steps:
        raise ValueError("max_lead 必须能被 lead_step 整除")

    data_root, data_fmt, src_cfg = _load_data_source(data_source, data_cfg_path)
    adapter = get_adapter(
        data_root,
        fmt=data_fmt,
        use_monthly_subdir=src_cfg.get("use_monthly_subdir", False),
    )

    dates = _parse_date_range(date_range)
    for date in dates:
        init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), init_hour)
        init_tag = init_dt.strftime("%Y%m%dT%H")
        init_blob = adapter.load_blob(date, init_hour)
        lat = init_blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        lon = init_blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))

        eval_out_dir = output_root / f"eval_{max_lead}h_{init_tag}"
        acc = MetricsAccumulator(
            lats=lat,
            metrics=metrics,
            save_diff=save_diff,
            save_diff_nc=save_diff_nc,
        )
        valid_times: Dict[Tuple[str, str], List] = {}

        # 预加载各模型各变量的 NPY
        pred_stacks: Dict[Tuple[str, str], np.ndarray] = {}
        for slug in model_slugs:
            disp = _SLUG_TO_DISPLAY.get(slug.lower(), slug)
            for var in variables:
                pth = _npy_path(output_root, disp, slug.lower(), var, init_tag)
                if not pth.is_file():
                    print(f"[eval_npy] 跳过缺失文件: {pth}", flush=True)
                    continue
                arr = np.load(str(pth), mmap_mode="r")
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                pred_stacks[(disp, var)] = arr

        for si, current_lead in enumerate(leads):
            if si >= n_steps:
                break
            valid_dt = init_dt + timedelta(hours=current_lead)
            truth_blob = adapter.load_blob_for_valid_time(valid_dt)
            if truth_blob is None:
                print(f"[eval_npy] 无真值 valid={valid_dt}, lead={current_lead}h", flush=True)
                continue
            truth_sfc = extract_surface_vars(truth_blob, variables)

            for slug in model_slugs:
                disp = _SLUG_TO_DISPLAY.get(slug.lower(), slug)
                for var_name in variables:
                    key = (disp, var_name)
                    if key not in pred_stacks:
                        continue
                    stack = pred_stacks[key]
                    if si >= stack.shape[0]:
                        continue
                    pred_arr = np.asarray(stack[si], dtype=np.float32)
                    if var_name not in truth_sfc:
                        continue
                    p_h, t_h = harmonize_surface_pair(
                        var_name, pred_arr, truth_sfc[var_name]
                    )
                    acc.add(disp, var_name, current_lead, p_h, t_h)
                    if save_diff_nc:
                        valid_times.setdefault((disp, var_name), []).append(valid_dt)

                    if spatial_plots:
                        plot_dir = output_root / "plots" / slug.lower() / init_tag
                        plot_compare(
                            plot_dir / f"{var_name}_lead{current_lead:03d}.png",
                            p_h,
                            t_h,
                            title=f"{disp} +{current_lead}h {var_name} | {init_tag} (from NPY)",
                            cmap="RdBu_r" if var_name in ("u10", "v10") else "viridis",
                        )

        if acc.to_dataframe().empty:
            print(f"[eval_npy] 无有效指标记录，跳过写入: {init_tag}", flush=True)
            continue
        acc.save(
            save_dir=eval_out_dir,
            time_tag=init_tag,
            lon=lon,
            valid_times=valid_times if save_diff_nc else None,
            model_order=_EVAL_MODEL_ORDER,
        )
        print(f"[eval_npy] 已写入 {eval_out_dir}", flush=True)


def main() -> None:
    cfg_path = _ZK_ROOT / "config" / "defaults.yaml"
    e_defaults: dict = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            e_defaults = (yaml.safe_load(f) or {}).get("evaluation", {})

    ap = argparse.ArgumentParser(description="从已有 NPY 重跑定量评估（合并多模型 CSV/时序图）")
    ap.add_argument("--data-source", default="gundong_20260324")
    ap.add_argument("--date-range", required=True, help="yyyymmdd 或 yyyymmdd:yyyymmdd")
    ap.add_argument("--init-hour", type=int, default=12)
    ap.add_argument("--lead-step", type=int, default=6)
    ap.add_argument("--max-lead", type=int, default=240)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["pangu", "fengwu", "fuxi", "graphcast", "graphcast_cs"],
        help="模型 slug：pangu fengwu fuxi graphcast",
    )
    ap.add_argument(
        "--variables",
        nargs="+",
        default=["u10", "v10", "t2m", "msl"],
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h"),
    )
    ap.add_argument("--metrics", nargs="+", default=e_defaults.get("metrics", ["W-MAE", "W-RMSE"]))
    ap.add_argument("--save-diff", action="store_true")
    ap.add_argument("--save-diff-nc", action="store_true")
    ap.add_argument("--data-config", type=Path, default=_ZK_ROOT / "config" / "data.yaml")
    ap.add_argument(
        "--spatial-plots",
        action="store_true",
        help="额外写三联对比图到 plots/{model}/{init_tag}/（与滚动推理一致）",
    )
    args = ap.parse_args()

    run_eval_from_npy(
        output_root=args.output_root,
        data_source=args.data_source,
        date_range=args.date_range,
        init_hour=args.init_hour,
        lead_step=args.lead_step,
        max_lead=args.max_lead,
        model_slugs=[m.lower() for m in args.models],
        variables=list(args.variables),
        metrics=list(args.metrics),
        save_diff=args.save_diff,
        save_diff_nc=args.save_diff_nc,
        data_cfg_path=args.data_config,
        spatial_plots=args.spatial_plots,
    )


if __name__ == "__main__":
    main()
