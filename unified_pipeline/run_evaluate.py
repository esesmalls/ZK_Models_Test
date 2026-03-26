#!/usr/bin/env python3
"""
独立评估入口：对已有 NPY 预报文件重新计算指标。

对应原代码：evaluate_models.py（但消除了重复 diff 计算）。

若已在 run_rolling.py 中开启 --enable-eval，则无需再运行此脚本。
此脚本适用于：
  1. 对历史存档 NPY 文件补充评估
  2. 调整指标或变量范围后重新计算

示例：
  # 评估所有模型，3 个变量
  python run_evaluate.py \\
      --time-tag 20260308T12 \\
      --models FengWu FuXi GraphCast PanGu \\
      --variables u10 v10 t2m

  # 评估并保存 diff npy + nc
  python run_evaluate.py \\
      --time-tag 20260308T12 \\
      --save-diff --save-diff-nc

  # 使用自定义路径
  python run_evaluate.py \\
      --time-tag 20260308T12 \\
      --pred-base-dir /path/to/results \\
      --era5-dir /path/to/era5/surface
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from runtime_paths import UNIFIED_PIPELINE_ROOT, GRAPH_CAST_ROOT

_ZK_ROOT = UNIFIED_PIPELINE_ROOT
_GRAPH_ROOT = GRAPH_CAST_ROOT
sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

from core.evaluation.metrics import MetricsAccumulator, compute_step_metrics


def _progress(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [evaluate] {msg}", flush=True)


def _load_defaults() -> dict:
    cfg_path = _ZK_ROOT / "config" / "defaults.yaml"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _npy_path(pred_base: Path, model: str, var: str, time_tag: str, pangu: bool) -> Path:
    fname = f"{var}_surface_{time_tag}.npy" if pangu else f"{var}_{time_tag}.npy"
    return pred_base / model / "ERA5_6H" / fname


def main():
    defaults = _load_defaults()
    e_defaults = defaults.get("evaluation", {})

    ap = argparse.ArgumentParser(
        description="独立评估：对已有 NPY 文件重新计算 W-RMSE/W-MAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--time-tag", required=True,
        help="时间标签，如 20260308T12",
    )
    ap.add_argument(
        "--models", nargs="+",
        default=["FengWu", "GraphCast", "FuXi", "PanGu"],
        help="模型目录名（大小写需与结果目录一致）",
    )
    ap.add_argument(
        "--variables", nargs="+",
        default=["u10", "v10", "t2m"],
        help="评估变量（默认 u10 v10 t2m）",
    )
    ap.add_argument(
        "--pred-base-dir", type=Path,
        default=Path(e_defaults.get("pred_base_dir",
                                    "/public/share/aciwgvx1jd/GunDong_Infer_result_12h")),
        help="预报 NPY 根目录",
    )
    ap.add_argument(
        "--era5-dir", type=Path,
        default=Path(e_defaults.get("era5_dir",
                                    "/public/share/aciwgvx1jd/20260324/surface")),
        help="ERA5 地表 NC 目录",
    )
    ap.add_argument(
        "--step-interval", type=int,
        default=e_defaults.get("step_interval_hours", 6),
        help="预报步长小时（默认 6）",
    )
    ap.add_argument(
        "--expected-steps", type=int,
        default=e_defaults.get("expected_steps", 40),
        help="预报总步数（默认 40，即 240h/6h）",
    )
    ap.add_argument(
        "--metrics", nargs="+",
        default=e_defaults.get("metrics", ["W-MAE", "W-RMSE"]),
        help="评估指标",
    )
    ap.add_argument(
        "--save-diff", action="store_true",
        default=e_defaults.get("save_diff", False),
        help="保存 diff npy 文件",
    )
    ap.add_argument(
        "--save-diff-nc", action="store_true",
        help="保存 diff nc 文件",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="评估结果输出目录（默认: {pred_base_dir}/eval_{max_lead}h_{time_tag}）",
    )

    args = ap.parse_args()

    time_tag = args.time_tag
    initial_time_str = f"{time_tag[:4]}-{time_tag[4:6]}-{time_tag[6:8]}T{time_tag[9:11]}:00:00"
    initial_dt = datetime.strptime(initial_time_str, "%Y-%m-%dT%H:%M:%S")
    max_lead = args.step_interval * args.expected_steps

    if args.output_dir is None:
        save_dir = args.pred_base_dir / f"eval_{max_lead}h_{time_tag}"
    else:
        save_dir = args.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取网格信息
    _progress("加载网格和纬度权重...")
    sample_pattern = str(args.era5_dir / "*.nc")
    sample_files = sorted(glob.glob(sample_pattern))
    if not sample_files:
        raise FileNotFoundError(f"ERA5 目录中未找到 NC 文件: {sample_pattern}")
    with xr.open_dataset(sample_files[0]) as ds_sample:
        lats = ds_sample.latitude.values.astype(np.float32)
        lons = ds_sample.longitude.values.astype(np.float32)
        time_dim = "valid_time" if "valid_time" in ds_sample.dims else "time"

    # 预报有效时间序列
    valid_times = [
        initial_dt + timedelta(hours=(i + 1) * args.step_interval)
        for i in range(args.expected_steps)
    ]

    acc = MetricsAccumulator(
        lats=lats,
        metrics=args.metrics,
        save_diff=args.save_diff,
        save_diff_nc=args.save_diff_nc,
    )
    valid_times_map = {}

    ds_cache = {}
    current_date = None

    for model in args.models:
        _progress(f"{'='*45}")
        _progress(f"处理模型: {model}")
        _progress(f"{'='*45}")

        is_pangu = model.lower() == "pangu"

        for var in args.variables:
            npy_path = _npy_path(args.pred_base_dir, model, var, time_tag, is_pangu)
            if not npy_path.is_file():
                _progress(f"  跳过（找不到文件）: {npy_path}")
                continue

            pred_data = np.load(str(npy_path)).squeeze()
            if pred_data.ndim > 3:
                pred_data = pred_data[0]
            if pred_data.ndim == 2:
                pred_data = pred_data[np.newaxis]

            vt_list = []

            for step_idx, valid_dt in enumerate(valid_times):
                lead_hours = (step_idx + 1) * args.step_interval
                target_date_str = valid_dt.strftime("%Y_%m_%d")

                # 更新 ERA5 日期缓存
                if current_date != target_date_str:
                    for ds in ds_cache.values():
                        ds.close()
                    ds_cache.clear()
                    era5_files = glob.glob(
                        str(args.era5_dir / f"{target_date_str}_surface_*.nc")
                    )
                    for f in era5_files:
                        ds_cache[f] = xr.open_dataset(f)
                    current_date = target_date_str

                # 查找真值
                true_data = None
                for ds_day in ds_cache.values():
                    if valid_dt in ds_day[time_dim].values:
                        era5_var = var if var in ds_day else var.replace("u10", "10u").replace("v10", "10v")
                        if era5_var in ds_day:
                            true_data = ds_day[era5_var].sel({time_dim: valid_dt}).values
                            break

                if true_data is not None and step_idx < pred_data.shape[0]:
                    acc.add(model, var, lead_hours, pred_data[step_idx], true_data)
                    vt_list.append(valid_dt)
                else:
                    _progress(f"  缺失 ERA5 数据: {valid_dt}，跳过")

            if args.save_diff_nc:
                valid_times_map[(model, var)] = vt_list

    for ds in ds_cache.values():
        ds.close()

    _progress("保存评估结果...")
    acc.save(
        save_dir=save_dir,
        time_tag=time_tag,
        lon=lons,
        valid_times=valid_times_map if args.save_diff_nc else None,
    )

    _progress(f"评估完成，结果保存至: {save_dir}")


if __name__ == "__main__":
    main()
