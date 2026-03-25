#!/usr/bin/env python3
"""
功能二 CLI 入口：滚动推理（含可选定量评估）。

对应原代码：GunDong_Infer/run_gundong_infer.py + evaluate_models.py。

示例：
  # 单日，FengWu + FuXi，240h，全地表变量，开启评估
  python run_rolling.py \\
      --models fengwu fuxi \\
      --data-source gundong_20260324 \\
      --date-range 20260308 \\
      --init-hour 12 --lead-step 6 --max-lead 240 \\
      --enable-eval

  # 多日范围，全部模型，不出图（只保存 NPY）
  python run_rolling.py \\
      --models all \\
      --data-source gundong_20260324 \\
      --date-range 20260301:20260318 \\
      --skip-plots

  # 自定义变量 + 保存 diff + 保存 NC
  python run_rolling.py \\
      --models pangu graphcast \\
      --data-source gundong_20260324 \\
      --date-range 20260308 \\
      --variables u10 v10 t2m \\
      --save-nc --enable-eval --save-diff \\
      --output-root /public/share/aciwgvx1jd/my_results
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------
# 必须在任何 ROCm/CUDA 相关 import 之前设置每进程可见设备。
# torchrun / srun 会注入 LOCAL_RANK；单进程时不做修改。
# 参考主分支 GunDong_Infer/run_gundong_infer.py 的相同处理。
# ---------------------------------------------------------------
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    for _k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
               "HIP_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES"):
        os.environ[_k] = str(_local_rank)

import argparse
from pathlib import Path
from typing import List, Optional

_ZK_ROOT = Path(__file__).resolve().parent
_GRAPH_ROOT = _ZK_ROOT.parent
sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

import yaml
from pipelines.rolling_pipeline import run_rolling
from core.monitoring import start_hardware_logger


def _load_defaults() -> dict:
    cfg_path = _ZK_ROOT / "config" / "defaults.yaml"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_all_enabled_models(models_cfg: Path) -> List[str]:
    with open(models_cfg, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [
        name for name, cfg in raw.get("models", {}).items()
        if cfg.get("enabled", True)
    ]


def main():
    defaults = _load_defaults()
    r_defaults = defaults.get("pipeline", {}).get("rolling", {})
    e_defaults = defaults.get("evaluation", {})

    ap = argparse.ArgumentParser(
        description="功能二：滚动推理（+ 可选定量评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- 模型与数据 ----
    ap.add_argument(
        "--models", nargs="+", default=["fengwu", "fuxi"],
        help="模型列表，或 'all' 代表全部启用模型",
    )
    ap.add_argument(
        "--data-source", default="gundong_20260324",
        help="数据源名称（config/data.yaml key）或数据根目录路径",
    )

    # ---- 时间参数 ----
    ap.add_argument(
        "--date-range", required=True,
        help="日期 'yyyymmdd' 或范围 'yyyymmdd:yyyymmdd'，如 20260308 或 20260301:20260318",
    )
    ap.add_argument(
        "--init-hour", type=int, default=r_defaults.get("init_hour", 12),
        help="起报时刻 UTC（默认 12）",
    )
    ap.add_argument(
        "--lead-step", type=int, default=r_defaults.get("lead_step_hours", 6),
        help="每步步长小时（默认 6）",
    )
    ap.add_argument(
        "--max-lead", type=int, default=r_defaults.get("max_lead_hours", 240),
        help="最大预报时长小时（默认 240）",
    )

    # ---- 变量 ----
    ap.add_argument(
        "--variables", nargs="*", metavar="VAR",
        help="输出变量（None=模型默认全部地表变量）",
    )

    # ---- 输出 ----
    ap.add_argument(
        "--output-root", type=Path,
        default=Path(r_defaults.get("output_root", "/public/share/aciwgvx1jd/GunDong_Infer_result_12h")),
        help="结果根目录",
    )
    ap.add_argument(
        "--skip-plots", action="store_true",
        default=r_defaults.get("skip_plots", False),
        help="跳过逐步对比图（节省时间）",
    )
    ap.add_argument(
        "--save-nc", action="store_true",
        default=r_defaults.get("save_nc", False),
        help="保存 per-step NetCDF 文件",
    )

    # ---- 评估 ----
    ap.add_argument(
        "--enable-eval", action="store_true",
        default=r_defaults.get("enable_eval", False),
        help="开启内嵌定量评估（自动输出 RMSE/MAE CSV + 时序图）",
    )
    ap.add_argument(
        "--save-diff", action="store_true",
        default=r_defaults.get("save_diff", False),
        help="保存差值场 npy 文件（需 --enable-eval）",
    )
    ap.add_argument(
        "--save-diff-nc", action="store_true",
        help="保存差值场 nc 文件（需 --enable-eval）",
    )
    ap.add_argument(
        "--metrics", nargs="+",
        default=e_defaults.get("metrics", ["W-MAE", "W-RMSE"]),
        help="评估指标（默认 W-MAE W-RMSE）",
    )

    # ---- 硬件/配置 ----
    ap.add_argument(
        "--device", default="auto",
        choices=["auto", "dcu", "cuda", "cpu"],
        help="推理设备",
    )
    ap.add_argument(
        "--parallel-mode", default="auto",
        choices=["auto", "date", "model"],
        help=(
            "多卡并行策略（需配合 torchrun/WORLD_SIZE 使用）:\n"
            "  auto  — 日期数 >= WORLD_SIZE 时按日期分片，否则按模型分片\n"
            "  date  — 每 rank 处理不同日期（18天×8卡 等多日场景）\n"
            "  model — 每 rank 处理不同模型（单日×4模型×4卡 等场景）"
        ),
    )
    ap.add_argument(
        "--models-config", type=Path,
        default=_ZK_ROOT / "config" / "models.yaml",
    )
    ap.add_argument(
        "--data-config", type=Path,
        default=_ZK_ROOT / "config" / "data.yaml",
    )

    # ---- 硬件监控 ----
    ap.add_argument(
        "--no-monitor", action="store_true",
        help="禁用硬件监控（默认开启）",
    )
    ap.add_argument(
        "--monitor-interval", type=int, default=30,
        help="硬件监控轮询间隔（秒，默认 30）",
    )

    args = ap.parse_args()

    if args.models == ["all"]:
        model_names = _get_all_enabled_models(args.models_config)
    else:
        model_names = args.models

    variables = args.variables if args.variables else None

    with start_hardware_logger(
        log_dir=_ZK_ROOT / "logs",
        poll_interval=args.monitor_interval,
        enabled=not args.no_monitor,
    ):
        run_rolling(
            model_names=model_names,
            data_source=args.data_source,
            date_range=args.date_range,
            init_hour=args.init_hour,
            lead_step=args.lead_step,
            max_lead=args.max_lead,
            variables=variables,
            output_root=args.output_root,
            device=args.device,
            skip_plots=args.skip_plots,
            save_nc=args.save_nc,
            enable_eval=args.enable_eval,
            save_diff=args.save_diff,
            save_diff_nc=args.save_diff_nc,
            metrics=args.metrics,
            models_cfg_path=args.models_config,
            data_cfg_path=args.data_config,
            parallel_mode=args.parallel_mode,
        )


if __name__ == "__main__":
    main()
