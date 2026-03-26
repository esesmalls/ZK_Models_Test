#!/usr/bin/env python3
"""
功能一 CLI 入口：初步推理验证。

对应原代码：run_four_models_test_era5.py（test_era5_data 数据格式）或其他数据格式。

示例：
  # 使用 test_era5 数据源验证 pangu 和 fengwu，2 步推理，全地表变量
  python run_verify.py --models pangu fengwu \\
      --data-source test_era5 --date 20260308 --hour 12 \\
      --num-steps 2 --all-surface

  # 自定义变量 + 气压层
  python run_verify.py --models graphcast \\
      --data-source /path/to/data --date 20260308 --hour 0 \\
      --variables u10 v10 t2m --pressure-vars z:1000,500 t:850

  # 所有模型，使用默认配置（gundong 数据源）
  python run_verify.py --models all --data-source gundong_20260324 \\
      --date 20260308 --hour 12
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ZK_ROOT = Path(__file__).resolve().parent
_GRAPH_ROOT = _ZK_ROOT.parent
sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

import yaml
from pipelines.verify_pipeline import run_verify
from core.monitoring import start_hardware_logger


def _parse_pressure_vars(args_list: List[str]) -> Dict[str, List[int]]:
    """
    解析 --pressure-vars 参数，格式: var:level1,level2 ...
    例如：z:1000,500 t:850
    """
    result: Dict[str, List[int]] = {}
    for item in args_list:
        if ":" in item:
            var, levels_str = item.split(":", 1)
            result[var] = [int(l) for l in levels_str.split(",")]
        else:
            # 默认 1000hPa
            result[item] = [1000]
    return result


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
    v_defaults = defaults.get("pipeline", {}).get("verify", {})

    ap = argparse.ArgumentParser(
        description="功能一：初步推理验证（1~N步推理 + 三联对比图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--models", nargs="+", default=["pangu"],
        help="模型名称列表，或 'all' 代表全部启用模型（如：pangu fengwu fuxi graphcast）",
    )
    ap.add_argument(
        "--data-source", default="test_era5",
        help="数据源名称（config/data.yaml 中的 key）或数据根目录路径",
    )
    ap.add_argument("--date", required=True, help="推理日期 yyyymmdd，如 20260308")
    ap.add_argument("--hour", type=int, default=12, help="起报时刻 UTC（默认 12）")
    ap.add_argument(
        "--num-steps", type=int,
        default=v_defaults.get("num_steps", 2),
        help="推理步数（默认 2）",
    )

    var_grp = ap.add_mutually_exclusive_group()
    var_grp.add_argument(
        "--variables", nargs="+", metavar="VAR",
        help="地表变量名（如 u10 v10 t2m msl）",
    )
    var_grp.add_argument(
        "--all-surface", action="store_true",
        help="使用所选模型的全部地表变量",
    )

    ap.add_argument(
        "--pressure-vars", nargs="*", metavar="VAR:LEVELS",
        help="气压层变量，格式 z:1000,500 t:850（不指定则不出气压图）",
    )

    ap.add_argument(
        "--output-root", type=Path,
        default=Path(v_defaults.get("output_root", "results/verify")),
        help="输出根目录",
    )
    ap.add_argument(
        "--device", default="auto",
        choices=["auto", "dcu", "cuda", "cpu"],
        help="推理设备（默认 auto）",
    )
    ap.add_argument(
        "--skip-plots", action="store_true",
        help="跳过出图（仅推理调试用）",
    )
    ap.add_argument(
        "--no-monitor", action="store_true",
        help="禁用硬件监控（默认开启）",
    )
    ap.add_argument(
        "--models-config", type=Path,
        default=_ZK_ROOT / "config" / "models.yaml",
        help="模型配置文件路径",
    )
    ap.add_argument(
        "--data-config", type=Path,
        default=_ZK_ROOT / "config" / "data.yaml",
        help="数据源配置文件路径",
    )

    args = ap.parse_args()

    # 解析模型列表
    if args.models == ["all"]:
        model_names = _get_all_enabled_models(args.models_config)
    else:
        model_names = args.models

    # 解析变量
    variables = None if args.all_surface else args.variables

    # 解析气压层变量
    pressure_vars = None
    if args.pressure_vars is not None:
        if len(args.pressure_vars) == 0:
            # --pressure-vars 但无参数 → 用默认值
            pv_defaults = v_defaults.get("default_pressure_vars", {})
            pressure_vars = {k: v for k, v in pv_defaults.items()} if pv_defaults else None
        else:
            pressure_vars = _parse_pressure_vars(args.pressure_vars)

    with start_hardware_logger(
        log_dir=_ZK_ROOT / "logs",
        poll_interval=30,
        enabled=not args.no_monitor,
    ):
        run_verify(
            model_names=model_names,
            data_source=args.data_source,
            date=args.date,
            hour=args.hour,
            num_steps=args.num_steps,
            variables=variables,
            pressure_vars=pressure_vars,
            output_root=args.output_root,
            device=args.device,
            models_cfg_path=args.models_config,
            data_cfg_path=args.data_config,
            skip_plots=args.skip_plots,
        )


if __name__ == "__main__":
    main()
