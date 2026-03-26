"""
功能一：初步推理验证流水线。

目的：快速跑通 1~N 步推理，出三联对比图，确认模型和数据输入管路正常。
对应原代码：run_four_models_test_era5.py

用法示例（Python 调用）：
    from pipelines.verify_pipeline import run_verify

    run_verify(
        model_names=["pangu", "fengwu"],
        data_source="test_era5",
        date="20260308",
        hour=12,
        num_steps=2,
        variables=["u10", "v10", "t2m", "msl"],
        output_root=Path("results/verify"),
        device="auto",
    )
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from runtime_paths import UNIFIED_PIPELINE_ROOT, GRAPH_CAST_ROOT

_ZK_ROOT = UNIFIED_PIPELINE_ROOT
_GRAPH_ROOT = GRAPH_CAST_ROOT
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

from core.data.detector import get_adapter
from core.data.channel_mapper import (
    PANGU_LEVELS,
    SURFACE_VAR_KEYS,
    PRESSURE_VAR_KEYS,
    extract_surface_vars,
    extract_pressure_var_at_level,
)
from core.models import build_registry
from zk_io.plot_utils import plot_compare, plot_verify_compare


def _verify_out_label(model_slug: str) -> str:
    """与 rolling NPY 目录命名一致（如 GraphCast_CS）。"""
    return {
        "pangu": "PanGu",
        "fengwu": "FengWu",
        "fuxi": "FuXi",
        "graphcast": "GraphCast",
        "graphcast_cs": "GraphCast_CS",
    }.get(model_slug.lower(), model_slug)


def _progress(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [verify] {msg}", flush=True)


def _load_data_cfg(cfg_path: Path) -> Dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_source(
    source_name: str,
    data_cfg_path: Optional[Path] = None,
) -> Tuple[Path, Optional[str], Dict]:
    """从 data.yaml 解析数据源根目录和格式。"""
    if data_cfg_path is None:
        data_cfg_path = _ZK_ROOT / "config" / "data.yaml"
    dcfg = _load_data_cfg(data_cfg_path)
    sources = dcfg.get("sources", {})
    if source_name in sources:
        src = sources[source_name]
        return Path(src["root"]), src.get("format"), src
    # 若 source_name 是路径，直接返回（格式自动探测）
    p = Path(source_name)
    if p.exists():
        return p, None, {}
    raise ValueError(f"未找到数据源 '{source_name}'，请检查 config/data.yaml 或提供有效路径")


def run_verify(
    model_names: List[str],
    data_source: str,
    date: str,
    hour: int,
    num_steps: int = 2,
    variables: Optional[List[str]] = None,
    pressure_vars: Optional[Dict[str, List[int]]] = None,
    output_root: Optional[Path] = None,
    device: Any = "auto",
    models_cfg_path: Optional[Path] = None,
    data_cfg_path: Optional[Path] = None,
    skip_plots: bool = False,
) -> None:
    """
    初步推理验证主函数。

    Args:
        model_names:    要运行的模型列表（如 ["pangu","fengwu"]）
        data_source:    数据源名称（config/data.yaml 中的 key）或路径字符串
        date:           推理日期 yyyymmdd
        hour:           起报时刻（UTC），如 12
        num_steps:      推理步数（1~N）
        variables:      地表变量名列表（None=使用模型默认地表变量）
        pressure_vars:  气压层变量选择，如 {"z":[1000],"t":[850,500]}（None=跳过气压场出图）
        output_root:    输出根目录（None 使用 config 默认值）
        device:         推理设备
        skip_plots:     跳过出图（仅推理不保存图片）
    """
    if models_cfg_path is None:
        models_cfg_path = _ZK_ROOT / "config" / "models.yaml"
    if output_root is None:
        output_root = _ZK_ROOT / "results" / "verify"

    # --- 加载数据适配器 ---
    data_root, data_fmt, src_cfg = _resolve_data_source(data_source, data_cfg_path)
    _progress(f"数据源: {data_root}（格式: {data_fmt or '自动探测'}）")
    adapter = get_adapter(
        data_root,
        fmt=data_fmt,
        use_monthly_subdir=src_cfg.get("use_monthly_subdir", False),
    )

    # 加载起报时刻 blob
    _progress(f"加载 {date} {hour:02d}Z 分析场...")
    init_blob = adapter.load_blob(date, hour)

    # 双帧模型需要 t-6h
    init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), hour)
    prev_dt = init_dt - timedelta(hours=6)
    prev_blob = adapter.load_blob_safe(prev_dt.strftime("%Y%m%d"), prev_dt.hour)

    # 逐模型加载（避免 GraphCast + GraphCast_CS 同时占满 VRAM）
    _progress(f"待验证模型: {model_names}")
    for model_name in model_names:
        _progress(f"[{model_name}] 加载权重...")
        registry = build_registry(models_cfg_path, device=device, only=[model_name])
        if model_name not in registry:
            _progress(f"模型 {model_name} 未加载，跳过")
            continue

        model = registry.get(model_name)
        label = _verify_out_label(model_name)
        try:
            step_h = model.get_step_hours()
            _progress(f"[{label}] 开始推理（{num_steps} 步，每步 {step_h}h）")

            try:
                state = model.init_state(init_blob, prev_blob=prev_blob, init_dt=init_dt)
            except ValueError as e:
                _progress(f"[{label}] init_state 失败: {e}")
                continue

            sfc_vars = variables if variables else model.get_surface_var_names()
            out_dir = Path(output_root) / label / f"{date}T{hour:02d}"

            for step in range(1, num_steps + 1):
                state = model.step(state)
                lead = state.lead
                valid_dt = init_dt + timedelta(hours=lead)
                truth_blob = adapter.load_blob_for_valid_time(valid_dt)

                _progress(f"[{label}] step={step}, lead={lead}h, valid={valid_dt.strftime('%Y-%m-%dT%H')}")

                if not skip_plots:
                    pred_sfc = extract_surface_vars(state.blob, sfc_vars)
                    truth_sfc = extract_surface_vars(truth_blob, sfc_vars) if truth_blob else None
                    plot_verify_compare(
                        save_dir=out_dir / "surface",
                        model_name=label,
                        init_tag=f"{date}T{hour:02d}",
                        step=step,
                        lead_hours=lead,
                        pred_vars=pred_sfc,
                        truth_vars=truth_sfc,
                    )

                    if pressure_vars and model.get_pressure_var_names():
                        for pvar, levels in pressure_vars.items():
                            if pvar not in PRESSURE_VAR_KEYS:
                                continue
                            for lev in levels:
                                if lev not in [int(l) for l in PANGU_LEVELS]:
                                    continue
                                pred_field = extract_pressure_var_at_level(state.blob, pvar, lev)
                                truth_field = None
                                if truth_blob:
                                    try:
                                        truth_field = extract_pressure_var_at_level(truth_blob, pvar, lev)
                                    except (KeyError, ValueError):
                                        pass
                                plot_compare(
                                    out_dir / "pressure" / f"{pvar}{lev}_step{step:02d}_lead{lead:03d}h.png",
                                    pred_field,
                                    truth_field,
                                    title=f"{label} +{lead}h {pvar}@{lev}hPa | {date}T{hour:02d}",
                                    cmap="viridis",
                                )

            _progress(f"[{label}] 推理验证完成，图像保存至: {out_dir}")
        finally:
            _progress(f"[{label}] 卸载模型…")
            model.unload()

    _progress("全部模型验证完成")
