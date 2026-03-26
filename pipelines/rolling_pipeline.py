"""
功能二：滚动推理流水线。

可定制：
  - 模型选择（多模型串行）
  - 变量选择（地表变量，或全部）
  - 时间范围（单日 / 日期范围）
  - 起报时刻（默认 12h）
  - 步长（默认 6h）
  - 总时长（默认 240h）
  - 输出路径（兼容 GunDong_Infer_result_12h 格式）
  - 可选：内嵌定量评估（--enable-eval）
  - 可选：diff npy/nc 保存（--save-diff）

多卡并行：通过 RANK/WORLD_SIZE 环境变量分片日期，与 Slurm torchrun 配合使用。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_ZK_ROOT = Path(__file__).resolve().parents[1]
_GRAPH_ROOT = _ZK_ROOT.parent
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))
os.chdir(_GRAPH_ROOT)

from core.data.detector import get_adapter
from core.data.channel_mapper import extract_surface_vars
from core.data.surface_units import harmonize_surface_pair
from core.models import build_registry
from core.evaluation.metrics import MetricsAccumulator
from zk_io.npy_writer import NpyStackWriter
from zk_io.nc_writer import write_step_nc
from zk_io.plot_utils import plot_compare

# 与 evaluate_models.py 中 MODELS 顺序一致，用于多模型时序图配色/图例
_EVAL_MODEL_ORDER = ["PanGu", "FengWu", "FuXi", "GraphCast"]


def _progress(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    prefix = f"[rank{rank}/{world}]" if world > 1 else ""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [rolling]{prefix} {msg}", flush=True)


def _shard_dates(dates: List[str]) -> List[str]:
    """按 RANK/WORLD_SIZE 分片日期：rank i 处理 dates[i::world]。"""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return [d for i, d in enumerate(dates) if i % world == rank]


def _shard_models(models: List[str]) -> List[str]:
    """按 RANK/WORLD_SIZE 分片模型：rank i 处理 models[i::world]。
    单日多模型并行时使用，每张卡独立推理一组模型。"""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assigned = [m for i, m in enumerate(models) if i % world == rank]
    if not assigned:
        _progress(f"无分配模型（共 {len(models)} 个模型，{world} 个 rank），退出")
    return assigned


def _decide_parallel_mode(mode: str, dates: List[str], models: List[str]) -> str:
    """
    决定并行策略：
      date  — 每个 rank 处理不同日期（多日任务）
      model — 每个 rank 处理不同模型（单日或模型多于日期时）
      auto  — 自动判断：dates >= WORLD_SIZE → date；否则 → model
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if mode == "auto":
        return "date" if len(dates) >= world else "model"
    return mode


def _load_data_source(source_name: str, data_cfg_path: Optional[Path] = None):
    """返回 (root, fmt, src_cfg)。"""
    if data_cfg_path is None:
        data_cfg_path = _ZK_ROOT / "config" / "data.yaml"
    with open(data_cfg_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    sources = dcfg.get("sources", {})
    if source_name in sources:
        src = sources[source_name]
        return Path(src["root"]), src.get("format"), src
    p = Path(source_name)
    if p.exists():
        return p, None, {}
    raise ValueError(f"未找到数据源 '{source_name}'")


def _parse_date_range(date_range: str) -> List[str]:
    """
    解析日期范围字符串：
      '20260308'          → ['20260308']
      '20260301:20260318' → ['20260301', '20260302', ..., '20260318']
    """
    if ":" in date_range:
        start_s, end_s = date_range.split(":", 1)
        start = datetime.strptime(start_s.strip(), "%Y%m%d")
        end = datetime.strptime(end_s.strip(), "%Y%m%d")
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return dates
    return [date_range.strip()]


def run_rolling(
    model_names: List[str],
    data_source: str,
    date_range: str,
    init_hour: int = 12,
    lead_step: int = 6,
    max_lead: int = 240,
    variables: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    device: Any = "auto",
    skip_plots: bool = False,
    save_nc: bool = False,
    enable_eval: bool = False,
    save_diff: bool = False,
    save_diff_nc: bool = False,
    metrics: Optional[List[str]] = None,
    models_cfg_path: Optional[Path] = None,
    data_cfg_path: Optional[Path] = None,
    parallel_mode: str = "auto",
) -> None:
    """
    滚动推理主函数。

    Args:
        model_names:    模型名称列表（如 ["fengwu","fuxi"]）
        data_source:    数据源名称或路径
        date_range:     日期 'yyyymmdd' 或范围 'yyyymmdd:yyyymmdd'
        init_hour:      起报时刻 UTC（默认 12）
        lead_step:      步长小时（默认 6）
        max_lead:       总预报时长小时（默认 240）
        variables:      地表变量列表（None=使用模型默认全部地表变量）
        output_root:    结果根目录（兼容 GunDong_Infer_result_12h 布局）
        device:         推理设备
        skip_plots:     跳过逐步对比图
        save_nc:        是否保存 per-step NetCDF（Pangu/GraphCast 推荐开启）
        enable_eval:    是否开启内嵌定量评估（W-RMSE/W-MAE CSV + 时序图）
        save_diff:      是否保存 diff npy 文件（需 enable_eval=True）
        save_diff_nc:   是否保存 diff nc 文件（需 enable_eval=True）
        metrics:        指标列表（None=["W-MAE","W-RMSE"]）
        parallel_mode:  多卡并行策略（auto | date | model）
                          auto  — 日期数 >= WORLD_SIZE → date 模式；否则 → model 模式
                          date  — 每 rank 处理不同日期（多日任务推荐）
                          model — 每 rank 处理不同模型（单日多模型推荐）
    """
    if models_cfg_path is None:
        models_cfg_path = _ZK_ROOT / "config" / "models.yaml"
    if output_root is None:
        output_root = Path("/public/share/aciwgvx1jd/GunDong_Infer_result_12h")
    if metrics is None:
        metrics = ["W-MAE", "W-RMSE"]

    n_steps = max_lead // lead_step
    leads = list(range(lead_step, max_lead + 1, lead_step))

    # --- 解析日期（分片前先拿全量，供 auto 模式判断）---
    all_dates = _parse_date_range(date_range)

    # --- 决定并行策略 ---
    pmode = _decide_parallel_mode(parallel_mode, all_dates, model_names)
    _progress(f"并行模式: {pmode} (WORLD_SIZE={os.environ.get('WORLD_SIZE','1')})")

    # --- 按策略分片 ---
    if pmode == "model":
        dates = all_dates
        model_names = _shard_models(model_names)
        if not model_names:
            return
    else:
        dates = _shard_dates(all_dates)
        if not dates:
            _progress("当前 RANK 无分配日期，退出")
            return

    # --- 加载数据适配器（轻量，无 VRAM 占用，一次性初始化）---
    data_root, data_fmt, src_cfg = _load_data_source(data_source, data_cfg_path)
    _progress(f"数据源: {data_root}（格式: {data_fmt or '自动探测'}）")
    adapter = get_adapter(
        data_root, fmt=data_fmt,
        use_monthly_subdir=src_cfg.get("use_monthly_subdir", False),
    )

    _progress(f"处理日期: {dates}  模型: {model_names}")

    # 按 init_tag 共享 MetricsAccumulator：所有模型跑完后一次性写 CSV/时序图，避免相互覆盖
    acc_by_tag: Dict[str, MetricsAccumulator] = {}
    valid_times_by_tag: Dict[str, Dict[Tuple[str, str], List]] = {}
    lon_by_tag: Dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------
    # 外层循环：模型；内层循环：日期
    # 每个模型独占一次 VRAM：加载 → 跑完所有日期 → 卸载 → 下一个模型
    # 避免多模型同时占用 VRAM 导致 OOM。
    # ---------------------------------------------------------------
    for model_name in model_names:
        display_name = {
            "pangu": "PanGu", "fengwu": "FengWu",
            "fuxi": "FuXi", "graphcast": "GraphCast",
        }.get(model_name.lower(), model_name)

        # 逐模型加载
        _progress(f"[{display_name}] 加载模型权重...")
        registry = build_registry(models_cfg_path, device=device, only=[model_name])
        if model_name not in registry:
            _progress(f"[{display_name}] 加载失败，跳过")
            continue
        model = registry.get(model_name)
        sfc_vars = variables if variables else model.get_surface_var_names()
        step_h = model.get_step_hours()
        is_pangu = model_name.lower() == "pangu"

        try:
            for date in dates:
                init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), init_hour)
                init_tag = init_dt.strftime("%Y%m%dT%H")

                # 加载起报 blob
                try:
                    init_blob = adapter.load_blob(date, init_hour)
                except FileNotFoundError as e:
                    _progress(f"[{display_name}] 跳过 {init_tag}：{e}")
                    continue

                prev_dt = init_dt - timedelta(hours=lead_step)
                prev_blob = adapter.load_blob_safe(prev_dt.strftime("%Y%m%d"), prev_dt.hour)

                lat = init_blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
                lon = init_blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))
                lon_by_tag[init_tag] = lon

                _progress(f"[{display_name}] 开始滚动推理 {init_tag}，步长={step_h}h，共 {n_steps} 步")

                # 初始化模型状态
                try:
                    state = model.init_state(init_blob, prev_blob=prev_blob, init_dt=init_dt)
                except ValueError as e:
                    _progress(f"[{display_name}] init_state 失败: {e}")
                    continue

                if enable_eval:
                    if init_tag not in acc_by_tag:
                        acc_by_tag[init_tag] = MetricsAccumulator(
                            lats=lat,
                            metrics=metrics,
                            save_diff=save_diff,
                            save_diff_nc=save_diff_nc,
                        )
                        valid_times_by_tag[init_tag] = {}
                acc = acc_by_tag.get(init_tag) if enable_eval else None

                # NPY 写出器
                with NpyStackWriter(
                    output_root=Path(output_root),
                    model_name=display_name,
                    init_tag=init_tag,
                    variables=sfc_vars,
                    n_steps=n_steps,
                    shape_hw=(len(lat), len(lon)),
                    pangu_suffix=is_pangu,
                ) as npy_writer:

                    for si, lead in enumerate(leads):
                        steps_needed = lead_step // step_h
                        for _ in range(steps_needed):
                            state = model.step(state)

                        current_lead = state.lead
                        valid_dt = init_dt + timedelta(hours=current_lead)
                        truth_blob = adapter.load_blob_for_valid_time(valid_dt)

                        pred_sfc = extract_surface_vars(state.blob, sfc_vars)
                        npy_writer.write_step(si, pred_sfc)

                        # NC 写出
                        if save_nc:
                            nc_dir = Path(output_root) / init_tag / "nc" / model_name.lower()
                            vars_2d = {f"sfc_{k}": v for k, v in pred_sfc.items()}
                            vars_3d = None
                            level_vals = None
                            if model.get_pressure_var_names():
                                from cepri_loader import PANGU_LEVELS
                                vars_3d = {}
                                for pv in model.get_pressure_var_names():
                                    bkey = f"pangu_{pv}"
                                    if bkey in state.blob:
                                        vars_3d[f"pres_{pv}"] = state.blob[bkey]
                                level_vals = np.asarray(PANGU_LEVELS, dtype=np.float32) if vars_3d else None
                            write_step_nc(
                                nc_dir / f"lead_{current_lead:03d}.nc",
                                model=display_name,
                                init_time=init_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                lead_hours=current_lead,
                                valid_time=valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                vars_2d=vars_2d,
                                vars_3d=vars_3d if vars_3d else None,
                                level_values=level_vals,
                                lat=lat,
                                lon=lon,
                            )

                        # 对比图
                        if not skip_plots:
                            truth_sfc = extract_surface_vars(truth_blob, sfc_vars) if truth_blob else None
                            plot_dir = Path(output_root) / "plots" / model_name.lower() / init_tag
                            for var_name, pred_arr in pred_sfc.items():
                                truth_arr = truth_sfc.get(var_name) if truth_sfc else None
                                if truth_arr is not None:
                                    pred_arr, truth_arr = harmonize_surface_pair(
                                        var_name, pred_arr, truth_arr
                                    )
                                plot_compare(
                                    plot_dir / f"{var_name}_lead{current_lead:03d}.png",
                                    pred_arr,
                                    truth_arr,
                                    title=f"{display_name} +{current_lead}h {var_name} | {init_tag}",
                                    cmap="RdBu_r" if var_name in ("u10", "v10") else "viridis",
                                )

                        # 内嵌评估
                        if acc is not None and truth_blob is not None:
                            truth_sfc_eval = extract_surface_vars(truth_blob, sfc_vars)
                            for var_name, pred_arr in pred_sfc.items():
                                if var_name in truth_sfc_eval:
                                    p_h, t_h = harmonize_surface_pair(
                                        var_name, pred_arr, truth_sfc_eval[var_name]
                                    )
                                    acc.add(
                                        display_name, var_name, current_lead, p_h, t_h
                                    )
                                    if save_diff_nc:
                                        valid_times_by_tag[init_tag].setdefault(
                                            (display_name, var_name), []
                                        ).append(valid_dt)

                        if si % 8 == 0 or si == 0:
                            _progress(f"[{display_name}] {init_tag} lead={current_lead}h done")

                _progress(f"[{display_name}] {init_tag} NPY 已保存: {npy_writer.get_paths()}")

                _progress(f"[{display_name}] 完成 {init_tag}")

        finally:
            # 无论推理是否成功都卸载，确保 VRAM 释放给下一个模型
            _progress(f"[{display_name}] 卸载模型，释放 VRAM...")
            model.unload()

    if enable_eval and acc_by_tag:
        for init_tag, acc in acc_by_tag.items():
            eval_out_dir = Path(output_root) / f"eval_{max_lead}h_{init_tag}"
            acc.save(
                save_dir=eval_out_dir,
                time_tag=init_tag,
                lon=lon_by_tag.get(init_tag),
                valid_times=valid_times_by_tag.get(init_tag) if save_diff_nc else None,
                model_order=_EVAL_MODEL_ORDER,
            )
            _progress(f"多模型评估已写入: {eval_out_dir}")

    _progress("全部模型/日期处理完成")
