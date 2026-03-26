"""
可视化工具。

plot_compare():           三联对比图（预报/真值/差值）
plot_metrics_timeseries(): W-RMSE / W-MAE 时序图（评估模块使用）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 字体配置（避免 CJK 乱码）
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
]

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def plot_compare(
    path: Path,
    pred: np.ndarray,
    truth: Optional[np.ndarray],
    *,
    title: str,
    cmap: str = "viridis",
) -> None:
    """
    三联对比图（预报 | ERA5 分析 | 差值）。
    truth=None 时只画单幅预报图。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(pred, dtype=np.float64)

    if truth is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(p, cmap=cmap, aspect="auto", origin="upper")
        ax.set_title(f"{title} | no ERA5 truth")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        fig.savefig(str(path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    t = np.asarray(truth, dtype=np.float64)
    d = p - t
    vmin = float(min(np.nanmin(p), np.nanmin(t)))
    vmax = float(max(np.nanmax(p), np.nanmax(t)))
    dv = float(np.nanpercentile(np.abs(d), 99))
    if not np.isfinite(dv) or dv < 1e-9:
        dv = max(float(np.nanmax(np.abs(d))), 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axs[0].imshow(p, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[0].set_title("Forecast")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)

    im1 = axs[1].imshow(t, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    axs[1].set_title("ERA5 analysis")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)

    im2 = axs[2].imshow(d, cmap="RdBu_r", aspect="auto", origin="upper", vmin=-dv, vmax=dv)
    axs[2].set_title("Forecast minus analysis")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_timeseries(
    df_metrics: pd.DataFrame,
    metric_name: str,
    y_label: str,
    save_path: Path,
    variables: List[str],
    models: List[str],
    *,
    max_lead: int = 240,
) -> None:
    """
    多变量多模型指标时序图。
    df_metrics 列：Model, Variable, Lead_Time, W-MAE, W-RMSE（至少包含 metric_name 列）。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(7 * n_vars + 1, 6))
    if n_vars == 1:
        axes = [axes]

    model_color = {m: _COLORS[i % len(_COLORS)] for i, m in enumerate(models)}

    for i, var in enumerate(variables):
        ax = axes[i]
        var_data = df_metrics[df_metrics["Variable"] == var]
        for model in models:
            md = var_data[var_data["Model"] == model].sort_values("Lead_Time")
            if not md.empty and metric_name in md.columns:
                ax.plot(
                    md["Lead_Time"], md[metric_name],
                    marker="o", markersize=3, linewidth=2,
                    label=model, color=model_color[model],
                )
        ax.set_title(f"{var.upper()} {metric_name} (Up to {max_lead}h)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Lead Time (Hours)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0, max_lead + 5)
        ax.set_xticks(np.arange(0, max_lead + 1, 24))

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_verify_compare(
    save_dir: Path,
    model_name: str,
    init_tag: str,
    step: int,
    lead_hours: int,
    pred_vars: Dict[str, np.ndarray],
    truth_vars: Optional[Dict[str, np.ndarray]],
) -> None:
    """
    功能一验证推理出图：对每个变量生成三联对比图。
    """
    save_dir = Path(save_dir)
    for var_name, pred_arr in pred_vars.items():
        truth_arr = None if truth_vars is None else truth_vars.get(var_name)
        out_path = save_dir / f"{var_name}_compare_step{step:02d}_lead{lead_hours:03d}h.png"
        plot_compare(
            out_path,
            pred_arr,
            truth_arr,
            title=f"{model_name} +{lead_hours}h {var_name} | {init_tag}",
            cmap="RdBu_r" if var_name in ("u10", "v10") else "viridis",
        )
