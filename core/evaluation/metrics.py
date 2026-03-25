"""
定量评估指标模块。

核心设计：
  - 指标在滚动推理每步立即计算（无需重新加载 NPY），消除 evaluate_models.py 的重复计算
  - MetricsAccumulator 收集每步指标，最终输出 DataFrame + CSV + 时序图
  - diff npy/nc 生成通过 save_diff=True 控制，默认关闭
  - 可扩展：新指标只需在 METRIC_FUNCS 中注册一个函数

当前支持指标：
  W-MAE  — 余弦纬度加权平均绝对误差
  W-RMSE — 余弦纬度加权均方根误差

扩展示例（未来）：
  ACC（异常相关系数）— 需要气候态均值
  BIAS — 系统偏差
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 纬度权重工具
# ------------------------------------------------------------------ #

def _make_lat_weights(lats: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    生成余弦纬度权重二维数组和权重总和。
    lats: (H,) 降序数组（90→-90）
    返回: weights_2d (H,1)，sum_weights scalar
    """
    w = np.cos(np.deg2rad(lats)).astype(np.float64)[:, np.newaxis]  # (H, 1)
    return w, float(np.nansum(w))


# ------------------------------------------------------------------ #
# 单步指标函数
# ------------------------------------------------------------------ #

def _w_mae(diff: np.ndarray, w2d: np.ndarray, sum_w: float) -> float:
    # w2d shape (H,1) 广播到 diff shape (H,W)；分母必须对应相同的 H×W 总权重，
    # 与主分支 evaluate_models.py 的 broadcast_to(H,W) 后 nansum 一致。
    w_bc = np.broadcast_to(w2d, diff.shape)
    return float(np.nansum(np.abs(diff) * w_bc) / np.nansum(w_bc))


def _w_rmse(diff: np.ndarray, w2d: np.ndarray, sum_w: float) -> float:
    w_bc = np.broadcast_to(w2d, diff.shape)
    return float(np.sqrt(np.nansum((diff ** 2) * w_bc) / np.nansum(w_bc)))


# 注册表：指标名 → 计算函数(diff, w2d, sum_w) → float
METRIC_FUNCS: Dict[str, Callable] = {
    "W-MAE": _w_mae,
    "W-RMSE": _w_rmse,
}


def compute_step_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    lats: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    计算单步预报 vs 真值的指定指标。

    Args:
        pred:    (H, W) float32 预报场
        truth:   (H, W) float32 真值场
        lats:    (H,)   纬度数组（降序，90→-90）
        metrics: 指标名称列表（None=全部注册指标）

    Returns:
        diff:    (H, W) float32 差值场（pred - truth）
        results: {指标名: float} 字典
    """
    if metrics is None:
        metrics = list(METRIC_FUNCS.keys())

    diff = (np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64))
    w2d, sum_w = _make_lat_weights(lats)

    results: Dict[str, float] = {}
    for m in metrics:
        if m in METRIC_FUNCS:
            results[m] = METRIC_FUNCS[m](diff, w2d, sum_w)
        else:
            raise ValueError(f"未知指标 '{m}'，支持: {list(METRIC_FUNCS.keys())}")

    return diff.astype(np.float32), results


def register_metric(name: str, func: Callable) -> None:
    """注册自定义指标函数。func(diff, w2d, sum_w) -> float"""
    METRIC_FUNCS[name] = func


# ------------------------------------------------------------------ #
# 便捷类：收集多步多变量多模型指标
# ------------------------------------------------------------------ #

class WeightedMetrics:
    """单步指标计算的便捷包装。"""

    def __init__(self, lats: np.ndarray, metrics: Optional[List[str]] = None):
        self._lats = lats
        self._metrics = metrics or list(METRIC_FUNCS.keys())
        self._w2d, self._sum_w = _make_lat_weights(lats)

    def compute(
        self, pred: np.ndarray, truth: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        diff = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        results = {m: METRIC_FUNCS[m](diff, self._w2d, self._sum_w) for m in self._metrics}
        return diff.astype(np.float32), results


class MetricsAccumulator:
    """
    跨多步/多变量/多模型收集指标记录。

    用法：
        acc = MetricsAccumulator(lats)
        for step, ...:
            diff, mvals = acc.add(model, var, lead_hours, pred, truth)
        df = acc.to_dataframe()
        acc.save(save_dir, time_tag)
    """

    def __init__(
        self,
        lats: np.ndarray,
        metrics: Optional[List[str]] = None,
        save_diff: bool = False,
        save_diff_nc: bool = False,
    ):
        self._wm = WeightedMetrics(lats, metrics)
        self._save_diff = save_diff
        self._save_diff_nc = save_diff_nc
        self._records: List[dict] = []
        # {(model, var): [(lead_hours, diff_array), ...]}
        self._diffs: Dict[Tuple[str, str], List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self._lats = lats

    def add(
        self,
        model: str,
        var: str,
        lead_hours: int,
        pred: np.ndarray,
        truth: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        计算单步指标并记录。
        返回 (diff, metrics_dict)。
        """
        diff, mvals = self._wm.compute(pred, truth)

        record = {"Model": model, "Variable": var, "Lead_Time": lead_hours}
        record.update(mvals)
        self._records.append(record)

        if self._save_diff or self._save_diff_nc:
            self._diffs[(model, var)].append((lead_hours, diff))

        return diff, mvals

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def save(
        self,
        save_dir: Path,
        time_tag: str,
        lon: Optional[np.ndarray] = None,
        valid_times: Optional[Dict[Tuple[str, str], List]] = None,
    ) -> None:
        """
        保存 CSV、时序图，可选 diff npy/nc。
        save_dir:    eval 输出目录
        time_tag:    如 "20260308T12"
        lon:         经度数组（diff nc 需要）
        valid_times: {(model,var): [datetime, ...]}（diff nc 需要）
        """
        from zk_io.plot_utils import plot_metrics_timeseries
        from zk_io.nc_writer import write_diff_nc

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if df.empty:
            return

        # CSV
        csv_path = save_dir / f"timeseries_metrics_{time_tag}.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"[metrics] CSV 已保存: {csv_path}", flush=True)

        models = sorted(df["Model"].unique())
        variables = sorted(df["Variable"].unique())
        max_lead = int(df["Lead_Time"].max()) if not df.empty else 240

        # 时序图：每个注册指标出一张图
        for metric in self._wm._metrics:
            if metric not in df.columns:
                continue
            fname = f"{metric.replace('-','_')}_{time_tag}.png"
            plot_metrics_timeseries(
                df,
                metric_name=metric,
                y_label=metric,
                save_path=save_dir / fname,
                variables=variables,
                models=models,
                max_lead=max_lead,
            )
            print(f"[metrics] 图像已保存: {save_dir / fname}", flush=True)

        # diff 文件（可选）
        if self._save_diff or self._save_diff_nc:
            if lon is None:
                lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)

            for (model, var), step_diffs in self._diffs.items():
                step_diffs_sorted = sorted(step_diffs, key=lambda x: x[0])
                leads_arr = [s[0] for s in step_diffs_sorted]
                diff_stack = np.stack([s[1] for s in step_diffs_sorted], axis=0)
                base_name = f"{model}_{var}_{time_tag}_diff"

                if self._save_diff:
                    npy_path = save_dir / f"{base_name}.npy"
                    np.save(str(npy_path), diff_stack)
                    print(f"[metrics] diff npy: {npy_path}", flush=True)

                if self._save_diff_nc and valid_times is not None:
                    vts = valid_times.get((model, var), [])
                    if vts:
                        nc_path = save_dir / f"{base_name}.nc"
                        write_diff_nc(nc_path, diff_stack, vts, self._lats, lon, var)
                        print(f"[metrics] diff nc: {nc_path}", flush=True)
