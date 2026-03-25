#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytz

from cepri_loader import (
    FUXI_LEVELS,
    load_cepri_fuxi_fields,
    load_cepri_time,
    specific_humidity_to_relative_humidity,
)
from infer_cepri_onnx import (
    build_fengwu_onnx_combo_input,
    create_session,
    fuxi_prepare_onnx_input,
    fuxi_temb,
    pick_providers,
)


PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
L1000 = PANGU_LEVELS.index(1000)
FUXI_I1000 = FUXI_LEVELS.index(1000)


@dataclass
class Stat:
    corr: float
    rmse: float
    std: float
    mean: float


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ax = np.asarray(a, dtype=np.float64).reshape(-1)
    bx = np.asarray(b, dtype=np.float64).reshape(-1)
    av = float(np.std(ax))
    bv = float(np.std(bx))
    if av < 1e-12 or bv < 1e-12:
        return float("nan")
    return float(np.corrcoef(ax, bx)[0, 1])


def _stat(pred: np.ndarray, truth: np.ndarray) -> Stat:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return Stat(
        corr=_corr(p, t),
        rmse=float(np.sqrt(np.mean((p - t) ** 2))),
        std=float(np.std(p)),
        mean=float(np.mean(p)),
    )


def _truth_fields(era5_root: Path, date: str, hour: int) -> Dict[str, np.ndarray]:
    b = load_cepri_time(era5_root, date, hour)
    return {
        "sfc_u10": b["surface_u10"],
        "sfc_v10": b["surface_v10"],
        "sfc_t2m": b["surface_t2m"],
        "sfc_msl": b["surface_msl"],
        "h1000_z": b["pangu_z"][L1000],
        "h1000_u": b["pangu_u"][L1000],
        "h1000_v": b["pangu_v"][L1000],
        "h1000_t": b["pangu_t"][L1000],
        "h1000_r": specific_humidity_to_relative_humidity(b["pangu_q"][L1000], b["pangu_t"][L1000], 1000.0),
    }


def _split_69(o69: np.ndarray) -> Dict[str, np.ndarray]:
    # 4 + 13x5 (z,r,u,v,t)
    return {
        "sfc0": o69[0],
        "sfc1": o69[1],
        "sfc2": o69[2],
        "sfc3": o69[3],
        "z1000": o69[4 + L1000],
        "r1000": o69[17 + L1000],
        "u1000": o69[30 + L1000],
        "v1000": o69[43 + L1000],
        "t1000": o69[56 + L1000],
    }


def diagnose_fengwu(era5_root: Path, date: str, hour: int, providers: List) -> None:
    onnx_p = Path(__file__).resolve().parent / "fengwu" / "fengwu_v2.onnx"
    sess = create_session(onnx_p, providers)
    x = build_fengwu_onnx_combo_input(era5_root, date, hour, sess)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    if y.ndim == 4 and y.shape[0] == 1:
        y = y[0]
    if y.shape[0] != 138:
        print(f"[fengwu] unexpected output shape: {y.shape}")
        return

    # 双帧输入按 (h,h+6)，单步通常对应锚点后 6h -> truth at h+12
    truth = _truth_fields(era5_root, date, min(hour + 12, 23))
    half0 = _split_69(y[:69])
    half1 = _split_69(y[69:138])

    print("\n[fengwu] ---- half0 (channels 0:69) ----")
    _print_half_stats(half0, truth)
    print("\n[fengwu] ---- half1 (channels 69:138) ----")
    _print_half_stats(half1, truth)

    # 自动建议哪一半更像有效预报（以 5 个关键变量相关系数绝对值均值衡量）
    score0 = _score_half(half0, truth)
    score1 = _score_half(half1, truth)
    pick = "half0" if score0 >= score1 else "half1"
    print(f"\n[fengwu] suggest use {pick}: score0={score0:.4f}, score1={score1:.4f}")


def _score_half(h: Dict[str, np.ndarray], t: Dict[str, np.ndarray]) -> float:
    pairs = [
        (h["z1000"], t["h1000_z"]),
        (h["u1000"], t["h1000_u"]),
        (h["v1000"], t["h1000_v"]),
        (h["r1000"], t["h1000_r"]),
        (h["t1000"], t["h1000_t"]),
    ]
    vals = []
    for p, q in pairs:
        c = _corr(p, q)
        vals.append(0.0 if not np.isfinite(c) else abs(c))
    return float(np.mean(vals))


def _print_half_stats(h: Dict[str, np.ndarray], t: Dict[str, np.ndarray]) -> None:
    # surface channels unknown mapping -> report best match among 4 truth vars
    s_truth = {
        "sfc_u10": t["sfc_u10"],
        "sfc_v10": t["sfc_v10"],
        "sfc_t2m": t["sfc_t2m"],
        "sfc_msl": t["sfc_msl"],
    }
    for i, key in enumerate(("sfc0", "sfc1", "sfc2", "sfc3")):
        best_name = None
        best_corr = -1.0
        best_stat = None
        for nm, arr in s_truth.items():
            st = _stat(h[key], arr)
            cabs = abs(st.corr) if np.isfinite(st.corr) else -1.0
            if cabs > best_corr:
                best_corr = cabs
                best_name = nm
                best_stat = st
        print(
            f"  {key} -> best {best_name}: corr={best_stat.corr:.4f}, rmse={best_stat.rmse:.4g}, "
            f"std={best_stat.std:.4g}, mean={best_stat.mean:.4g}"
        )
    for nm_pred, nm_truth in (
        ("z1000", "h1000_z"),
        ("u1000", "h1000_u"),
        ("v1000", "h1000_v"),
        ("r1000", "h1000_r"),
        ("t1000", "h1000_t"),
    ):
        st = _stat(h[nm_pred], t[nm_truth])
        print(f"  {nm_pred}: corr={st.corr:.4f}, rmse={st.rmse:.4g}, std={st.std:.4g}, mean={st.mean:.4g}")


def diagnose_fuxi(era5_root: Path, date: str, hour: int, providers: List, lead_hours: int) -> None:
    onnx_p = Path(__file__).resolve().parent / "fuxi" / "short.onnx"
    sess = create_session(onnx_p, providers)
    h0, h1 = hour, min(23, hour + 6)
    raw = load_cepri_fuxi_fields(era5_root, date, h0, h1)
    x, layout = fuxi_prepare_onnx_input(raw, sess)
    temb = fuxi_temb(lead_hours)
    feeds = {}
    for inp in sess.get_inputs():
        if "temb" in inp.name.lower():
            feeds[inp.name] = temb
        else:
            feeds[inp.name] = x.astype(np.float32)
    y = sess.run(None, feeds)[0]
    print(f"\n[fuxi] input layout={layout}, output shape={y.shape}")
    if y.ndim != 5 or y.shape[0] != 1:
        print("[fuxi] unexpected output layout, skip")
        return

    # 对两个时间位都诊断，避免选错 time index
    for tidx in range(min(2, y.shape[1])):
        out = y[0, tidx]
        truth = _truth_fields(era5_root, date, min(hour + 12 + 6 * tidx, 23))
        print(f"\n[fuxi] ---- time index {tidx} ----")
        mapping = {
            "sfc_u10": out[0],
            "sfc_v10": out[1],
            "sfc_t2m": out[2],
            "sfc_msl": out[3],
            "h1000_z": out[5 + FUXI_I1000],
            "h1000_r": out[5 + 13 + FUXI_I1000],
            "h1000_u": out[5 + 26 + FUXI_I1000],
            "h1000_v": out[5 + 39 + FUXI_I1000],
            "h1000_t": out[5 + 52 + FUXI_I1000],
        }
        for k, p in mapping.items():
            st = _stat(p, truth[k])
            print(f"  {k}: corr={st.corr:.4f}, rmse={st.rmse:.4g}, std={st.std:.4g}, mean={st.mean:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-root", type=Path, default=Path(__file__).resolve().parent.parent / "test_era5_data")
    ap.add_argument("--date", default="20260301")
    ap.add_argument("--hour", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["auto", "dcu", "cuda", "cpu"])
    ap.add_argument("--lead-hours", type=int, default=6)
    args = ap.parse_args()

    providers = pick_providers(args.device)
    print("providers:", providers)
    diagnose_fengwu(args.era5_root, args.date, args.hour, providers)
    diagnose_fuxi(args.era5_root, args.date, args.hour, providers, args.lead_hours)


if __name__ == "__main__":
    main()

