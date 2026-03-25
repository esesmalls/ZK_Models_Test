#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
class Score:
    corr: float
    rmse: float
    std: float
    mean: float
    affine_corr: float
    affine_rmse: float
    a: float
    b: float


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    vx = float(np.var(xv))
    if vx < 1e-12:
        return 0.0, float(np.mean(yv))
    cov = float(np.mean((xv - xv.mean()) * (yv - yv.mean())))
    a = cov / vx
    b = float(yv.mean() - a * xv.mean())
    return a, b


def score(pred: np.ndarray, truth: np.ndarray) -> Score:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    c = corr2(p, t)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    a, b = fit_affine(p, t)
    p2 = a * p + b
    c2 = corr2(p2, t)
    rmse2 = float(np.sqrt(np.mean((p2 - t) ** 2)))
    return Score(
        corr=c,
        rmse=rmse,
        std=float(np.std(p)),
        mean=float(np.mean(p)),
        affine_corr=c2,
        affine_rmse=rmse2,
        a=a,
        b=b,
    )


def truth_blob(era5_root: Path, date: str, hour: int) -> Dict[str, np.ndarray]:
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


def split_69(o69: np.ndarray) -> Dict[str, np.ndarray]:
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
        "t_block": o69[56:69],
    }


def fmt(s: Score) -> str:
    return (
        f"corr={s.corr:.4f}, rmse={s.rmse:.4g}, std={s.std:.4g}, mean={s.mean:.4g}, "
        f"affine(corr={s.affine_corr:.4f}, rmse={s.affine_rmse:.4g}, a={s.a:.4g}, b={s.b:.4g})"
    )


def diagnose_fengwu(era5_root: Path, date: str, hour: int, providers: List[str]) -> str:
    out = []
    sess = create_session(Path(__file__).resolve().parent / "fengwu" / "fengwu_v2.onnx", providers)
    x = build_fengwu_onnx_combo_input(era5_root, date, hour, sess)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    if y.ndim == 4:
        y = y[0]
    if y.shape[0] != 138:
        return f"[fengwu] unexpected output shape: {y.shape}\n"

    # evaluate both half and both anchors (+6,+12 from hour)
    for hname, hslice in (("half0", y[:69]), ("half1", y[69:138])):
        h = split_69(hslice)
        out.append(f"\n[fengwu] {hname}")
        for lead in (6, 12):
            tb = truth_blob(era5_root, date, min(hour + lead, 23))
            out.append(f"  lead=+{lead}h")
            # surface mapping candidates
            for sk in ("sfc0", "sfc1", "sfc2", "sfc3"):
                best = None
                best_key = None
                for tk in ("sfc_u10", "sfc_v10", "sfc_t2m", "sfc_msl"):
                    sc = score(h[sk], tb[tk])
                    v = abs(sc.corr) if np.isfinite(sc.corr) else -1.0
                    if best is None or v > best:
                        best = v
                        best_key = (tk, sc)
                out.append(f"    {sk} -> best {best_key[0]}: {fmt(best_key[1])}")

            for pk, tk in (("z1000", "h1000_z"), ("u1000", "h1000_u"), ("v1000", "h1000_v"), ("r1000", "h1000_r")):
                out.append(f"    {pk}: {fmt(score(h[pk], tb[tk]))}")

            # t1000 direct and best-in-temperature-block
            s_direct = score(h["t1000"], tb["h1000_t"])
            out.append(f"    t1000_direct: {fmt(s_direct)}")
            best_li = 0
            best_sc = score(h["t_block"][0], tb["h1000_t"])
            for li in range(1, 13):
                sc = score(h["t_block"][li], tb["h1000_t"])
                v = abs(sc.corr) if np.isfinite(sc.corr) else -1.0
                vb = abs(best_sc.corr) if np.isfinite(best_sc.corr) else -1.0
                if v > vb:
                    best_li = li
                    best_sc = sc
            out.append(f"    t1000_best_in_t_block[level_idx={best_li}]: {fmt(best_sc)}")
    return "\n".join(out) + "\n"


def diagnose_fuxi(era5_root: Path, date: str, hour: int, providers: List[str], lead_hours: int) -> str:
    out = []
    sess = create_session(Path(__file__).resolve().parent / "fuxi" / "short.onnx", providers)
    h0 = hour
    h1 = min(23, hour + 6)
    raw = load_cepri_fuxi_fields(era5_root, date, h0, h1)
    x, layout = fuxi_prepare_onnx_input(raw, sess)
    feeds = {}
    for inp in sess.get_inputs():
        if "temb" in inp.name.lower():
            feeds[inp.name] = fuxi_temb(lead_hours)
        else:
            feeds[inp.name] = x.astype(np.float32)
    y = sess.run(None, feeds)[0]
    out.append(f"\n[fuxi] layout={layout}, output_shape={y.shape}")
    if y.ndim != 5:
        return "\n".join(out) + "\n"

    for tidx in range(min(2, y.shape[1])):
        outv = y[0, tidx]
        tb = truth_blob(era5_root, date, min(hour + 12 + 6 * tidx, 23))
        out.append(f"  time_index={tidx}")
        # surface candidates
        sfc = [outv[0], outv[1], outv[2], outv[3]]
        for i, a in enumerate(sfc):
            best = None
            best_key = None
            for tk in ("sfc_u10", "sfc_v10", "sfc_t2m", "sfc_msl"):
                sc = score(a, tb[tk])
                v = abs(sc.corr) if np.isfinite(sc.corr) else -1.0
                if best is None or v > best:
                    best = v
                    best_key = (tk, sc)
            out.append(f"    sfc{i} -> best {best_key[0]}: {fmt(best_key[1])}")

        out.append(f"    z1000: {fmt(score(outv[5 + FUXI_I1000], tb['h1000_z']))}")
        out.append(f"    r1000: {fmt(score(outv[5 + 13 + FUXI_I1000], tb['h1000_r']))}")
        out.append(f"    u1000: {fmt(score(outv[5 + 26 + FUXI_I1000], tb['h1000_u']))}")
        out.append(f"    v1000: {fmt(score(outv[5 + 39 + FUXI_I1000], tb['h1000_v']))}")
        t_direct = score(outv[5 + 52 + FUXI_I1000], tb["h1000_t"])
        out.append(f"    t1000_direct: {fmt(t_direct)}")
        # temp block best
        tblock = [outv[5 + 52 + li] for li in range(13)]
        best_li = 0
        best_sc = score(tblock[0], tb["h1000_t"])
        for li in range(1, 13):
            sc = score(tblock[li], tb["h1000_t"])
            v = abs(sc.corr) if np.isfinite(sc.corr) else -1.0
            vb = abs(best_sc.corr) if np.isfinite(best_sc.corr) else -1.0
            if v > vb:
                best_li = li
                best_sc = sc
        out.append(f"    t1000_best_in_t_block[level_idx={best_li}]: {fmt(best_sc)}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-root", type=Path, default=Path(__file__).resolve().parent.parent / "test_era5_data")
    ap.add_argument("--date", default="20260301")
    ap.add_argument("--hour", type=int, default=0)
    ap.add_argument("--device", default="dcu", choices=["auto", "dcu", "cuda", "cpu"])
    ap.add_argument("--lead-hours", type=int, default=6)
    ap.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "logs" / "fw_fx_mapping_report.md")
    args = ap.parse_args()

    providers = pick_providers(args.device)
    txt = [f"# Fengwu/Fuxi mapping diagnosis\n", f"providers: {providers}\n"]
    txt.append(diagnose_fengwu(args.era5_root, args.date, args.hour, providers))
    txt.append(diagnose_fuxi(args.era5_root, args.date, args.hour, providers, args.lead_hours))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(txt), encoding="utf-8")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()

