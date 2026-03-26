#!/usr/bin/env python3
"""
ONNX inference for ZK_Models (Pangu / Fengwu / Fuxi) using CEPRI ERA5 NetCDF.

阶梯调度与 I/O 命名对齐 `example.py` 中的 Pangu 逻辑；ZK 目录无 `pangu_weather_3.onnx` 时，
在原本使用 3h 模型的步上用 **1h 模型** 代替（与官方复合步长略有差异，仅作联调用）。

建议在 **DCU 计算节点** 上运行：加载 `compiler/dtk/25.04` 后，若安装了带 ROCm/MIGraphX 的
onnxruntime，将自动优先使用；若缺少对应 EP，默认直接报错退出（避免误回退 CPU）。

推荐环境：`conda activate torch2.4_dtk25.04_cp310_e2s`（与 graphcast / e2s 脚本一致）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("需要 onnxruntime：在 e2s 环境中安装，见 run_zk_infer_slurm_dcu.sh 注释。", file=sys.stderr)
    raise

# 全局 logger：抑制 constant_folding 等对 Clip 节点的刷屏 [W:onnxruntime:...]（会话级 log_severity_level 未必覆盖全部）
# 0=VERBOSE … 3=ERROR；需要调试 ORT 时 export ORT_LOG_SEVERITY_LEVEL=2
_ort_log = int(os.environ.get("ORT_LOG_SEVERITY_LEVEL", "3"))
if hasattr(ort, "set_default_logger_severity"):
    ort.set_default_logger_severity(_ort_log)

from cepri_loader import (
    FUXI_LEVELS,
    load_cepri_fengwu_fields,
    load_cepri_fuxi_fields,
    load_cepri_time,
    pack_pangu_onnx,
)

from runtime_paths import ZK_MODELS_ROOT

ZK_ROOT = ZK_MODELS_ROOT


def _fengwu_69_from_blob_q_order(blob: Dict[str, np.ndarray]) -> np.ndarray:
    """
    按 GitHub 说明构造 69 通道：
    [u10,v10,t2m,msl, z(50..1000), q(50..1000), u(50..1000), v(50..1000), t(50..1000)]
    其中 blob 的 pangu_* 层序为 [1000..50]，这里重排为 [50..1000]。
    """
    levels_src = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    order = [levels_src.index(int(lv)) for lv in FUXI_LEVELS]  # 50..1000
    sfc = np.stack(
        [blob["surface_u10"], blob["surface_v10"], blob["surface_t2m"], blob["surface_msl"]],
        axis=0,
    ).astype(np.float32)
    z = blob["pangu_z"][order].astype(np.float32)
    q = blob["pangu_q"][order].astype(np.float32)
    u = blob["pangu_u"][order].astype(np.float32)
    v = blob["pangu_v"][order].astype(np.float32)
    t = blob["pangu_t"][order].astype(np.float32)
    return np.concatenate([sfc, z, q, u, v, t], axis=0).astype(np.float32)


def fuxi_normalize_for_layout(x5d: np.ndarray, stats_dir: Path, layout: str) -> np.ndarray:
    """
    对 Fuxi 输入做归一化，兼容两种常见布局：
      - layout='NTCHW': x.shape=(1,2,70,H,W)  -> mu/sd 扩到 (1,1,70,1,1)
      - layout='NCTHW': x.shape=(1,70,2,H,W)  -> mu/sd 扩到 (1,70,1,1,1)
    """
    mu = np.load(stats_dir / "global_means.npy")[:, :70, :, :].astype(np.float32)
    sd = np.load(stats_dir / "global_stds.npy")[:, :70, :, :].astype(np.float32)
    if layout == "NTCHW":
        mu = mu[:, np.newaxis, :, :, :]
        sd = sd[:, np.newaxis, :, :, :]
    elif layout == "NCTHW":
        mu = mu[:, :, np.newaxis, :, :]
        sd = sd[:, :, np.newaxis, :, :]
    else:
        raise ValueError(f"unknown Fuxi layout: {layout}")
    return (x5d.astype(np.float32) - mu) / np.maximum(sd, 1e-6)


def fuxi_prepare_onnx_input(raw_tchw: np.ndarray, sess: ort.InferenceSession) -> Tuple[np.ndarray, str]:
    """
    raw_tchw: (2,70,H,W) from CEPRI loader.
    Returns:
      x5d: 按 ONNX 期望排列后的输入
      layout: 'NTCHW' 或 'NCTHW'
    """
    x_ntchw = raw_tchw[np.newaxis, ...].astype(np.float32)  # (1,2,70,H,W)
    data_inp = None
    for inp in sess.get_inputs():
        if "temb" not in inp.name.lower():
            data_inp = inp
            break
    if data_inp is None:
        # 保守回退：多数导出是 NTCHW
        return x_ntchw, "NTCHW"

    shape = list(data_inp.shape)
    if len(shape) >= 3:
        d1, d2 = shape[1], shape[2]
        if isinstance(d1, int) and isinstance(d2, int):
            if d1 == 2 and d2 == 70:
                return x_ntchw, "NTCHW"
            if d1 == 70 and d2 == 2:
                return np.transpose(x_ntchw, (0, 2, 1, 3, 4)), "NCTHW"

    # 动态维或未知布局时，优先 NTCHW（与当前 short.onnx 一致）
    return x_ntchw, "NTCHW"


def fengwu_normalize_for_onnx(x_nchw: np.ndarray, stats_dir: Path) -> np.ndarray:
    """
    Fengwu 输入归一化（可选）：
    - x_nchw 可能是 (1,138,H,W) 或 (1,189,H,W)
    - stats 期望形状 [1,C,1,1]，C 可为 69/138/189
    """
    c_in = int(x_nchw.shape[1])
    # Prefer workspace / GraphCast-style global stats (legacy pipeline); else official 69-vec files.
    gm = stats_dir / "global_means.npy"
    gs = stats_dir / "global_stds.npy"
    if gm.is_file() and gs.is_file():
        mu = np.load(gm).astype(np.float32)
        sd = np.load(gs).astype(np.float32)
        if mu.ndim != 4 or sd.ndim != 4:
            raise ValueError(f"fengwu stats shape invalid: means={mu.shape}, stds={sd.shape}")
        c_stats = int(mu.shape[1])
        if c_stats == c_in:
            m = mu[:, :c_in]
            s = sd[:, :c_in]
        elif c_in == 138 and c_stats == 69:
            m = np.concatenate([mu[:, :69], mu[:, :69]], axis=1)
            s = np.concatenate([sd[:, :69], sd[:, :69]], axis=1)
        else:
            raise ValueError(f"fengwu stats channels mismatch: input={c_in}, stats={c_stats}")
        return (x_nchw.astype(np.float32) - m) / np.maximum(s, 1e-6)

    m_off = stats_dir / "data_mean.npy"
    s_off = stats_dir / "data_std.npy"
    if m_off.is_file() and s_off.is_file():
        mu69 = np.load(m_off).astype(np.float32)
        sd69 = np.load(s_off).astype(np.float32)
        if mu69.ndim != 1 or sd69.ndim != 1:
            raise ValueError(f"official fengwu stats shape invalid: mean={mu69.shape}, std={sd69.shape}")
        if int(mu69.shape[0]) != 69:
            raise ValueError(f"official fengwu stats channels must be 69, got {mu69.shape}")
        if c_in == 69:
            m = mu69[np.newaxis, :, np.newaxis, np.newaxis]
            s = sd69[np.newaxis, :, np.newaxis, np.newaxis]
        elif c_in == 138:
            m = np.concatenate([mu69, mu69], axis=0)[np.newaxis, :, np.newaxis, np.newaxis]
            s = np.concatenate([sd69, sd69], axis=0)[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            raise ValueError(f"official fengwu stats only support input C=69/138, got {c_in}")
        return (x_nchw.astype(np.float32) - m) / np.maximum(s, 1e-6)

    raise FileNotFoundError(f"FengWu stats: need global_means/global_stds or data_mean/data_std under {stats_dir}")


def fengwu_denorm_chw(y_chw: np.ndarray, stats_dir: Path) -> np.ndarray:
    """
    Fengwu 输出反归一化（可选）：
    - y_chw 为拆分后单帧通道 (69,H,W) 或 (189,H,W)
    """
    gm = stats_dir / "global_means.npy"
    gs = stats_dir / "global_stds.npy"
    if gm.is_file() and gs.is_file():
        mu = np.load(gm).astype(np.float32)
        sd = np.load(gs).astype(np.float32)
        c = int(y_chw.shape[0])
        c_stats = int(mu.shape[1]) if mu.ndim == 4 else -1
        if c_stats == c:
            m = mu[0, :c]
            s = sd[0, :c]
            return (y_chw.astype(np.float32) * s + m).astype(np.float32)

    m_off = stats_dir / "data_mean.npy"
    s_off = stats_dir / "data_std.npy"
    if m_off.is_file() and s_off.is_file():
        mu69 = np.load(m_off).astype(np.float32)
        sd69 = np.load(s_off).astype(np.float32)
        c = int(y_chw.shape[0])
        if c == 69:
            m = mu69[:, np.newaxis, np.newaxis]
            s = sd69[:, np.newaxis, np.newaxis]
            return (y_chw.astype(np.float32) * s + m).astype(np.float32)
        raise ValueError(f"official fengwu denorm expects 69 channels, got {c}")

    raise FileNotFoundError(f"FengWu denorm stats missing under {stats_dir}")


def unpack_fengwu_ort_outputs(outs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    多输出：surface(4,H,W), z,r,u,v,t 各 (Nz,H,W)。
    单输出：常见为 (1,189,H,W) 拼在通道维 — 4 + 37×5（z,r,u,v,t）。
    """
    def _squeeze_to_chw(a: np.ndarray) -> np.ndarray:
        x = np.asarray(a, dtype=np.float32)
        while x.ndim > 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Fengwu 输出期望规约到 3D (C,H,W)，得到 shape={x.shape} ndim={x.ndim}")
        # (H,W,C) 且 C 为变量维
        if x.shape[-1] in (189, 138, 69, 37, 13) and x.shape[0] >= 100 and x.shape[0] != x.shape[-1]:
            x = np.moveaxis(x, -1, 0)
        return x

    def _split_fengwu_69(o69: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            o69[:4],
            o69[4:17],
            o69[17:30],
            o69[30:43],
            o69[43:56],
            o69[56:69],
        )

    if len(outs) >= 6:
        surf = _squeeze_to_chw(outs[0])
        z = _squeeze_to_chw(outs[1])
        r = _squeeze_to_chw(outs[2])
        u = _squeeze_to_chw(outs[3])
        v = _squeeze_to_chw(outs[4])
        t_ = _squeeze_to_chw(outs[5])
        return surf, z, r, u, v, t_

    if len(outs) == 1:
        o = _squeeze_to_chw(outs[0])
        c = int(o.shape[0])
        if c == 189:
            return o[:4], o[4:41], o[41:78], o[78:115], o[115:152], o[152:189]
        if c == 138:
            # 138 常表示两帧 69；按 +6h 语义默认取前 69。若需 +12h 可在上层改为后 69。
            return _split_fengwu_69(o[:69])
        if c == 69:
            return _split_fengwu_69(o)
        raise ValueError(
            f"Fengwu 单输出通道数 {c}，支持 189 / 138（2×69）/ 69"
        )

    raise ValueError(f"Fengwu ONNX 输出个数 {len(outs)}：需要 1 个拼张量或 6 个分张量")


# Pangu 气压层顺序与 cepri_loader.PANGU_LEVELS / example.py 一致
_PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def _pangu_level_index(hpa: int) -> int:
    return _PANGU_LEVELS.index(hpa)


def save_pangu_pngs(plot_dir: Path, tag: str, p_out: np.ndarray, s_out: np.ndarray) -> None:
    """s_out: (1,4,H,W) 或 (4,H,W)；p_out: (1,5,13,H,W) 或 (5,13,H,W) — msl,u10,v10,t2m；z,q,t,u,v。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    if s_out.ndim == 4:
        s = s_out[0]
    else:
        s = s_out
    if p_out.ndim == 5:
        pr = p_out[0]
    else:
        pr = p_out
    i500 = _pangu_level_index(500)
    t2m = s[3]
    z500 = pr[0, i500]
    u10 = s[1]

    def one(fig_path: Path, arr: np.ndarray, title: str, cmap: str = "viridis") -> None:
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="upper")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    one(plot_dir / f"{tag}_t2m.png", t2m, f"{tag} 2m temperature (K)")
    one(plot_dir / f"{tag}_z500.png", z500, f"{tag} z @ 500 hPa (m^2/s^2)")
    one(plot_dir / f"{tag}_u10.png", u10, f"{tag} 10m u-wind (m/s)", cmap="RdBu_r")


def _session_options() -> ort.SessionOptions:
    o = ort.SessionOptions()
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    o.enable_mem_reuse = False
    o.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "1"))
    o.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_NUM_THREADS", "1"))
    # 抑制 constant_folding 等对 Clip 等节点的刷屏警告（不影响推理）
    # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    o.log_severity_level = int(os.environ.get("ORT_LOG_SEVERITY_LEVEL", "3"))
    return o


def pick_providers(device: str, allow_cpu_fallback: bool = False) -> List[Any]:
    """device: auto | dcu | cuda | cpu"""
    avail = set(ort.get_available_providers())
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if not allow_cpu_fallback:
            raise RuntimeError(
                "未检测到 CUDAExecutionProvider。"
                " 若确认需要回退 CPU，请显式添加 --allow-cpu-fallback。"
            )
        print("警告: 未检测到 CUDAExecutionProvider，按要求回退 CPU。", file=sys.stderr)
        return ["CPUExecutionProvider"]
    if device == "dcu":
        for p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider"):
            if p in avail:
                return [p, "CPUExecutionProvider"]
        if not allow_cpu_fallback:
            raise RuntimeError(
                "当前 onnxruntime 无 ROCM/MIGraphX EP；DCU 上请安装对应 onnxruntime 构建或使用 PyTorch 路径。"
                " 若确认需要回退 CPU，请显式添加 --allow-cpu-fallback。"
            )
        print(
            "警告: 当前 onnxruntime 无 ROCM/MIGraphX EP；DCU 上请安装对应 onnxruntime 构建或使用 PyTorch 路径。"
            " 按要求回退 CPU。",
            file=sys.stderr,
        )
        return ["CPUExecutionProvider"]
    # auto
    for p in (
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ):
        if p in avail:
            if p == "CPUExecutionProvider":
                return [p]
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(onnx_path: Path, providers: Sequence) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=_session_options(),
        providers=list(providers),
    )


def log_ort_session(tag: str, sess: ort.InferenceSession) -> None:
    """作业日志中确认实际使用的 ExecutionProvider（避免误用 CPU）。"""
    try:
        prov = sess.get_providers()
        print(f"[ORT] {tag} InferenceSession providers (order): {prov}", flush=True)
    except Exception as ex:
        print(f"[ORT] {tag} get_providers failed: {ex}", flush=True)


def _pangu_pressure_arr(p_in: np.ndarray) -> np.ndarray:
    """ONNX 期望 rank=4: (5, 13, H, W)，去掉任意前缀 batch 维 (1,…,1,5,13,H,W)。"""
    x = np.asarray(p_in, dtype=np.float32)
    while x.ndim > 4:
        if x.shape[0] != 1:
            raise ValueError(f"Pangu input 形状 {x.shape} 无法规约到 4 维 (5,13,H,W)")
        x = x[0]
    if x.ndim != 4:
        raise ValueError(f"Pangu input 期望 4 维，得到 {x.ndim} 维 shape={x.shape}")
    return x


def _pangu_surface_arr(s_in: np.ndarray) -> np.ndarray:
    """ONNX 期望 rank=3: (4, H, W)，去掉任意前缀 batch 维。"""
    x = np.asarray(s_in, dtype=np.float32)
    while x.ndim > 3:
        if x.shape[0] != 1:
            raise ValueError(f"Pangu input_surface 形状 {x.shape} 无法规约到 3 维 (4,H,W)")
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Pangu input_surface 期望 3 维，得到 {x.ndim} 维 shape={x.shape}")
    return x


def pangu_one_step(
    sess: ort.InferenceSession,
    p_in: np.ndarray,
    s_in: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pi = _pangu_pressure_arr(p_in)
    si = _pangu_surface_arr(s_in)
    outs = sess.run(None, {"input": pi, "input_surface": si})
    op, os_ = outs[0], outs[1]
    if op.ndim == 4:
        op = op[np.newaxis, ...]
    if os_.ndim == 3:
        os_ = os_[np.newaxis, ...]
    return op, os_


def run_pangu(
    era5_root: Path,
    date_yyyymmdd: str,
    hour: int,
    num_steps: int,
    providers: Sequence,
    out_dir: Path,
    save_pngs: bool,
) -> None:
    paths = {
        "1h": ZK_ROOT / "pangu" / "pangu_weather_1.onnx",
        "6h": ZK_ROOT / "pangu" / "pangu_weather_6.onnx",
        "24h": ZK_ROOT / "pangu" / "pangu_weather_24.onnx",
    }
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(p)
    has_3h = (ZK_ROOT / "pangu" / "pangu_weather_3.onnx").is_file()
    if has_3h:
        paths["3h"] = ZK_ROOT / "pangu" / "pangu_weather_3.onnx"

    sessions = {k: create_session(p, providers) for k, p in paths.items()}
    blob = load_cepri_time(era5_root, date_yyyymmdd, hour)
    p_in, s_in = pack_pangu_onnx(blob)
    plot_dir = out_dir / "plots"
    if save_pngs:
        save_pangu_pngs(plot_dir, "input_t0", p_in, s_in)

    cur_p, cur_s = p_in.copy(), s_in.copy()
    c1p, c1s = cur_p.copy(), cur_s.copy()
    c3p, c3s = cur_p.copy(), cur_s.copy()
    c6p, c6s = cur_p.copy(), cur_s.copy()
    c24p, c24s = cur_p.copy(), cur_s.copy()

    preds: List[Tuple[np.ndarray, np.ndarray]] = []
    for step in range(1, num_steps + 1):
        use_24h = step % 24 == 0
        use_6h = (not use_24h) and (step % 6 == 0)
        use_3h = (not use_24h) and (not use_6h) and (step % 3 == 0)
        if use_24h:
            op, os_ = pangu_one_step(sessions["24h"], c24p, c24s)
        elif use_6h:
            op, os_ = pangu_one_step(sessions["6h"], c6p, c6s)
        elif use_3h:
            if has_3h:
                op, os_ = pangu_one_step(sessions["3h"], c3p, c3s)
            else:
                op, os_ = pangu_one_step(sessions["1h"], c1p, c1s)
        else:
            op, os_ = pangu_one_step(sessions["1h"], c1p, c1s)

        preds.append((op, os_))
        if save_pngs:
            save_pangu_pngs(plot_dir, f"step_{step:02d}h", op, os_)
        cur_p, cur_s = op, os_
        c1p, c1s = cur_p.copy(), cur_s.copy()
        c3p, c3s = cur_p.copy(), cur_s.copy()
        c6p, c6s = cur_p.copy(), cur_s.copy()
        c24p, c24s = cur_p.copy(), cur_s.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (op, os_) in enumerate(preds):
        np.save(out_dir / f"pangu_step{i+1:03d}_pressure.npy", op)
        np.save(out_dir / f"pangu_step{i+1:03d}_surface.npy", os_)
    print(f"Pangu: wrote {len(preds)} steps to {out_dir}")
    if save_pngs:
        print(f"Pangu: PNGs -> {plot_dir}")


def _fengwu_expected_combo_channels(sess: ort.InferenceSession) -> Optional[int]:
    """单张量输入时，通道维上的期望长度（固定导出的 ONNX 为 int；动态则为 None）。"""
    shape = sess.get_inputs()[0].shape
    if len(shape) < 2:
        return None
    c = shape[1]
    return c if isinstance(c, int) and c > 0 else None


def build_fengwu_onnx_combo_input(
    era5_root: Path,
    date_yyyymmdd: str,
    hour: int,
    sess: ort.InferenceSession,
) -> np.ndarray:
    """
    189：单帧 4+37×5（与 onescience Fengwu pressure_level=37 一致），取 `--hour` 对应整点分析场。
    138：两帧各 69=4+13×5，较早时刻在前 [h, h+6]（裁剪到同日 0–23 时）。
    """
    exp = _fengwu_expected_combo_channels(sess)
    if exp is None or exp == 189:
        fields = load_cepri_fengwu_fields(era5_root, date_yyyymmdd, int(hour))
        surf = fields["surface"][np.newaxis, ...]
        return np.concatenate(
            [
                surf,
                fields["z"][np.newaxis, ...],
                fields["r"][np.newaxis, ...],
                fields["u"][np.newaxis, ...],
                fields["v"][np.newaxis, ...],
                fields["t"][np.newaxis, ...],
            ],
            axis=1,
        ).astype(np.float32)
    if exp == 138:
        h0 = int(max(0, min(23, hour)))
        h_prev, h_curr = h0, int(min(23, h0 + 6))
        b_prev = load_cepri_time(era5_root, date_yyyymmdd, h_prev)
        b_curr = load_cepri_time(era5_root, date_yyyymmdd, h_curr)
        a = _fengwu_69_from_blob_q_order(b_prev)
        b = _fengwu_69_from_blob_q_order(b_curr)
        return np.concatenate([a, b], axis=0)[np.newaxis, ...].astype(np.float32)
    raise ValueError(
        f"Fengwu 单输入 ONNX 期望通道维 {exp}；已实现 189（37 层单帧）与 138（13 层×两帧）。"
    )


def fuxi_temb(lead_hours: int) -> np.ndarray:
    """(1, 12) 简易时间嵌入；若与训练不一致可换用 Earth2Studio / 训练端公式。"""
    h = np.float32(lead_hours)
    xs = []
    for k in range(6):
        w = np.float32(1.0 + k)
        ang = h / 24.0 * w * np.float32(np.pi)
        xs.extend([np.sin(ang), np.cos(ang)])
    return np.array(xs, dtype=np.float32)[np.newaxis, :]


def run_fengwu(
    era5_root: Path,
    date_yyyymmdd: str,
    hour: int,
    providers: Sequence,
    out_dir: Path,
    num_steps: int,
    model_version: str,
    stats_dir: Optional[Path],
) -> None:
    onnx_p = ZK_ROOT / "fengwu" / f"fengwu_{model_version}.onnx"
    if not onnx_p.is_file():
        raise FileNotFoundError(onnx_p)
    sess = create_session(onnx_p, providers)
    inps = sess.get_inputs()
    if len(inps) != 1:
        raise RuntimeError("Fengwu CLI 路径期望单张量输入 ONNX")

    x = build_fengwu_onnx_combo_input(era5_root, date_yyyymmdd, hour, sess)
    c_in = int(x.shape[1])
    if stats_dir is not None:
        x = fengwu_normalize_for_onnx(x, stats_dir)
    else:
        print("警告: Fengwu 未提供 stats-dir，输入未归一化。", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    cur = x.astype(np.float32)
    for i in range(num_steps):
        y = sess.run(None, {inps[0].name: cur})[0]
        np.save(out_dir / f"fengwu_step{i+1:03d}_raw.npy", y)
        yo = np.asarray(y, dtype=np.float32)
        while yo.ndim > 4 and yo.shape[0] == 1:
            yo = yo[0]
        if yo.ndim == 4 and yo.shape[0] == 1:
            yo = yo[0]
        if yo.ndim != 3:
            raise RuntimeError(f"Fengwu unexpected output rank: {y.shape}")
        if yo.shape[0] >= 69:
            pred69_norm = yo[:69]
        else:
            raise RuntimeError(f"Fengwu output channels < 69: {yo.shape}")
        if stats_dir is not None:
            pred69_denorm = fengwu_denorm_chw(pred69_norm, stats_dir)
        else:
            pred69_denorm = pred69_norm
        np.save(out_dir / f"fengwu_step{i+1:03d}_pred69_denorm.npy", pred69_denorm)

        if c_in == 138 and stats_dir is not None:
            pred69_norm2 = fengwu_normalize_for_onnx(pred69_norm[np.newaxis, ...], stats_dir)[0]
            cur = np.concatenate([cur[:, 69:], pred69_norm2[np.newaxis, ...]], axis=1).astype(np.float32)
        else:
            break
    print(f"Fengwu({model_version}): wrote {num_steps} step(s) to {out_dir}")


def run_fuxi(
    model_key: str,
    era5_root: Path,
    date_yyyymmdd: str,
    hour0: int,
    hour1: int,
    num_steps: int,
    providers: Sequence,
    stats_dir: Optional[Path],
    out_dir: Path,
    lead_hours_temb: int = 6,
) -> None:
    rel = "short.onnx" if model_key == "fuxi_short" else "medium.onnx"
    onnx_p = ZK_ROOT / "fuxi" / rel
    if not onnx_p.is_file():
        raise FileNotFoundError(onnx_p)
    sess = create_session(onnx_p, providers)
    raw = load_cepri_fuxi_fields(era5_root, date_yyyymmdd, hour0, hour1)
    x, layout = fuxi_prepare_onnx_input(raw, sess)
    print(f"Fuxi input layout: {layout}, shape={x.shape}")
    if stats_dir is not None:
        x = fuxi_normalize_for_layout(x, stats_dir, layout)
    else:
        print("警告: 未提供 --stats-dir，Fuxi 输入未做训练同款归一化，数值仅作联调。", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    cur = x.astype(np.float32)
    inps = sess.get_inputs()
    for step in range(num_steps):
        feeds = {}
        temb_step = fuxi_temb(int(lead_hours_temb) * (step + 1))
        for inp in inps:
            n = inp.name.lower()
            if "temb" in n or inp.name == "temb":
                feeds[inp.name] = temb_step
            else:
                feeds[inp.name] = cur
        new_input = sess.run(None, feeds)[0]
        np.save(out_dir / f"fuxi_step{step+1:03d}_new_input.npy", new_input)

        # Follow official FuXi: output is the latest time slot from returned window.
        y = np.asarray(new_input, dtype=np.float32)
        if y.ndim == 5 and layout == "NTCHW":
            out_latest = y[:, -1]
        elif y.ndim == 5 and layout == "NCTHW":
            out_latest = y[:, :, -1]
        else:
            out_latest = y
        np.save(out_dir / f"fuxi_step{step+1:03d}_latest.npy", out_latest)
        cur = new_input.astype(np.float32)
    print(f"Fuxi({model_key}): wrote {num_steps} rollout steps -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="ZK_Models ONNX + CEPRI ERA5 推理")
    p.add_argument(
        "--model",
        required=True,
        choices=["pangu", "fengwu", "fuxi_short", "fuxi_medium"],
    )
    p.add_argument("--era5-root", default="/public/share/aciwgvx1jd/CEPRI_ERA5")
    p.add_argument("--date", default="20200101", help="YYYYMMDD")
    p.add_argument("--hour", type=int, default=0)
    p.add_argument("--hour1", type=int, default=None, help="Fuxi 第二时刻，默认 hour+1（与旧版脚本一致）")
    p.add_argument("--num-steps", type=int, default=2, help="Pangu/Fengwu/Fuxi: 迭代步数")
    p.add_argument("--lead-hours", type=int, default=6, help="保留参数：旧版 Fuxi temb 路径用")
    p.add_argument("--device", default="auto", choices=["auto", "dcu", "cuda", "cpu"])
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="当目标设备 EP 缺失时允许回退 CPU（默认关闭；DCU/CUDA 场景默认 fail-fast）",
    )
    p.add_argument(
        "--stats-dir",
        default=None,
        help="Fuxi 可选：含 global_means.npy / global_stds.npy（与 ERA5HDF5Datapipe 一致）",
    )
    p.add_argument(
        "--fengwu-model-version",
        default="v2",
        choices=["v1", "v2"],
        help="Fengwu 模型版本（旧版工作流默认 v2，可与官方 ERA5=v1 对照）",
    )
    p.add_argument(
        "--fengwu-stats-dir",
        default=None,
        help="Fengwu 官方 stats 目录（含 data_mean.npy/data_std.npy）。不填时默认 ZK_Models/fengwu",
    )
    p.add_argument(
        "--save-pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pangu：保存 input 与各步 t2m / z500 / u10 到 output-dir/plots/",
    )
    args = p.parse_args()

    era5_root = Path(args.era5_root)
    providers = pick_providers(args.device, allow_cpu_fallback=bool(args.allow_cpu_fallback))
    print("onnxruntime providers:", ort.get_available_providers())
    print("using:", providers)

    out = Path(args.output_dir) if args.output_dir else ZK_ROOT / "results" / args.model

    if args.model == "pangu":
        run_pangu(
            era5_root,
            args.date,
            args.hour,
            args.num_steps,
            providers,
            out,
            save_pngs=args.save_pngs,
        )
    elif args.model == "fengwu":
        fw_stats = Path(args.fengwu_stats_dir) if args.fengwu_stats_dir else (ZK_ROOT / "fengwu")
        run_fengwu(
            era5_root,
            args.date,
            args.hour,
            providers,
            out,
            num_steps=args.num_steps,
            model_version=args.fengwu_model_version,
            stats_dir=fw_stats,
        )
    else:
        h1 = args.hour1 if args.hour1 is not None else min(args.hour + 1, 23)
        stats = Path(args.stats_dir) if args.stats_dir else None
        run_fuxi(
            args.model,
            era5_root,
            args.date,
            args.hour,
            h1,
            args.num_steps,
            providers,
            stats,
            out,
            lead_hours_temb=int(args.lead_hours),
        )


if __name__ == "__main__":
    main()
