"""
硬件资源监控模块。

在后台守护线程中周期性轮询 GPU/CPU 利用率和显存，写入日志文件。

支持：
  - rocm-smi（DCU / AMD GPU，优先）
  - nvidia-smi（CUDA GPU）
  - 仅 CPU（无 GPU 环境）

用法：
    from core.monitoring import start_hardware_logger

    with start_hardware_logger(log_dir="logs", poll_interval=30) as logger:
        run_inference(...)  # 推理在此运行
    # 退出时自动打印摘要

或手动控制：
    logger = HardwareLogger(log_dir="logs", poll_interval=30)
    logger.start()
    ...
    summary = logger.stop()
    print(summary)
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _detect_backend() -> str:
    """检测可用的 GPU 监控工具。"""
    for cmd in ["rocm-smi", "nvidia-smi"]:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "cpu-only"


def _query_rocm_smi() -> List[Dict]:
    """解析 rocm-smi 输出，返回每卡信息列表。

    rocm-smi CSV 列名因版本而异，常见形式：
      time, device, DCU use (%), vram Total Memory (B), vram Total Used Memory (B)
    或旧版：
      device, GPU%, VRAM Total Memory (B), VRAM In Use (B)
    统一映射到内部字段：gpu_id, gpu_util, vram_used_mb, vram_total_mb。
    """
    # CSV 列名到内部字段的模糊映射规则（顺序：先匹配者优先）
    _COL_RULES = [
        # (内部字段,  匹配关键字列表,  转换函数)
        ("gpu_id",       ["device", "card"],                            lambda v: v),
        ("gpu_util",     ["dcu use", "gpu use", "gpu%", "use (%)"],     lambda v: float(v or 0)),
        ("vram_used_mb", ["used memory", "in use", "used"],             lambda v: float(v or 0) / 1024**2),
        ("vram_total_mb",["total memory", "vram total", "total mem"],   lambda v: float(v or 0) / 1024**2),
    ]

    def _map_header(raw_headers: List[str]) -> Dict[int, str]:
        """将 CSV 原始列名映射到内部字段，返回 {列索引: 内部字段名}。"""
        mapping: Dict[int, str] = {}
        matched_fields: set = set()
        for col_i, raw in enumerate(raw_headers):
            rl = raw.lower()
            for field, keywords, _ in _COL_RULES:
                if field in matched_fields:
                    continue
                if any(kw in rl for kw in keywords):
                    mapping[col_i] = field
                    matched_fields.add(field)
                    break
        return mapping

    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return _parse_rocm_text_fallback()

        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if not lines:
            return _parse_rocm_text_fallback()

        # 找到 CSV 标题行（含逗号，且含时间或 device 字样）
        header_idx = -1
        header_parts: List[str] = []
        for i, line in enumerate(lines):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                low = line.lower()
                if "device" in low or "card" in low or "dcu" in low or "gpu" in low:
                    header_idx = i
                    header_parts = parts
                    break

        if header_idx < 0:
            return _parse_rocm_text_fallback()

        col_map = _map_header(header_parts)
        field_conv = {field: fn for field, _kws, fn in _COL_RULES}

        gpu_records: List[Dict] = []
        for line in lines[header_idx + 1:]:
            if not line or not "," in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            rec: Dict = {}
            for col_i, field in col_map.items():
                if col_i < len(parts):
                    try:
                        rec[field] = field_conv[field](parts[col_i])
                    except (ValueError, ZeroDivisionError):
                        rec[field] = 0.0
            if rec:
                gpu_records.append(rec)

        return gpu_records if gpu_records else _parse_rocm_text_fallback()

    except Exception:
        return _parse_rocm_text_fallback()


def _parse_rocm_text_fallback() -> List[Dict]:
    """fallback：用非 CSV 文本模式查询 rocm-smi。"""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10
        )
        return _parse_rocm_text(result.stdout)
    except Exception:
        return []


def _parse_rocm_text(text: str) -> List[Dict]:
    """解析 rocm-smi 非 CSV 文本输出。"""
    records = []
    current: Dict = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("GPU["):
            if current:
                records.append(current)
            gpu_id = line.split("[")[1].split("]")[0]
            current = {"gpu_id": gpu_id}
        elif any(k in line for k in ("GPU use (%)", "GPU%", "DCU use")):
            try:
                current["gpu_util"] = float(line.split(":")[-1].strip().replace("%", ""))
            except ValueError:
                pass
        elif "Total Memory" in line:
            try:
                val_b = float(line.split(":")[-1].strip().split()[0])
                if "Used" in line or "In Use" in line:
                    current["vram_used_mb"] = val_b / 1024**2
                else:
                    current["vram_total_mb"] = val_b / 1024**2
            except ValueError:
                pass
    if current:
        records.append(current)
    return records


def _query_nvidia_smi() -> List[Dict]:
    """解析 nvidia-smi 输出。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        records = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                records.append({
                    "gpu_id": parts[0],
                    "gpu_util": float(parts[1]),
                    "vram_used_mb": float(parts[2]),
                    "vram_total_mb": float(parts[3]),
                })
        return records
    except Exception:
        return []


def _query_cpu() -> Dict:
    """查询 CPU 使用率（使用 /proc/stat）。"""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = [int(x) for x in line.split()[1:]]
        idle = vals[3]
        total = sum(vals)
        return {"cpu_total": total, "cpu_idle": idle}
    except Exception:
        return {}


class HardwareLogger:
    """
    后台硬件资源监控器。
    启动守护线程，每 poll_interval 秒采集一次，写入日志文件。
    stop() 时输出峰值/均值摘要。
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        poll_interval: int = 30,
        backend: str = "auto",
        job_id: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval

        if backend == "auto":
            self._backend = _detect_backend()
        else:
            self._backend = backend

        jid = job_id or os.environ.get("SLURM_JOB_ID", "local")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.log_dir / f"hardware_{jid}_{ts}.log"

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_utils: List[float] = []
        self._vram_used: List[float] = []
        self._prev_cpu: Optional[Dict] = None
        self._cpu_usages: List[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(
            f"[HardwareLogger] 监控启动 | 后端={self._backend} | 间隔={self.poll_interval}s | 日志={self._log_path}",
            flush=True,
        )

    def stop(self) -> str:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 5)
        summary = self._make_summary()
        print(f"[HardwareLogger] 监控停止\n{summary}", flush=True)
        return summary

    def _run(self) -> None:
        with open(self._log_path, "w", buffering=1) as f:
            f.write(f"# hardware_logger | backend={self._backend} | start={datetime.now()}\n")
            f.write("timestamp,gpu_id,gpu_util_pct,vram_used_mb,vram_total_mb,cpu_util_pct\n")
            while not self._stop_event.is_set():
                self._collect(f)
                self._stop_event.wait(self.poll_interval)

    def _collect(self, f) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gpu_records = []

        if self._backend == "rocm-smi":
            gpu_records = _query_rocm_smi()
        elif self._backend == "nvidia-smi":
            gpu_records = _query_nvidia_smi()

        # CPU
        cpu_info = _query_cpu()
        cpu_pct = 0.0
        if self._prev_cpu and cpu_info:
            d_total = cpu_info["cpu_total"] - self._prev_cpu["cpu_total"]
            d_idle = cpu_info["cpu_idle"] - self._prev_cpu["cpu_idle"]
            if d_total > 0:
                cpu_pct = round(100.0 * (1.0 - d_idle / d_total), 1)
        self._prev_cpu = cpu_info
        self._cpu_usages.append(cpu_pct)

        if gpu_records:
            for rec in gpu_records:
                gid = rec.get("gpu_id", "?")
                gu = float(rec.get("gpu_util", 0) or 0)
                vu = float(rec.get("vram_used_mb", 0) or 0)
                vt = float(rec.get("vram_total_mb", 0) or 0)
                f.write(f"{ts},{gid},{gu:.1f},{vu:.1f},{vt:.1f},{cpu_pct:.1f}\n")
                self._gpu_utils.append(gu)
                self._vram_used.append(vu)
        else:
            f.write(f"{ts},none,0.0,0.0,0.0,{cpu_pct:.1f}\n")

    def _make_summary(self) -> str:
        lines = ["[HardwareLogger] ===== 资源使用摘要 ====="]
        if self._gpu_utils:
            lines.append(
                f"  GPU 利用率: 均值={np.mean(self._gpu_utils):.1f}%  "
                f"峰值={np.max(self._gpu_utils):.1f}%"
            )
        if self._vram_used:
            lines.append(
                f"  显存占用:   均值={np.mean(self._vram_used):.0f}MB  "
                f"峰值={np.max(self._vram_used):.0f}MB"
            )
        if self._cpu_usages:
            lines.append(
                f"  CPU 利用率: 均值={np.mean(self._cpu_usages):.1f}%  "
                f"峰值={np.max(self._cpu_usages):.1f}%"
            )
        lines.append(f"  日志文件: {self._log_path}")
        return "\n".join(lines)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


@contextmanager
def start_hardware_logger(
    log_dir: str | Path = "logs",
    poll_interval: int = 30,
    backend: str = "auto",
    enabled: bool = True,
):
    """
    上下文管理器，方便在推理代码中嵌入监控。

    with start_hardware_logger(log_dir="logs", poll_interval=30) as logger:
        run_inference(...)
    """
    if not enabled:
        yield None
        return
    logger = HardwareLogger(log_dir=log_dir, poll_interval=poll_interval, backend=backend)
    logger.start()
    try:
        yield logger
    finally:
        logger.stop()
