"""
NPY 输出工具。

NpyStackWriter 使用 memmap 方式，边推理边写，避免全部积累在内存。
输出格式与现有 GunDong_Infer_result_12h 兼容：
  {output_root}/{ModelName}/ERA5_6H/{var}_{TAG}.npy
  shape: (n_steps, H, W), dtype=float32
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class NpyStackWriter:
    """
    预分配 memmap 文件，逐步写入预报结果（节省内存）。

    Args:
        output_root:  结果根目录
        model_name:   模型目录名（如 "FengWu"）
        init_tag:     时间标签（如 "20260308T12"）
        variables:    地表变量名列表（如 ["u10","v10","t2m","msl"]）
        n_steps:      预报总步数
        shape_hw:     (H, W)，通常 (721, 1440)
        pangu_suffix: PanGu 使用 {var}_surface_{tag}.npy 命名，设为 True
    """

    def __init__(
        self,
        output_root: Path,
        model_name: str,
        init_tag: str,
        variables: List[str],
        n_steps: int,
        shape_hw: Tuple[int, int] = (721, 1440),
        pangu_suffix: bool = False,
    ):
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.init_tag = init_tag
        self.variables = variables
        self.n_steps = n_steps
        self.shape_hw = shape_hw
        self._pangu_suffix = pangu_suffix
        self._step_idx = 0

        base = self.output_root / model_name / "ERA5_6H"
        base.mkdir(parents=True, exist_ok=True)
        self._memmaps: Dict[str, np.memmap] = {}
        for var in variables:
            fname = (
                f"{var}_surface_{init_tag}.npy"
                if pangu_suffix
                else f"{var}_{init_tag}.npy"
            )
            path = base / fname
            mm = np.memmap(
                str(path),
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, *shape_hw),
            )
            self._memmaps[var] = mm

    def write_step(self, step_idx: int, preds: Dict[str, np.ndarray]) -> None:
        """写入第 step_idx 步的预报值（0-based）。"""
        for var, mm in self._memmaps.items():
            if var in preds:
                arr = np.asarray(preds[var], dtype=np.float32)
                if arr.shape != self.shape_hw:
                    arr = arr[:self.shape_hw[0], :self.shape_hw[1]]
                mm[step_idx] = arr

    def flush(self) -> None:
        """强制刷写所有 memmap。"""
        for mm in self._memmaps.values():
            mm.flush()

    def close(self) -> None:
        self.flush()
        for mm in self._memmaps.values():
            del mm
        self._memmaps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_paths(self) -> Dict[str, Path]:
        base = self.output_root / self.model_name / "ERA5_6H"
        result = {}
        for var in self.variables:
            fname = (
                f"{var}_surface_{self.init_tag}.npy"
                if self._pangu_suffix
                else f"{var}_{self.init_tag}.npy"
            )
            result[var] = base / fname
        return result
