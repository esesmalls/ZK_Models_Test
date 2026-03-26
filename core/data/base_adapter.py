"""
抽象数据适配器接口。

所有 DataAdapter 实现的 load_blob() 必须返回同一结构的 blob dict：
    {
        "surface_msl":  np.ndarray (H, W) float32,
        "surface_u10":  np.ndarray (H, W) float32,
        "surface_v10":  np.ndarray (H, W) float32,
        "surface_t2m":  np.ndarray (H, W) float32,
        "pangu_z":      np.ndarray (13, H, W) float32,  # on PANGU_LEVELS
        "pangu_q":      np.ndarray (13, H, W) float32,
        "pangu_t":      np.ndarray (13, H, W) float32,
        "pangu_u":      np.ndarray (13, H, W) float32,
        "pangu_v":      np.ndarray (13, H, W) float32,
        "pressure_src": np.ndarray (N,) float32,        # original levels
        "lat":          np.ndarray (H,) float32,        # N→S, 90…-90
        "lon":          np.ndarray (W,) float32,        # 0…359.75
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class DataAdapter(ABC):
    """统一数据读取接口，屏蔽不同 ERA5 文件布局的差异。"""

    FORMAT_NAME: str = "base"

    def __init__(self, root: Path, **kwargs):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # 必须实现
    # ------------------------------------------------------------------

    @abstractmethod
    def load_blob(self, date_yyyymmdd: str, hour: int) -> Dict[str, np.ndarray]:
        """
        读取指定日期+小时的分析场，返回统一 blob dict。
        date_yyyymmdd: '20260308'
        hour: 0..23 (UTC)
        """

    @abstractmethod
    def list_dates(self) -> List[str]:
        """返回数据目录中所有可用日期（yyyymmdd 字符串列表，升序）。"""

    # ------------------------------------------------------------------
    # 可选覆盖
    # ------------------------------------------------------------------

    def load_blob_safe(
        self, date_yyyymmdd: str, hour: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """读取失败时返回 None，而非抛异常（用于真值加载）。"""
        try:
            return self.load_blob(date_yyyymmdd, hour)
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return None

    def load_blob_for_valid_time(
        self, valid_dt: datetime
    ) -> Optional[Dict[str, np.ndarray]]:
        """给定 datetime 对象加载真值，失败返回 None。"""
        return self.load_blob_safe(valid_dt.strftime("%Y%m%d"), int(valid_dt.hour))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root})"
