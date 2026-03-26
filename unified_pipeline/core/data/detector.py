"""
数据格式自动探测器。

检查规则（优先级从高到低）：
1. 若 {root}/pressure/pressure/ 子目录存在且含 *_pressure.nc → gundong_20260324
2. 若 {root}/ 直接含 *_pressure.nc 或 {root}/YYYY_MM/*_pressure.nc → era5_flat
3. 否则抛出 ValueError
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base_adapter import DataAdapter
from .era5_adapter import ERA5FlatAdapter
from .gundong_adapter import GunDongAdapter

_FORMAT_MAP = {
    "era5_flat": ERA5FlatAdapter,
    "gundong_20260324": GunDongAdapter,
}


def detect_format(root: Path) -> str:
    """返回格式名称字符串。"""
    root = Path(root)
    # GunDong 特征：pressure/pressure/ 子目录
    pdir = root / "pressure" / "pressure"
    if pdir.is_dir() and any(pdir.glob("*_pressure.nc")):
        return "gundong_20260324"

    # ERA5 flat：根目录或月份子目录含 *_pressure.nc
    if any(root.glob("*_pressure.nc")):
        return "era5_flat"
    if any(root.glob("????_??/*_pressure.nc")):
        return "era5_flat"

    raise ValueError(
        f"无法自动识别 {root} 的数据格式。\n"
        "请在 config/data.yaml 中明确设置 format 字段，\n"
        "或通过 --data-format 参数指定（era5_flat | gundong_20260324）。"
    )


def get_adapter(
    root: Path,
    fmt: Optional[str] = None,
    **kwargs,
) -> DataAdapter:
    """
    根据格式名称（或自动探测）返回对应的 DataAdapter 实例。

    Args:
        root:   数据根目录
        fmt:    格式名称（None 则自动探测）
        kwargs: 传递给 Adapter 构造函数的额外参数（如 use_monthly_subdir）
    """
    if fmt is None:
        fmt = detect_format(Path(root))
    fmt = fmt.lower().strip()
    if fmt not in _FORMAT_MAP:
        raise ValueError(
            f"未知数据格式 '{fmt}'，支持的格式：{list(_FORMAT_MAP.keys())}"
        )
    return _FORMAT_MAP[fmt](Path(root), **kwargs)


def register_format(name: str, adapter_cls: type) -> None:
    """注册自定义格式，允许第三方扩展。"""
    _FORMAT_MAP[name] = adapter_cls
