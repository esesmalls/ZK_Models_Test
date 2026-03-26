#!/usr/bin/env python3
"""基础核心模块测试脚本。"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent.parent)

print("Step 1: importing data modules...")
from core.data.base_adapter import DataAdapter
from core.data.detector import detect_format, get_adapter
from core.data.channel_mapper import blob_to_pangu_onnx, blob_to_fengwu_69ch
print("  OK")

print("Step 2: importing model base...")
from core.models.base_model import WeatherModel, ModelState
from core.models.model_registry import ModelRegistry
print("  OK")

print("Step 3: importing evaluation metrics...")
from core.evaluation.metrics import MetricsAccumulator, compute_step_metrics
print("  OK")

print("Step 4: importing monitoring...")
from core.monitoring.hardware_logger import HardwareLogger
print("  OK")

print("Step 5: importing zk_io...")
from zk_io.npy_writer import NpyStackWriter
from zk_io.nc_writer import write_step_nc
from zk_io.plot_utils import plot_compare
print("  OK")

print("Step 6: format detection...")
fmt = detect_format(Path('/public/share/aciwgvx1jd/20260324'))
print(f"  gundong 格式: {fmt}")

print("Step 7: metrics computation...")
import numpy as np
lats = np.linspace(90, -90, 721, dtype=np.float32)
acc = MetricsAccumulator(lats)
pred = np.ones((721, 1440), dtype=np.float32)
truth = np.zeros((721, 1440), dtype=np.float32)
diff, mvals = acc.add('FengWu', 'u10', 6, pred, truth)
print(f"  指标: {mvals}")

print("Step 8: test GunDong adapter blob loading...")
adapter = get_adapter(Path('/public/share/aciwgvx1jd/20260324'), fmt='gundong_20260324')
dates = adapter.list_dates()
print(f"  可用日期 {len(dates)} 天: {dates[:3]}...")
if dates:
    blob = adapter.load_blob(dates[0], 12)
    print(f"  blob keys: {list(blob.keys())}")
    print(f"  surface_t2m shape: {blob['surface_t2m'].shape}")
    print(f"  pangu_z shape: {blob['pangu_z'].shape}")

print("\n所有测试通过！")
