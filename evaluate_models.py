import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# ===================== 1. 配置路径与参数 =====================
ERA5_DIR = "/public/share/aciwgvx1jd/20260324/surface/"
PRED_BASE_DIR = "/public/share/aciwgvx1jd/GunDong_Infer_result_12h"

MODELS = ["FengWu", "GraphCast", "FuXi", "PanGu"]
VARIABLES = ["u10", "v10", "t2m"]

TIME_TAG = "20260308T12"
INITIAL_TIME = "2026-03-08T12:00:00"  # 起报时间
STEP_INTERVAL = 6                     # 预报间隔 6 小时
EXPECTED_STEPS = 40                   # 240小时 / 6小时 = 40步

SAVE_DIR = os.path.join(PRED_BASE_DIR, f"eval_240h_{TIME_TAG}")

# ===================== 2. 准备工作：提取全局网格与权重 =====================
os.makedirs(SAVE_DIR, exist_ok=True)

# 找一天的数据来获取经纬度网格和权重
sample_pattern = os.path.join(ERA5_DIR, "2026_03_08_surface_*.nc")
sample_files = glob.glob(sample_pattern)
if not sample_files:
    raise FileNotFoundError(f"🚨 找不到基准网格文件，请检查 ERA5_DIR: {sample_pattern}")

print("🔄 正在初始化网格和纬度权重...")
with xr.open_dataset(sample_files[0]) as ds_sample:
    lats = ds_sample.latitude.values
    lons = ds_sample.longitude.values
    time_dim = 'valid_time' if 'valid_time' in ds_sample.dims else 'time'

# 预先生成 40 个有效时间 (Valid Times)
valid_times = [pd.to_datetime(INITIAL_TIME) + pd.Timedelta(hours=(i + 1) * STEP_INTERVAL) for i in range(EXPECTED_STEPS)]

lat_rad = np.deg2rad(lats)
weights = np.cos(lat_rad)
weights_2d = np.broadcast_to(weights[:, np.newaxis], (len(lats), len(lons)))
sum_weights = np.nansum(weights_2d)

metrics_records = []

# ===================== 3. 高效加载数据、算偏差、双格式保存 =====================
# 建立一个字典来缓存当天的所有 Dataset
ds_cache = {}
current_era5_date = None

for model in MODELS:
    print(f"\n{'='*45}")
    print(f"🚀 开始处理并评估模型: {model}")
    print(f"{'='*45}")
    
    for var in VARIABLES:
        npy_file = f"{var}_surface_{TIME_TAG}.npy" if model == "PanGu" else f"{var}_{TIME_TAG}.npy"
        npy_path = os.path.join(PRED_BASE_DIR, model, "ERA5_6H", npy_file)
        
        if not os.path.exists(npy_path):
            print(f"  ❌ 找不到预报文件: {npy_file}")
            continue
            
        pred_data = np.load(npy_path).squeeze()
        if pred_data.ndim > 3: pred_data = pred_data[0]
        if pred_data.ndim == 2: pred_data = pred_data[np.newaxis, :, :]
            
        diff_data_array = np.zeros_like(pred_data)
        
        for step_idx in range(EXPECTED_STEPS):
            valid_time = valid_times[step_idx]
            lead_time_hours = (step_idx + 1) * STEP_INTERVAL
            target_date_str = valid_time.strftime("%Y_%m_%d")
            
            # ✅ 核心修复：安全可靠的缓存机制，缓存当天的【所有】文件
            if current_era5_date != target_date_str:
                # 跨日时，先安全关闭并清空上一天的缓存
                for ds in ds_cache.values():
                    ds.close()
                ds_cache.clear()
                
                # 读取当天的所有文件 (accum 和 instant 都抓进来)
                era5_files_for_day = glob.glob(os.path.join(ERA5_DIR, f"{target_date_str}_surface_*.nc"))
                for f in era5_files_for_day:
                    ds_cache[f] = xr.open_dataset(f)
                    
                current_era5_date = target_date_str
            
            # 在当天的所有缓存文件中寻找包含目标变量的那一个
            true_step_data = None
            for ds_day in ds_cache.values():
                if valid_time in ds_day[time_dim].values:
                    era5_var_name = var if var in ds_day else var.replace("u10", "10u").replace("v10", "10v")
                    if era5_var_name in ds_day:
                        # 找到了！提取数据并跳出寻找
                        true_step_data = ds_day[era5_var_name].sel({time_dim: valid_time}).values
                        break 
            
            if true_step_data is not None:
                # 算偏差
                diff_step = pred_data[step_idx, :, :] - true_step_data
                diff_data_array[step_idx, :, :] = diff_step
                
                # 算指标
                w_mae = np.nansum(np.abs(diff_step) * weights_2d) / sum_weights
                w_rmse = np.sqrt(np.nansum((diff_step ** 2) * weights_2d) / sum_weights)
                
                metrics_records.append({
                    "Model": model, "Variable": var, "Lead_Time": lead_time_hours,
                    "W-MAE": w_mae, "W-RMSE": w_rmse
                })
            else:
                print(f"    ⚠️ 缺失 ERA5 数据: {valid_time}，填 NaN。")
                diff_data_array[step_idx, :, :] = np.nan 

        # 循环结束，保存双格式
        base_savename = f"{model}_{var}_{TIME_TAG}_diff"
        npy_save_path = os.path.join(SAVE_DIR, f"{base_savename}.npy")
        np.save(npy_save_path, diff_data_array)
        
        diff_da = xr.DataArray(
            diff_data_array,
            coords={'time': valid_times, 'latitude': lats, 'longitude': lons},
            dims=['time', 'latitude', 'longitude'],
            name=f"{var}_diff"
        )
        nc_save_path = os.path.join(SAVE_DIR, f"{base_savename}.nc")
        diff_da.to_netcdf(nc_save_path)
        
        print(f"  💾 偏差场双格式已保存: {base_savename} (.npy / .nc)")

# 最后清理一次内存
for ds in ds_cache.values():
    ds.close()

# ===================== 4. 汇总与绘图 =====================
df_metrics = pd.DataFrame(metrics_records)
csv_path = os.path.join(SAVE_DIR, f"timeseries_metrics_240h_{TIME_TAG}.csv")
df_metrics.to_csv(csv_path, index=False)
print(f"\n📊 240小时评估指标已保存: {csv_path}")

colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#C44E52']
model_color_map = {model: colors[i] for i, model in enumerate(MODELS)}

def plot_240h_timeseries(metric_name, y_label, filename_suffix):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for i, var in enumerate(VARIABLES):
        ax = axes[i]
        var_data = df_metrics[df_metrics['Variable'] == var]
        
        for model in MODELS:
            model_data = var_data[var_data['Model'] == model].sort_values("Lead_Time")
            if not model_data.empty:
                ax.plot(model_data['Lead_Time'], model_data[metric_name], 
                        marker='o', markersize=3, linewidth=2, 
                        label=model, color=model_color_map[model])
        
        ax.set_title(f"{var.upper()} {metric_name} (Up to 240h)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Lead Time (Hours)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc="upper left")
        ax.set_xlim(0, 245)
        ax.set_xticks(np.arange(0, 241, 24)) 
             
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"{filename_suffix}_{TIME_TAG}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 图表已保存: {plot_path}")

plot_240h_timeseries("W-MAE", "Weighted MAE", "W_MAE_240h_timeseries")
plot_240h_timeseries("W-RMSE", "Weighted RMSE", "W_RMSE_240h_timeseries")

print("🎉 评估及双格式偏差场生成全部完成！")