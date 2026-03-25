import numpy as np
import xarray as xr
import os
import re
import warnings
from datetime import datetime

# ================= 配置与警告屏蔽 =================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 【更新】研究区域范围 (更广阔的亚洲区域)
REGION_LON_MIN, REGION_LON_MAX = 60, 140
REGION_LAT_MIN, REGION_LAT_MAX = 10, 60

# 【关键】基础数据目录
# 预期结构: D:\My_Data\BigModel\Ours\20230101T00\t2m.nc
BASE_DATA_DIR = r'D:\My_Data\BigModel\Ours'

# 需要处理的目标变量关键字
TARGET_VARS = ['t2m', 'u10', 'v10', 'ssrd']

# 输出目录
OUTPUT_DIR = r'.\output_nc'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 插值分辨率
TARGET_RESOLUTION = 0.1


# =================================================

def parse_folder_name(folder_name):
    """
    验证文件夹名是否为合法的起报时间格式
    支持: 20230101T00 或 2023010100
    返回标准化字符串: YYYYMMDDTHH
    """
    pattern = r'^(\d{8})T?(\d{2})$'
    match = re.match(pattern, folder_name)
    if match:
        return f"{match.group(1)}T{match.group(2)}"
    return None


def find_var_file(folder_path, var_keyword):
    """在文件夹中查找包含变量关键字的 .nc 文件"""
    for fname in os.listdir(folder_path):
        if not fname.endswith('.nc'):
            continue
        if var_keyword.lower() in fname.lower():
            return os.path.join(folder_path, fname)
    return None


def get_region_indices(lon, lat, lon_min, lon_max, lat_min, lat_max):
    """获取区域索引"""
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"在给定范围内未找到数据点。\nLat范围: {lat_min}-{lat_max}, 数据Lat: {lat.min()}-{lat.max()}\nLon范围: {lon_min}-{lon_max}, 数据Lon: {lon.min()}-{lon.max()}")
    return lat_idx, lon_idx


def convert_lon_360_to_180_numpy(lon, data_values):
    """将 0-360 经度转换为 -180-180 并排序"""
    new_lon = lon.copy()
    mask = new_lon > 180
    new_lon[mask] -= 360
    sort_idx = np.argsort(new_lon)
    new_lon = new_lon[sort_idx]

    if data_values.ndim == 2:
        new_data = data_values[:, sort_idx]
    elif data_values.ndim == 3:
        new_data = data_values[:, :, sort_idx]
    else:
        new_data = np.take(data_values, sort_idx, axis=-1)

    return new_lon, new_data


def interpolate_to_grid_numpy(data_values, lon, lat, resolution):
    """
    通用插值函数
    :param data_values: 数据数组 (step, lat, lon) 或 (lat, lon)
    :param lon: 原始经度
    :param lat: 原始纬度
    :param resolution: 目标分辨率
    :return: interpolated_values, new_lon, new_lat
    """
    print(f"      -> 执行 {resolution}° 插值 (数据维度: {data_values.shape})...")

    dims = ['step', 'latitude', 'longitude']
    if data_values.ndim == 2:
        dims = ['latitude', 'longitude']

    temp_da = xr.DataArray(data_values, dims=dims, coords={'latitude': lat, 'longitude': lon})

    step = resolution
    buffer = step / 2.0

    new_lat = np.arange(REGION_LAT_MIN - buffer, REGION_LAT_MAX + buffer + step, step)
    new_lon = np.arange(REGION_LON_MIN - buffer, REGION_LON_MAX + buffer + step, step)

    interpolated_da = temp_da.interp(latitude=new_lat, longitude=new_lon, method='linear')

    # 裁剪到精确的研究区域
    interpolated_da = interpolated_da.sel(
        latitude=slice(REGION_LAT_MIN, REGION_LAT_MAX),
        longitude=slice(REGION_LON_MIN, REGION_LON_MAX)
    )

    return interpolated_da.values, interpolated_da.longitude.values, interpolated_da.latitude.values


def save_interpolated_data_to_nc(data_val, lon, lat, output_path, var_name='t2m', init_time_str=None, original_ds=None):
    """
    将插值后的数据保存为 NC 文件
    """
    print(f"      -> 保存数据到: {os.path.basename(output_path)}")

    dims = []
    coords = {}

    if data_val.ndim == 3:
        dims = ['step', 'latitude', 'longitude']
        if original_ds is not None and 'step' in original_ds.coords:
            coords['step'] = original_ds.coords['step'].values
        else:
            coords['step'] = np.arange(data_val.shape[0])

    elif data_val.ndim == 2:
        dims = ['latitude', 'longitude']
    else:
        raise ValueError("数据维度不支持，必须是 2D 或 3D")

    coords['latitude'] = lat
    coords['longitude'] = lon

    new_da = xr.DataArray(data_val, dims=dims, coords=coords, name=var_name)

    # 添加属性
    units_map = {'t2m': 'K', 'u10': 'm s-1', 'v10': 'm s-1', 'ssrd': 'J m-2'}
    new_da.attrs['units'] = units_map.get(var_name, 'unknown')
    new_da.attrs['long_name'] = f'Interpolated {var_name} at {abs(lat[1] - lat[0]):.2f} degree resolution'
    new_da.attrs['source'] = 'Interpolated from original model output (All time steps)'

    # 【重要】写入起报时间
    if init_time_str:
        new_da.attrs['init_time'] = init_time_str

    if 'step' in coords:
        if original_ds is not None and 'step' in original_ds.coords:
            new_da.coords['step'].attrs = original_ds.coords['step'].attrs
        else:
            new_da.coords['step'].attrs['units'] = 'hours'
            new_da.coords['step'].attrs['long_name'] = 'Forecast step'

    ds_out = new_da.to_dataset()

    ds_out.attrs['Conventions'] = 'CF-1.8'
    ds_out.attrs[
        'history'] = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} via interpolation to {abs(lat[1] - lat[0]):.3f} degrees.'
    ds_out.attrs['region'] = f'Asia ({REGION_LON_MIN}-{REGION_LON_MAX}E, {REGION_LAT_MIN}-{REGION_LAT_MAX}N)'
    if init_time_str:
        ds_out.attrs['source_init_time'] = init_time_str

    encoding = {var_name: {'zlib': True, 'complevel': 4, '_FillValue': 1e20}}

    ds_out.to_netcdf(output_path, encoding=encoding)

    steps_count = data_val.shape[0] if data_val.ndim == 3 else 1
    print(f"      ✅ 成功保存 (包含 {steps_count} 个预报时次): {output_path}")


########################################################################################################################
# 主程序
########################################################################################################################
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 开始批量处理多时次预报数据 (按文件夹时间顺序)")
    print(f"📂 根目录: {BASE_DATA_DIR}")
    print(f"🎯 目标变量: {TARGET_VARS}")
    print(f"🌍 区域: [{REGION_LON_MIN}, {REGION_LON_MAX}] x [{REGION_LAT_MIN}, {REGION_LAT_MAX}]")
    print(f"📐 分辨率: {TARGET_RESOLUTION}°")
    print("=" * 70)

    if not os.path.exists(BASE_DATA_DIR):
        print(f"❌ 错误：根目录不存在 -> {BASE_DATA_DIR}")
        exit(1)

    # 1. 扫描所有一级子文件夹，筛选出符合时间格式的文件夹
    time_folders = []
    try:
        entries = os.listdir(BASE_DATA_DIR)
    except Exception as e:
        print(f"❌ 无法读取目录: {e}")
        exit(1)

    for entry in entries:
        full_path = os.path.join(BASE_DATA_DIR, entry)
        if os.path.isdir(full_path):
            init_time = parse_folder_name(entry)
            if init_time:
                time_folders.append({
                    'path': full_path,
                    'name': entry,
                    'init_time': init_time
                })

    if not time_folders:
        print("⚠️ 未找到任何符合格式 (YYYYMMDDTHH) 的时间文件夹。")
        print("   请检查目录结构，例如: D:\\...\\Ours\\20230101T00\\")
        exit(0)

    # 2. 按时间顺序排序 (从早到晚)
    time_folders.sort(key=lambda x: x['init_time'])

    print(f"📅 发现 {len(time_folders)} 个起报时次，将按顺序处理:")
    for i, tf in enumerate(time_folders):
        print(f"   [{i + 1}] {tf['init_time']} (文件夹: {tf['name']})")
    print("-" * 70)

    total_success = 0
    total_fail = 0

    # 3. 循环处理每个时间文件夹 (从第一个到最后一个)
    for t_info in time_folders:
        folder_path = t_info['path']
        init_time_str = t_info['init_time']

        print(f"\n🕒 正在处理起报时间: {init_time_str}")
        print(f"   📂 来源文件夹: {t_info['name']}")

        # 在该文件夹内循环处理每个目标变量
        for var_key in TARGET_VARS:
            file_path = find_var_file(folder_path, var_key)

            if not file_path:
                # 静默跳过或打印提示，视需求而定，这里打印简要提示
                # print(f"   ⚠️  跳过变量 {var_key}: 未找到文件")
                continue

            print(f"   📄 处理变量: {var_key} -> {os.path.basename(file_path)}")

            try:
                ds = xr.open_dataset(file_path)

                # 确认变量名
                actual_var = var_key
                if actual_var not in ds.variables:
                    candidates = [v for v in ds.variables if var_key in v.lower()]
                    if candidates:
                        actual_var = candidates[0]
                        print(f"      ℹ️  自动映射变量名: {actual_var}")
                    else:
                        raise ValueError(f"文件中未找到包含 '{var_key}' 的变量")

                data_var = ds[actual_var]

                # 获取经纬度
                lon_vals = None
                lat_vals = None

                if 'lon' in ds:
                    lon_vals = ds['lon'].values
                elif 'longitude' in ds:
                    lon_vals = ds['longitude'].values
                elif 'lon' in data_var.dims:
                    lon_vals = ds.coords['lon'].values

                if 'lat' in ds:
                    lat_vals = ds['lat'].values
                elif 'latitude' in ds:
                    lat_vals = ds['latitude'].values
                elif 'lat' in data_var.dims:
                    lat_vals = ds.coords['lat'].values

                if lon_vals is None or lat_vals is None:
                    raise ValueError("无法提取经纬度坐标")

                if hasattr(lon_vals, 'values'): lon_vals = lon_vals.values
                if hasattr(lat_vals, 'values'): lat_vals = lat_vals.values

                # 空间截取
                lat_idx, lon_idx = get_region_indices(lon_vals, lat_vals, REGION_LON_MIN, REGION_LON_MAX,
                                                      REGION_LAT_MIN, REGION_LAT_MAX)

                # 时间维度处理 (保留所有 Step)
                if 'init_time' in data_var.dims:
                    subset = data_var.isel(init_time=0)
                elif 'time' in data_var.dims:
                    subset = data_var.isel(time=0)
                else:
                    subset = data_var

                region_data = subset.isel({'lat': lat_idx, 'lon': lon_idx})

                n_steps = region_data.sizes.get('step', 1)
                print(f"      📊 数据维度: {region_data.shape}, 预报时效数: {n_steps}")

                # 提取数值 (全量)
                raw_vals = region_data.values

                # 缩放处理 (仅针对温度)
                proc_vals = raw_vals
                if var_key == 't2m' and np.nanmax(raw_vals) > 1000:
                    proc_vals = raw_vals / 100.0
                    print(f"      ⚖️  检测到整型缩放，已除以 100")

                reg_lat = lat_vals[lat_idx]
                reg_lon = lon_vals[lon_idx]

                # 经度转换
                p_lon, p_data = reg_lon, proc_vals
                if np.max(reg_lon) > 180:
                    p_lon, p_data = convert_lon_360_to_180_numpy(reg_lon, proc_vals)

                # 插值
                interp_val, interp_lon, interp_lat = interpolate_to_grid_numpy(
                    p_data, p_lon, reg_lat, TARGET_RESOLUTION
                )

                # 构造输出文件名: {变量}_{起报时间}_interp_{分辨率}deg.nc
                res_str = str(TARGET_RESOLUTION).replace('.', '')
                out_name = f"{var_key}_{init_time_str}_interp_{res_str}deg.nc"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                # 保存
                save_interpolated_data_to_nc(
                    interp_val, interp_lon, interp_lat,
                    out_path,
                    var_name=actual_var,
                    init_time_str=init_time_str,
                    original_ds=region_data
                )

                ds.close()
                total_success += 1

            except Exception as e:
                print(f"      ❌ 处理失败: {e}")
                import traceback

                traceback.print_exc()
                total_fail += 1
                if 'ds' in locals():
                    try:
                        ds.close()
                    except:
                        pass

    print("\n" + "=" * 70)
    print(f"🏁 全部任务结束")
    print(f"✅ 成功处理文件数: {total_success}")
    print(f"❌ 失败文件数: {total_fail}")
    if total_success > 0:
        print(f"💾 输出目录: {os.path.abspath(OUTPUT_DIR)}")
        print("💡 提示: 输出文件已包含完整的预报时效序列，并按起报时间命名。")
    print("=" * 70)