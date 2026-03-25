import os
import numpy as np
import onnxruntime as ort
import cupy as cp
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from tqdm import tqdm
import zarr
import pandas as pd

MODEL_BASE = "/data/tyf/models"
ZARR_PATH = '/data_large/zarr_datasets/ERA5_0_25_1h_zarr_2023only'
print('DATA PATH :', ZARR_PATH)
SAVE_DIR = 'results' 
print('save path:', SAVE_DIR)    
NUM_STEPS = 48   # 预测 0-48h#

# pcc窗口配置
WINDOW_SIZE_CONT = 24  # 连续: 25个点 (1h-25h, span=24h)
WINDOW_STEP_INT = 6    # 间隔步长
WINDOW_POINTS_INT = 5  # 间隔: 5个点 (0,6,12,18,24, span=24h)

TARGET_VARS = ['u10', 'v10', 't2m', 'z500', 't850']

# 基础变量配置
PANGU_DEC = {
    'mode': 'SP', 's_ord': ['msl', 'u10', 'v10', 't2m'], 
    'p_ord': ['z', 'q', 't', 'u', 'v'], 
    'l_ord': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
}
OURS_DEC = {
    'mode': 'SP', 's_ord': ['t2m', 'u10', 'v10', 'msl', 'ssrd', 'tcc'], 
    'p_ord': ['z', 'q', 'u', 'v', 't'], 
    'l_ord': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
}

CONFIGS = {
    # Pangu (1h, 3h, 6h, 24h 全单帧)
    'Pangu_all': {
        'dec': PANGU_DEC, 'use_norm': False,
        'model_paths': {
            '1h': f'{MODEL_BASE}/pangu/pangu_weather_1.onnx',
            '3h': f'{MODEL_BASE}/pangu/pangu_weather_3.onnx',
            '6h': f'{MODEL_BASE}/pangu/pangu_weather_6.onnx',
            '24h': f'{MODEL_BASE}/pangu/pangu_weather_24.onnx'
        }
    },
    # Ours (1h单帧, 3h单帧, 6h双帧)
    'Ours_all': {
        'dec': OURS_DEC, 'use_norm': True, 
        'param': f'{MODEL_BASE}/parameters/s6p5l13_721_1440',
        'model_paths': {
            '1h': f'{MODEL_BASE}/efw_new_v2/model_1h.onnx',
            '3h': f'{MODEL_BASE}/efw_new_v2/model_3h.onnx',
            '6h': f'{MODEL_BASE}/efw_new_v2/model_6h.onnx'
        }
    }
}


class WB2EvaluationDataset(Dataset):
    def __init__(self, zarr_path, start_dates, dec_config):
        self.zarr_path = zarr_path
        self.start_dates = start_dates
        self.dec = dec_config
        self.ds = None  
        
        _ds = zarr.open(zarr_path, mode='r')
        s_vars = list(_ds['surface']['var'][:])
        p_vars = list(_ds['pressure']['var'][:])
        l_vars = list(_ds['pressure']['level'][:])
        self.s_map = [s_vars.index(v) for v in dec_config['s_ord']]
        self.p_map = [p_vars.index(v) for v in dec_config['p_ord']]
        self.l_map = [l_vars.index(l) for l in dec_config['l_ord']]

    def __len__(self): return len(self.start_dates)

    def date_to_idx(self, date_str):
        t = np.datetime64(date_str)
        base = np.datetime64('1979-01-01T00:00:00')
        return int((t - base) // np.timedelta64(1, 'h'))

    def __getitem__(self, index):
        if self.ds is None: self.ds = zarr.open(self.zarr_path, mode='r')
        start_date = self.start_dates[index]
        t0_idx = self.date_to_idx(start_date)
        
        hist_indices = [t0_idx - 24, t0_idx - 6, t0_idx - 1, t0_idx]
        future_indices = np.arange(t0_idx + 1, t0_idx + NUM_STEPS + 1)
        
        indices = np.concatenate([hist_indices, future_indices])
        
        s_data = self.ds['surface']['data'][indices][:, self.s_map]
        p_data = self.ds['pressure']['data'][indices][:, self.p_map][:, :, self.l_map]
        p_data = np.reshape(p_data, (p_data.shape[0], -1, p_data.shape[3], p_data.shape[4]))
        
        if self.dec['mode'] == 'SP':
            batch = np.concatenate([s_data, p_data], axis=1) 
        else:
            batch = np.concatenate([p_data, s_data], axis=1)
            
        return torch.from_numpy(batch.astype(np.float32)), start_date


class IntegratedEvaluator:
    def __init__(self, model_name, device_id):
        self.model_name = model_name
        self.device = device_id
        self.conf = CONFIGS[model_name]
        
        self.sessions = {k: self._create_session(v) for k, v in self.conf['model_paths'].items()}
        
        if self.conf['use_norm']:
            p = self.conf['param']
            self.mean = cp.asarray(np.load(f"{p}/mean.npy")[:, np.newaxis, np.newaxis].astype(np.float32))
            self.std = cp.asarray(np.load(f"{p}/std.npy")[:, np.newaxis, np.newaxis].astype(np.float32))
            
        self.total_ch = len(self.conf['dec']['s_ord']) + len(self.conf['dec']['p_ord']) * len(self.conf['dec']['l_ord'])
        self._build_var_map()

    #def _create_session(self, path):
    #    options = ort.SessionOptions()
    #    providers = [
    #        ('CUDAExecutionProvider', {
    #            'device_id': self.device,
    #            'gpu_mem_limit': 36 * 1024 * 1024 * 1024, 
    #            'arena_extend_strategy': 'kSameAsRequested',
    #        }), 
    #        'CPUExecutionProvider'
    #    ]
    #    return ort.InferenceSession(path, sess_options=options, providers=providers)

    def _create_session(self, path):
        options = ort.SessionOptions()
        # 彻底关闭显存缓存池，极大降低显存峰值
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        #options.intra_op_num_threads = 1
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.device,
                'arena_extend_strategy': 'kSameAsRequested',
            }), 
            'CPUExecutionProvider'
        ]
        return ort.InferenceSession(path, sess_options=options, providers=providers)


    def _build_var_map(self):
        v_map = {}
        dec = self.conf['dec']; num_s = len(dec['s_ord'])
        for v in TARGET_VARS:
            if v in dec['s_ord']: v_map[v] = dec['s_ord'].index(v)
            else:
                v_pre, lev = v[0], int(v[1:])
                v_map[v] = num_s + dec['p_ord'].index(v_pre) * len(dec['l_ord']) + dec['l_ord'].index(lev)
        if 'ssrd' in dec['s_ord']: v_map['ssrd'] = dec['s_ord'].index('ssrd')
        self.var_map = v_map

    def calc_spatial_pcc_map(self, p_list, g_list):
        p, g = cp.asarray(p_list), cp.asarray(g_list)
        pm, gm = cp.mean(p, 0), cp.mean(g, 0)
        pd, gd = p - pm, g - gm
        num = cp.sum(pd * gd, 0)
        den = cp.sqrt(cp.sum(pd**2, 0) * cp.sum(gd**2, 0))
        spatial_pcc = (num / (den + 1e-12))
        return spatial_pcc.get()

    def run_eval(self, gt_batch, start_date):
        gt_np = gt_batch.numpy() if torch.is_tensor(gt_batch) else gt_batch
        
        idx_t0 = 3 
        
        if 'Pangu' in self.model_name:
            s_in = gt_np[idx_t0][:4].astype(np.float32)
            p_in = gt_np[idx_t0][4:].reshape(5, 13, 721, 1440).astype(np.float32)
            
            # Pangu 所有模型初始都是单帧 t0
            current_input_1h_s = s_in.copy(); current_input_1h_p = p_in.copy()
            current_input_3h_s = s_in.copy(); current_input_3h_p = p_in.copy()
            current_input_6h_s = s_in.copy(); current_input_6h_p = p_in.copy()
            current_input_24h_s = s_in.copy(); current_input_24h_p = p_in.copy()
            
        elif 'Ours' in self.model_name:
            idx_tm6 = 1
            f_t0 = (cp.asarray(gt_np[idx_t0]) - self.mean) / self.std
            f_tm6 = (cp.asarray(gt_np[idx_tm6]) - self.mean) / self.std
            # Ours 1h, 3h 为单帧 t0
            current_input_1h = f_t0[np.newaxis, ...].get().astype(np.float32)
            current_input_3h = f_t0[np.newaxis, ...].get().astype(np.float32)
            # Ours 6h 为双帧 [t-6, t0]
            current_input_6h = cp.concatenate([f_tm6, f_t0], 0)[np.newaxis, ...].get().astype(np.float32)

        preds = [] 
        for step in range(1, NUM_STEPS + 1):
            
            # 严格阶梯触发逻辑
            use_24h = ('Pangu' in self.model_name) and (step % 24 == 0)
            use_6h  = (not use_24h) and (step % 6 == 0)
            use_3h  = (not use_24h) and (not use_6h) and (step % 3 == 0)
            if 'Pangu' in self.model_name:
                if use_24h:
                    p_out, s_out = self.sessions['24h'].run(None, {'input': current_input_24h_p, 'input_surface': current_input_24h_s})
                    curr_s, curr_p = s_out, p_out
                    current_input_1h_s, current_input_1h_p = curr_s, curr_p
                    current_input_3h_s, current_input_3h_p = curr_s, curr_p
                    current_input_6h_s, current_input_6h_p = curr_s, curr_p
                    current_input_24h_s, current_input_24h_p = curr_s, curr_p
                elif use_6h:
                    p_out, s_out = self.sessions['6h'].run(None, {'input': current_input_6h_p, 'input_surface': current_input_6h_s})
                    curr_s, curr_p = s_out, p_out
                    current_input_1h_s, current_input_1h_p = curr_s, curr_p
                    current_input_3h_s, current_input_3h_p = curr_s, curr_p
                    current_input_6h_s, current_input_6h_p = curr_s, curr_p
                elif use_3h:
                    p_out, s_out = self.sessions['3h'].run(None, {'input': current_input_3h_p, 'input_surface': current_input_3h_s})
                    curr_s, curr_p = s_out, p_out
                    current_input_1h_s, current_input_1h_p = curr_s, curr_p
                    current_input_3h_s, current_input_3h_p = curr_s, curr_p
                else: # 1h
                    p_out, s_out = self.sessions['1h'].run(None, {'input': current_input_1h_p, 'input_surface': current_input_1h_s})
                    curr_s, curr_p = s_out, p_out
                    current_input_1h_s, current_input_1h_p = curr_s, curr_p
                    
                out_gpu = cp.concatenate([cp.asarray(curr_s), cp.asarray(curr_p).reshape(-1, 721, 1440)])
            
            elif 'Ours' in self.model_name:
                if use_6h:
                    out_val = self.sessions['6h'].run(None, {'input': current_input_6h})[0]
                    out_gpu = cp.asarray(out_val[0, :self.total_ch])
                    out_4d = cp.asarray(out_val[:, :self.total_ch])
                    current_input_1h = out_4d.get().astype(np.float32)
                    current_input_3h = out_4d.get().astype(np.float32)
                    # 6h是双帧，丢弃老的，拼接新的
                    current_input_6h = cp.concatenate([cp.asarray(current_input_6h[:, self.total_ch:]), out_4d], axis=1).get().astype(np.float32)
                elif use_3h:
                    out_val = self.sessions['3h'].run(None, {'input': current_input_3h})[0]
                    out_gpu = cp.asarray(out_val[0, :self.total_ch])
                    out_4d = cp.asarray(out_val[:, :self.total_ch])
                    current_input_1h = out_4d.get().astype(np.float32)
                    current_input_3h = out_4d.get().astype(np.float32) # 单帧直接覆盖
                else: # 1h
                    out_val = self.sessions['1h'].run(None, {'input': current_input_1h})[0]
                    out_gpu = cp.asarray(out_val[0, :self.total_ch])
                    out_4d = cp.asarray(out_val[:, :self.total_ch])
                    current_input_1h = out_4d.get().astype(np.float32) # 单帧直接覆盖

            preds.append(out_gpu)

        # 指标计算
        temp_buffer = {v: {'rmse': [], 'preds_np': [], 'gts_np': []} for v in self.var_map}
        for step, pred in enumerate(preds):
            real_step = step + 1
            gt = cp.asarray(gt_np[idx_t0 + real_step])
            if self.conf['use_norm']: pred = pred * self.std + self.mean
            for v, idx in self.var_map.items():
                p_v, g_v = pred[idx], gt[idx]
                temp_buffer[v]['rmse'].append(float(cp.sqrt(cp.mean((p_v - g_v)**2)).get()))
                temp_buffer[v]['preds_np'].append(p_v.get())
                temp_buffer[v]['gts_np'].append(g_v.get())

        final_res = {}
        for v in self.var_map:
            pcc_cont = []
            num_windows_cont = NUM_STEPS - WINDOW_SIZE_CONT + 1 
            for start_h in range(num_windows_cont):
                end_h = start_h + WINDOW_SIZE_CONT 
                win_p = temp_buffer[v]['preds_np'][start_h:end_h]
                win_g = temp_buffer[v]['gts_np'][start_h:end_h]
                pcc_cont.append(self.calc_spatial_pcc_map(win_p, win_g))

            pcc_int = []
            # TODO 这里是硬编码 到时候可以改改
            num_windows_int = NUM_STEPS - 24 
            for start_h in range(num_windows_int):
                indices = [start_h + i * WINDOW_STEP_INT for i in range(WINDOW_POINTS_INT)]
                win_p = [temp_buffer[v]['preds_np'][i] for i in indices]
                win_g = [temp_buffer[v]['gts_np'][i] for i in indices]
                pcc_int.append(self.calc_spatial_pcc_map(win_p, win_g))
            
            final_res[v] = {
                'rmse': temp_buffer[v]['rmse'],
                'pcc_continuous': np.array(pcc_cont), 
                'pcc_interval': np.array(pcc_int)     
            }
            
        cp.get_default_memory_pool().free_all_blocks()
        return final_res


def gpu_worker(gpu_idx, model_name, date_list, num_gpus, return_dict):
    evaluator = IntegratedEvaluator(model_name, gpu_idx)
    dataset = WB2EvaluationDataset(ZARR_PATH, date_list, CONFIGS[model_name]['dec'])
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    
    agg = {}
    count = 0
    
    for i, (gt_batch, start_date) in enumerate(tqdm(loader, desc=f"GPU {gpu_idx}")):
        try:
            res = evaluator.run_eval(gt_batch[0], start_date[0])
            
            if count == 0:
                num_cont = NUM_STEPS - WINDOW_SIZE_CONT + 1
                num_int = NUM_STEPS - 24
                for v in evaluator.var_map:
                    agg[v] = {
                        'rmse': np.zeros(NUM_STEPS),
                        'pcc_continuous': np.zeros((num_cont, 721, 1440)),
                        'pcc_interval': np.zeros((num_int, 721, 1440))
                    }
            
            for v in res:
                agg[v]['rmse'] += np.array(res[v]['rmse'])
                agg[v]['pcc_continuous'] += res[v]['pcc_continuous']
                agg[v]['pcc_interval'] += res[v]['pcc_interval']
            
            count += 1
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"Error {start_date}: {e}")
            continue

    return_dict[gpu_idx] = (agg, count)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=['Pangu_all', 'Ours_all'])
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    dates = []
    curr, end = datetime(2023, 1, 1), datetime(2023, 1, 31)
    while curr <= end: dates.append(curr.strftime('%Y-%m-%dT%H:%M:%S')); curr += timedelta(days=1)
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    chunks = np.array_split(dates, args.gpus)
    
    procs = [mp.Process(target=gpu_worker, args=(i, args.model, chunks[i], args.gpus, return_dict)) 
             for i in range(args.gpus) if len(chunks[i]) > 0]
    
    for p in procs: p.start()
    for p in procs: p.join()

    final_agg, total = {}, 0
    first = True
    for i in range(args.gpus):
        if i in return_dict:
            res, cnt = return_dict[i]
            total += cnt
            if first:
                final_agg = res
                first = False
            else:
                for v in final_agg:
                    final_agg[v]['rmse'] += res[v]['rmse']
                    final_agg[v]['pcc_continuous'] += res[v]['pcc_continuous']
                    final_agg[v]['pcc_interval'] += res[v]['pcc_interval']

    if total > 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
        csv_rows = []
        for v, d in final_agg.items():
            avg_rmse = d['rmse'] / total
            
            avg_pcc_cont_map = d['pcc_continuous'] / total
            avg_pcc_cont = np.mean(avg_pcc_cont_map, axis=(1, 2))
            
            avg_pcc_int_map = d['pcc_interval'] / total
            avg_pcc_int = np.mean(avg_pcc_int_map, axis=(1, 2))
            
            csv_rows.append([f"{v}_rmse"] + avg_rmse.tolist())
            csv_rows.append([f"{v}_pcc_continuous"] + avg_pcc_cont.tolist())
            csv_rows.append([f"{v}_pcc_interval"] + avg_pcc_int.tolist())

        pd.DataFrame(csv_rows).to_csv(os.path.join(SAVE_DIR, f"{args.model}_metrics.csv"), index=False, header=False)
        print(f"完成！CSV 见: {SAVE_DIR}/{args.model}_metrics.csv")