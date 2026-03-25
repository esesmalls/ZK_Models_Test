import os
import numpy as np
import cupy as cp
import onnxruntime as ort

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .dataloader import MyDataloader 
import pandas as pd

class WeatherForecast:
    def __init__(self, device=0):
        self.device = device
        self.sessions = {}     
        self.norm_params = {}  
        
        ours_path = '/data/tyf/models'

        model_base_path = '/data/webset/models'
        parameter_base_path = '/data/webset/parameters'
        self.configs = {
            'FengWu': {
                'dec': {'mode': 'SP', 's_ord': ['u10', 'v10', 't2m', 'msl'], 
                        'p_ord': ['z', 'q', 'u', 'v', 't'], 'l_ord': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
                'in_frames': 2, 'use_norm': True, 'param': os.path.join(parameter_base_path, 'fengwu'),
                'model_path': os.path.join(model_base_path, 'fengwu/fengwu_v2.onnx')
            },
            'PanGu': {
                'dec': {'mode': 'SP', 's_ord': ['msl', 'u10', 'v10', 't2m'], 
                        'p_ord': ['z', 'q', 't', 'u', 'v'], 'l_ord': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]},
                'in_frames': 1, 'use_norm': False, 'param': None,
                'model_paths': {
                    #'1h': os.path.join(base_path, 'pangu/pangu_weather_1.onnx'),# 
                    '6h': os.path.join(model_base_path, 'pangu/pangu_weather_6.onnx'), 
                    '24h': os.path.join(model_base_path, 'pangu/pangu_weather_24.onnx')
                }
            },
            'Ours': {
                'dec': {'mode': 'SP', 's_ord': ['t2m', 'u10', 'v10', 'msl', 'ssrd', 'tcc'], 
                        'p_ord': ['z', 'q', 'u', 'v', 't'], 'l_ord': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]},
                'in_frames': 'special',  # 标记为特殊处理，需要 t-24h, t-6h, t0
                'use_norm': True, 
                'param': os.path.join(parameter_base_path, 's6p5l13_721_1440'),
                'model_paths': {
                    '6h': os.path.join(ours_path, 'efw/model_6h.onnx'),
                    '24h': os.path.join(ours_path, 'efw_new/model_24h.onnx')
                }
            },
            'EFengWu': {
                'dec': {'mode': 'SP', 's_ord': ['t2m', 'u10', 'v10', 'msl', 'ssrd', 'tcc'], 
                        'p_ord': ['z', 'q', 'u', 'v', 't'], 'l_ord': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]},
                'in_frames': 2, 'use_norm': True, 'param': os.path.join(parameter_base_path, 's6p5l13_721_1440'),
                'model_paths': {
                    #'1h': os.path.join(base_path, 'efw/model_1h.onnx'),#
                    '6h': os.path.join(model_base_path, 'efw/model_6h.onnx')
                }
            },
            'FuXi': {
                'dec': {
                    'mode': 'PS', 's_ord': ['t2m', 'u10', 'v10', 'msl', 'tp'], 
                    'p_ord': ['z', 't', 'u', 'v', 'r'], 'l_ord': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]},
                'in_frames': 2, 'use_norm': False, 'param': None,
                'model_paths': {
                    'short': os.path.join(model_base_path, 'fuxi/short.onnx'),
                    'medium': os.path.join(model_base_path, 'fuxi/medium.onnx')
                }
            }
        }

    def _get_fuxi_tembs(self, start_date_str, total_step=40):
        init_time = pd.to_datetime(start_date_str)
        tembs = []
        for i in range(total_step):
            hours = np.array([pd.Timedelta(hours=t*6) for t in [i-1, i, i+1]])
            times = init_time + hours
            times_encoded = [(t.dayofyear/366, t.hour/24) for t in times]
            temb = np.array(times_encoded, dtype=np.float32).flatten()
            temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
            tembs.append(temb[np.newaxis, :].astype(np.float32))
        return tembs

    def _clear_gpu_memory(self):
        import gc
        self.sessions.clear()
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

    def _create_session(self, path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena, options.enable_mem_pattern, options.enable_mem_reuse = True, True, False
        options.intra_op_num_threads = 16
        options.log_severity_level = 3  # filter warning
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.device,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'HEURISTIC',  # 改为 HEURISTIC 降低显存压力
                'do_copy_in_default_stream': True,
            }), 
            'CPUExecutionProvider'
        ]
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def _prepare_model_resources(self, model_name):
        if model_name in self.sessions: return 
        
        self._clear_gpu_memory()
        
        print(f'正在加载 {model_name} 模型')
        conf = self.configs[model_name]
        if model_name == 'FengWu':
            self.sessions[model_name] = self._create_session(conf['model_path'])
        else:
            self.sessions[model_name] = {k: self._create_session(v) for k, v in conf['model_paths'].items()}

        if conf['use_norm']:
            p = conf['param']
            m_f, s_f = ('data_mean.npy', 'data_std.npy') if model_name == 'FengWu' else ('mean.npy', 'std.npy')
            mean = cp.asarray(np.load(os.path.join(p, m_f))[:, np.newaxis, np.newaxis].astype(np.float32))
            std = cp.asarray(np.load(os.path.join(p, s_f))[:, np.newaxis, np.newaxis].astype(np.float32))
            self.norm_params[model_name] = {'mean': mean, 'std': std}

    def run(self, model: str, start_date: str):
        try: 
            if model not in self.configs: return False, f'不支持的模型类型: {model}' 
            self._prepare_model_resources(model)
            conf, dec = self.configs[model], self.configs[model]['dec']
            
            start_dt = np.datetime64(start_date)
            offset = 6 if conf['in_frames'] == 2 else 0
            dt_start = str(start_dt - np.timedelta64(offset, 'h'))
            dl = MyDataloader('/data_large/zarr_datasets/ERA5_0_25_1h_zarr_2023only', dt_start, dec, n_threads=0)
            
            frames = []
            if model == 'Ours':
                dt_base = str(start_dt - np.timedelta64(24, 'h'))
                dl = MyDataloader('/data_large/zarr_datasets/ERA5_0_25_1h_zarr_2023only', dt_base, dec, n_threads=0)
                
                f_m24 = dl.get_data_by_idx(0)   # 对应 t-24h
                f_m6  = dl.get_data_by_idx(18)  # 对应 t-6h (24-6=18)
                f_0   = dl.get_data_by_idx(24)  # 对应 t0
                
                if any(f is None for f in [f_m24, f_m6, f_0]): return False, 'Ours 初始场加载不足'
                frames = [f_m24, f_m6, f_0]
            else:
                offset = 6 if conf['in_frames'] == 2 else 0
                dt_load_start = str(start_dt - np.timedelta64(offset, 'h'))
                dl = MyDataloader('/data_large/zarr_datasets/ERA5_0_25_1h_zarr_2023only', dt_load_start, dec, n_threads=0)
                
                if conf['in_frames'] == 2:
                    f_old, f_now = dl.get_data_by_idx(0), dl.get_data_by_idx(6)
                    if f_old is not None: frames.append(f_old)
                    if f_now is not None: frames.append(f_now)
                else:
                    f_now = dl.get_data_by_idx(0)
                    if f_now is not None: frames.append(f_now)

            if len(frames) < (3 if model == 'Ours' else conf['in_frames']):
                return False, '初始场加载失败'

            if conf['use_norm']:
                p_np = {k: v.get() for k, v in self.norm_params[model].items()}
                frames = [(f - p_np['mean']) / p_np['std'] for f in frames]

            num_s, n_lev = len(dec['s_ord']), len(dec['l_ord'])
            total_ch = num_s + len(dec['p_ord']) * n_lev
            final_results = np.zeros((40, total_ch, 721, 1440), dtype=np.float32)
            sess_obj = self.sessions[model]

            if model == 'PanGu':
                s_in, p_in = frames[-1][:4].astype(np.float32), frames[-1][4:].reshape(5, 13, 721, 1440).astype(np.float32)
                s_24, p_24 = s_in, p_in
                for step in tqdm(range(40), desc='盘古推理'):
                    if (step + 1) % 4 == 0:
                        p_24, s_24 = sess_obj['24h'].run(None, {'input': p_24, 'input_surface': s_24})
                        s_in, p_in = s_24, p_24
                    else:
                        p_in, s_in = sess_obj['6h'].run(None, {'input': p_in, 'input_surface': s_in})
                    final_results[step] = np.concatenate([s_in, p_in.reshape(-1, 721, 1440)], axis=0)

            elif model in ['FengWu', 'EFengWu']:
                input_arr = np.concatenate([frames[0], frames[1]], axis=0)[np.newaxis, ...].astype(np.float32)
                cur_sess = sess_obj if model == 'FengWu' else sess_obj['6h']
                for step in tqdm(range(40), desc=f'{model}推理'):
                    output = cur_sess.run(None, {'input': input_arr})[0]
                    input_arr = np.concatenate((input_arr[:, total_ch:], output[:, :total_ch]), axis=1)
                    final_results[step] = output[0, :total_ch]
            
            elif model == 'FuXi':
                input_arr = np.stack([frames[0], frames[1]], axis=0) 
                input_arr = input_arr[np.newaxis, ...].astype(np.float32) 
                tembs = self._get_fuxi_tembs(start_date, 40)
                for step in tqdm(range(40), desc='FuXi 推理'):
                    temb = tembs[step]
                    active_sess = sess_obj['short'] if step < 20 else sess_obj['medium']
                    output = active_sess.run(None, {'input': input_arr, 'temb': temb})[0]
                    final_results[step] = output[0, -1, :total_ch]
                    input_arr = output

            elif model == 'Ours':
                cur_in_6 = np.concatenate([frames[1], frames[2]], axis=0)[np.newaxis, ...].astype(np.float32)
                cur_in_24 = np.concatenate([frames[0], frames[2]], axis=0)[np.newaxis, ...].astype(np.float32)
                
                for step in tqdm(range(40), desc='Ours推理'):
                    is_24h_node = (step + 1) % 4 == 0
                    
                    if is_24h_node and (step + 1) > 6:
                        output = sess_obj['24h'].run(None, {'input': cur_in_24})[0]
                    else:
                        output = sess_obj['6h'].run(None, {'input': cur_in_6})[0]
                    
                    pred_frame = output[:, :total_ch]
                    
                    if is_24h_node:
                        cur_in_24 = np.concatenate([cur_in_24[:, total_ch:], pred_frame], axis=1)
                        cur_in_6 = np.concatenate([cur_in_6[:, total_ch:], pred_frame], axis=1)
                    else:
                        cur_in_6 = np.concatenate([cur_in_6[:, total_ch:], pred_frame], axis=1)
                    
                    final_results[step] = pred_frame[0]

            if conf['use_norm']:
                print('正在反归一化')
                p = self.norm_params[model]
                for step in range(40):
                    gpu_frame = cp.asarray(final_results[step])
                    gpu_frame = (gpu_frame * p['std']) + p['mean']
                    final_results[step] = gpu_frame.get()
                cp.get_default_memory_pool().free_all_blocks()

            print('\n正在保存预报结果')
            target_dir = f'/data_large/web_database/forecast/{model}'
            os.makedirs(target_dir, exist_ok=True)
            tag = start_date.replace('-', '').replace(':', '')[:11]
            s_start, p_start = (0, num_s) if dec['mode'] == 'SP' else (total_ch - num_s, 0)

            with ThreadPoolExecutor(max_workers=10) as executor:
                for i, v in enumerate(dec['s_ord']):
                    path = f'{target_dir}/{v}_{tag}.npy'
                    executor.submit(np.save, path, final_results[:, s_start + i])

                for v_i, v in enumerate(dec['p_ord']):
                    for l_i, l_val in enumerate(dec['l_ord']):
                        idx = p_start + v_i * n_lev + l_i
                        path = f'{target_dir}/{v}_{l_val}_{tag}.npy'
                        executor.submit(np.save, path, final_results[:, idx])

            return True, f'{model} 推理任务成功执行 ({tag})' 
        except Exception as e:
            return False, f'推理崩溃: {str(e)}'