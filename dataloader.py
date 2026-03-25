import time
import zarr
import threading
import numpy as np

class MyDataloader:
    def __init__(self, data_dir, start_date, des_dic, buff_len=32, n_threads=16):
        self._buff_len = buff_len
        # 计算相对于 1979-01-01 的小时偏移 
        self.start_idx = (np.datetime64(start_date) - np.datetime64('1979-01-01T00:00:00')) // np.timedelta64(1, 'h')

        self._ds = zarr.open(data_dir, mode='r')
        self._max_len = self._ds['pressure']['data'].shape[0]

        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._task_idx = self.start_idx
        self._task_lock = threading.Lock()
        self._current_idx = self.start_idx
        self._current_lock = threading.Lock()
        self._running = 0
        self._running_lock = threading.Lock()

        # 变量映射 
        surface_ord = self._ds['surface']['var'][:]
        pressure_ord = self._ds['pressure']['var'][:]
        level_ord = self._ds['pressure']['level'][:]

        self._mode = des_dic['mode']
        self._surface_map = self._conver_slice(surface_ord, des_dic['s_ord'])
        self._pressure_map = self._conver_slice(pressure_ord, des_dic['p_ord'])
        self._level_map = self._conver_slice(level_ord, des_dic['l_ord'])

        self._threads = []
        if n_threads > 0:
            for _ in range(n_threads):
                thr = threading.Thread(target=self._worker, daemon=True)
                thr.start()
                self._threads.append(thr)

    @staticmethod
    def _conver_slice(ori_ord, new_ord, bias=0):
        map_slice = []
        ori_ord_list = ori_ord.tolist()
        for x in new_ord:
            map_slice.append(ori_ord_list.index(x) + bias)
        return map_slice

    def get_data_by_idx(self, offset_idx):

        task = self.start_idx + offset_idx
        if task >= self._max_len:
            return None
        
        s_vars = self._ds['surface']['data'][task][self._surface_map]
        p_vars = self._ds['pressure']['data'][task][self._pressure_map][:, self._level_map]
        p_vars = np.reshape(p_vars, [-1, p_vars.shape[2], p_vars.shape[3]])

        if self._mode == 'SP':
            data_arr = np.concatenate([s_vars, p_vars], axis=0).astype(np.float32)
        elif self._mode == 'PS':
            data_arr = np.concatenate([p_vars, s_vars], axis=0).astype(np.float32)
        
        return data_arr

    def _worker(self):
        try:
            with self._running_lock:
                self._running += 1
            while True:
                with self._task_lock:
                    if self._task_idx < self._max_len:
                        task = self._task_idx
                        self._task_idx += 1
                    else:
                        task = None

                if task is not None:
                    # 读取原始数据并按映射重排
                    s_vars = self._ds['surface']['data'][task][self._surface_map]
                    p_vars = self._ds['pressure']['data'][task][self._pressure_map][:, self._level_map]
                    date = self._ds['pressure']['time'][task]
                    p_vars = np.reshape(p_vars, [-1, p_vars.shape[2], p_vars.shape[3]])

                    if self._mode == 'SP':
                        data_arr = np.concatenate([s_vars, p_vars], axis=0).astype(np.float32)
                    elif self._mode == 'PS':
                        data_arr = np.concatenate([p_vars, s_vars], axis=0).astype(np.float32)
                    
                    while True:
                        self._current_lock.acquire()
                        if task == self._current_idx and len(self._buffer) < self._buff_len:
                            with self._buffer_lock:
                                self._buffer.append((data_arr, date))
                                self._current_idx += 1
                                self._current_lock.release()
                                break
                        else:
                            self._current_lock.release()
                            time.sleep(0.05)
                else:
                    break
            with self._running_lock:
                self._running -= 1
        except Exception as e:
            print(f'Dataloader 错误: {str(e)}')

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            with self._buffer_lock:
                if len(self._buffer) != 0:
                    data_arr, date = self._buffer.pop(0)
                    return data_arr, date
            with self._running_lock:
                if self._running <= 0: raise StopIteration
            time.sleep(0.1)