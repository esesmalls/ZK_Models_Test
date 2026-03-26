"""
GraphCast PyTorch 模型封装。

依赖 onescience 包中的 GraphCastNet、YParams、StaticData 等。
每步推理 cfg.dt 小时（通常 6h）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel
from ..data.channel_mapper import (
    blob_to_graphcast_norm,
    graphcast_norm_to_blob,
)

_ZK_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_ROOT = _ZK_ROOT.parent
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))


class GraphCastModel(WeatherModel):
    MODEL_NAME = "graphcast"

    def __init__(self):
        self._model = None
        self._cfg = None
        self._mu: Optional[np.ndarray] = None
        self._sd: Optional[np.ndarray] = None
        self._static_data = None
        self._latlon_torch = None
        self._device = None
        self._channels: List[str] = []
        self._step_h = 6
        self._loaded = False

    def load(self, cfg: Dict, device: Any = "cuda") -> None:
        import torch
        from ruamel.yaml.scalarfloat import ScalarFloat
        from onescience.models.graphcast.graph_cast_net import GraphCastNet
        from onescience.utils.fcn.YParams import YParams
        from onescience.utils.graphcast.data_utils import StaticData
        from onescience.datapipes.climate.utils.invariant import latlon_grid

        torch.serialization.add_safe_globals([ScalarFloat])

        model_config = cfg.get("model_config", str(_GRAPH_ROOT / "conf/config.yaml"))
        model_config_key = cfg.get("model_config_key", "model")
        model_cfg = YParams(str(model_config), model_config_key)

        # 加载 metadata，获取 channel 列表和索引
        metadata_json = cfg.get("metadata_json")
        if metadata_json:
            variables = json.load(open(metadata_json, encoding="utf-8"))["variables"]
            channel_indices = [variables.index(c) for c in model_cfg.channels]
        else:
            channel_indices = list(range(len(model_cfg.channels)))

        stats_dir = Path(cfg.get("stats_dir", str(_GRAPH_ROOT / "graphcast_model/stats")))
        mu_full = np.load(stats_dir / "global_means.npy")
        sd_full = np.load(stats_dir / "global_stds.npy")
        self._mu = mu_full[0, channel_indices, 0, 0].astype(np.float32)
        self._sd = sd_full[0, channel_indices, 0, 0].astype(np.float32)

        # 构建模型
        if isinstance(device, str):
            device = torch.device(device if device != "auto" else
                                  ("cuda" if torch.cuda.is_available() else "cpu"))
        self._device = device
        self._cfg = model_cfg
        self._channels = list(model_cfg.channels)
        self._step_h = int(model_cfg.dt)

        input_dim = (
            (len(model_cfg.channels) + model_cfg.use_cos_zenith + 4 * model_cfg.use_time_of_year_index)
            * (model_cfg.num_history + 1)
            + model_cfg.num_channels_static
        )
        model = GraphCastNet(
            mesh_level=model_cfg.mesh_level,
            multimesh=model_cfg.multimesh,
            input_res=tuple(model_cfg.img_size),
            input_dim_grid_nodes=input_dim,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=len(model_cfg.channels),
            processor_type=model_cfg.processor_type,
            khop_neighbors=model_cfg.khop_neighbors,
            num_attention_heads=model_cfg.num_attention_heads,
            processor_layers=model_cfg.processor_layers,
            hidden_dim=model_cfg.hidden_dim,
            norm_type=model_cfg.norm_type,
            do_concat_trick=model_cfg.concat_trick,
            recompute_activation=model_cfg.recompute_activation,
        )
        model.set_checkpoint_encoder(model_cfg.checkpoint_encoder)
        model.set_checkpoint_decoder(model_cfg.checkpoint_decoder)

        ckpt_path = Path(cfg.get("checkpoint", str(_GRAPH_ROOT / "graphcast_model/graphcast_finetune.pth")))
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dtype=torch.float32).to(device)
        model.eval()
        self._model = model

        static_dir = cfg.get("static_dir", str(_GRAPH_ROOT / "graphcast_model/static"))
        self._static_data = StaticData(static_dir, model.latitudes, model.longitudes).get().to(
            device=device, dtype=torch.float32
        )

        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=model_cfg.img_size[-2:])
        import torch as _torch
        self._latlon_torch = _torch.tensor(
            np.stack(latlon, axis=0), dtype=_torch.float32
        )
        self._loaded = True
        print(
            f"[GraphCast] device={self._device} torch.cuda.is_available()={torch.cuda.is_available()}",
            flush=True,
        )
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                print(
                    f"[GraphCast] cuda.current_device={idx} name={torch.cuda.get_device_name(idx)}",
                    flush=True,
                )
            except Exception as ex:
                print(f"[GraphCast] cuda device info: {ex}", flush=True)

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Optional[datetime] = None,
    ) -> ModelState:
        import torch
        norm = blob_to_graphcast_norm(init_blob, self._channels, self._mu, self._sd)
        state_t = torch.from_numpy(norm).unsqueeze(0).to(device=self._device, dtype=torch.float32)
        return ModelState(
            data={"state": state_t},
            blob=init_blob,
            lead=0,
            extra={"forecast_time": init_dt or datetime.utcnow()},
        )

    def step(self, state: ModelState) -> ModelState:
        import torch
        from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle

        cfg = self._cfg
        ft: datetime = state.extra["forecast_time"]
        ts = torch.tensor([ft.timestamp()], dtype=torch.float32, device=self._device)
        cz = cos_zenith_angle(ts, latlon=self._latlon_torch.to(self._device)).float()
        cz = torch.squeeze(cz, dim=2)
        cz = torch.clamp(cz, min=0.0) - 1.0 / torch.pi

        doy = float(ft.timetuple().tm_yday)
        tod = ft.hour + ft.minute / 60.0 + ft.second / 3600.0
        ndy = torch.tensor((doy / 365.0) * (np.pi / 2), dtype=torch.float32, device=self._device)
        ntd = torch.tensor((tod / (24.0 - cfg.dt)) * (np.pi / 2), dtype=torch.float32, device=self._device)
        H, W = cfg.img_size[0], cfg.img_size[1]
        sin_dy = torch.sin(ndy).expand(1, 1, H, W)
        cos_dy = torch.cos(ndy).expand(1, 1, H, W)
        sin_td = torch.sin(ntd).expand(1, 1, H, W)
        cos_td = torch.cos(ntd).expand(1, 1, H, W)

        invar = torch.cat(
            (state.data["state"], cz, self._static_data, sin_dy, cos_dy, sin_td, cos_td), dim=1
        )

        with torch.no_grad():
            pred = self._model(invar)

        arr = pred.float().cpu().numpy()[0]  # (C, H, W)
        blob_new = graphcast_norm_to_blob(arr, self._channels, self._mu, self._sd)
        blob_new["lat"] = state.blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        blob_new["lon"] = state.blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))

        return ModelState(
            data={"state": pred},
            blob=blob_new,
            lead=state.lead + self._step_h,
            extra={"forecast_time": ft + timedelta(hours=self._step_h)},
        )

    def unload(self) -> None:
        """将 PyTorch 模型移回 CPU 并删除，释放 GPU 显存。"""
        import gc
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None
        self._static_data = None
        self._latlon_torch = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def get_surface_var_names(self) -> List[str]:
        mapping = {
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
            "mean_sea_level_pressure": "msl",
        }
        return [mapping[c] for c in self._channels if c in mapping]

    def get_pressure_var_names(self) -> List[str]:
        return ["z", "q", "t", "u", "v"]

    def get_step_hours(self) -> int:
        return self._step_h
