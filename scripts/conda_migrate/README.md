# Conda Migration Scripts

这套脚本用于执行计划中的 `conda-migrate`：
- 导出并打包可运行 Conda 环境
- 在新服务器恢复环境并做基础可用性检查
- 按最小命令链完成迁移验收

## 1) 源服务器：导出+打包环境

在 `graphcast` 根目录执行：

```bash
bash ZK_Models/scripts/conda_migrate/export_pack_conda_env.sh \
  --env-name torch2.4_dtk25.04_cp310_e2s \
  --with-explicit-lock
```

默认产物目录：`ZK_Models/conda_migration_artifacts`
- `environment.<env>.<ts>.yml`
- `conda_env.<env>.<ts>.tar.gz`
- `explicit.<env>.<ts>.txt`（可选）
- `export_meta.<env>.<ts>.txt`

将上述产物与代码、模型、测试输入一起同步到新服务器。

## 2) 新服务器：恢复环境+验证依赖

```bash
bash ZK_Models/scripts/conda_migrate/restore_conda_env_new_server.sh \
  --pack /srv/graphcast/artifacts/conda_env.torch2.4_dtk25.04_cp310_e2s.<ts>.tar.gz \
  --prefix /srv/graphcast/conda_env
```

恢复完成后可直接启用：

```bash
source /srv/graphcast/conda_env/bin/activate
```

验证项包括：
- `onnxruntime/torch/netCDF4/xarray/matplotlib/ruamel.yaml` 可导入
- `onnxruntime.get_available_providers()` 输出可见 provider
- `torch.cuda.is_available()` 状态（DCU/ROCm 场景下通常为 True）

## 3) 最小命令链验收

```bash
source /srv/graphcast/conda_env/bin/activate
bash ZK_Models/scripts/conda_migrate/run_minimal_chain.sh \
  --graphcast-root /srv/graphcast/code/examples/earth/graphcast \
  --era5-root /srv/graphcast/data/CEPRI_ERA5 \
  --test-data /srv/graphcast/code/examples/earth/graphcast/test_era5_data \
  --date 20260301 \
  --hour 0 \
  --num-steps 2 \
  --device dcu
```

脚本顺序执行：
1. `infer_cepri_onnx.py --model pangu`
2. `run_pangu_two_interps_noxarray.py`
3. `run_four_models_test_era5.py --only-models fengwu,fuxi`
4. `run_four_models_test_era5.py`（全模型）

输出根目录默认：
- `ZK_Models/results/conda_migrate_<timestamp>`

关键日志：
- `minimal_chain.log`

## 4) 快速预演（不实际执行）

```bash
bash ZK_Models/scripts/conda_migrate/run_minimal_chain.sh --dry-run
```
