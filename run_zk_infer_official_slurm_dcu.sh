#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J zk_official_onnx
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_zk_official_onnx.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_zk_official_onnx.err
#
# Slurm launcher for infer_cepri_onnx.py (FengWu default v1 + ZK_Models/fengwu stats path below).
# Usage examples:
#   sbatch run_zk_infer_official_slurm_dcu.sh fuxi_short --date 20200101 --hour 0 --num-steps 2
#   sbatch run_zk_infer_official_slurm_dcu.sh fengwu --date 20200101 --hour 0 --num-steps 2 --fengwu-model-version v1
# onnxruntime 需带 ROCm/MIGraphX EP 才能在 DCU 上加速；默认缺失 EP 时直接报错退出。
# 如需临时允许回退 CPU：设置 ALLOW_CPU_FALLBACK=1，并在参数追加 --allow-cpu-fallback
# 若有解压版 onnxruntime-rocm 目录，可设置 ORT_ROCM_SITE 自动注入 PYTHONPATH。

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
mkdir -p "${SCRIPT_DIR}/logs"

source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
conda activate torch2.4_dtk25.04_cp310_e2s

# Optional: use unpacked onnxruntime-rocm site dir without replacing whole conda env.
if [ -n "${ORT_ROCM_SITE:-}" ] && [ -d "${ORT_ROCM_SITE}" ]; then
  export PYTHONPATH="${ORT_ROCM_SITE}:${PYTHONPATH:-}"
  echo "[env] prepend ORT_ROCM_SITE to PYTHONPATH: ${ORT_ROCM_SITE}"
fi

module purge
module load compiler/dtk/25.04
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_SDMA_GANG=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export DGL_GRAPHBOLT=0
export DGL_USE_GRAPHBOLT=0
export DGL_LOAD_GRAPHBOLT=0

export OMP_NUM_THREADS=16
export HIP_VISIBLE_DEVICES=0

which python
python - <<'PY'
import os
import sys

try:
  import onnxruntime as ort
except Exception as exc:
  print(f"[FATAL] import onnxruntime failed: {exc}", file=sys.stderr)
  sys.exit(2)

providers = ort.get_available_providers()
print("ORT providers:", providers)
allow_cpu = os.environ.get("ALLOW_CPU_FALLBACK", "0") == "1"
if not any(p in providers for p in ("ROCMExecutionProvider", "MIGraphXExecutionProvider")):
  msg = (
    "[FATAL] 当前 onnxruntime 无 ROCM/MIGraphX EP。"
    "请安装/注入 onnxruntime-rocm（可设置 ORT_ROCM_SITE），"
    "或改走 PyTorch 路径。"
  )
  if allow_cpu:
    print(msg + " 已设置 ALLOW_CPU_FALLBACK=1，允许继续（后续可传 --allow-cpu-fallback）。", file=sys.stderr)
  else:
    print(msg + " 如需临时回退 CPU，请设置 ALLOW_CPU_FALLBACK=1 并在命令追加 --allow-cpu-fallback。", file=sys.stderr)
    sys.exit(3)
PY

cd "${SCRIPT_DIR}"

MODEL="${1:-fuxi_short}"
shift || true

# Model-specific defaults (paths / v1 FengWu).
if [ "${MODEL}" = "fengwu" ]; then
  python infer_cepri_onnx.py \
    --model fengwu \
    --device dcu \
    --era5-root /public/share/aciwgvx1jd/CEPRI_ERA5 \
    --fengwu-model-version v1 \
    --fengwu-stats-dir "${SCRIPT_DIR}/fengwu" \
    "$@"
elif [ "${MODEL}" = "fuxi_short" ] || [ "${MODEL}" = "fuxi_medium" ]; then
  python infer_cepri_onnx.py \
    --model "${MODEL}" \
    --device dcu \
    --era5-root /public/share/aciwgvx1jd/CEPRI_ERA5 \
    --stats-dir /public/home/aciwgvx1jd/newh5/stats \
    "$@"
else
  python infer_cepri_onnx.py --model "${MODEL}" --device dcu "$@"
fi

echo "END TIME: $(date)"
