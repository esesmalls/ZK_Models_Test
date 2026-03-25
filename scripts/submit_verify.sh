#!/bin/bash
# =============================================================
# 功能一：初步推理验证 Slurm 提交脚本
# 用法：
#   sbatch scripts/submit_verify.sh
#   或覆盖变量后提交：
#   MODELS="pangu fengwu" DATE=20260308 sbatch scripts/submit_verify.sh
# =============================================================
#SBATCH -J zk_verify
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:8
#SBATCH -o logs/verify_%j.out
#SBATCH -e logs/verify_%j.err

set -euo pipefail

# ---- 工作目录 ----
WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
ZK_ROOT="${WORKDIR}/ZK_Models"
LOG_DIR="${ZK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---- 可配置参数（通过环境变量覆盖）----
MODELS="${MODELS:-pangu fengwu fuxi graphcast}"   # 空格分隔 或 "all"
DATA_SOURCE="${DATA_SOURCE:-test_era5}"
DATE="${DATE:-20260308}"
HOUR="${HOUR:-12}"
NUM_STEPS="${NUM_STEPS:-2}"
VARIABLES="${VARIABLES:-}"                        # 空=模型默认全地表变量
ALL_SURFACE="${ALL_SURFACE:-1}"                   # 1=--all-surface
PRESSURE_VARS="${PRESSURE_VARS:-z:1000 t:1000}"   # 气压层变量（空=不出气压图）
DEVICE="${DEVICE:-auto}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ZK_ROOT}/results/verify}"

# ---- 环境 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] models=${MODELS}"
echo "[info] data_source=${DATA_SOURCE}"
echo "[info] date=${DATE}  hour=${HOUR}"
echo "[info] num_steps=${NUM_STEPS}"
echo "[info] output=${OUTPUT_ROOT}"
echo "=========================================="

if [ -f /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
fi
conda activate torch2.4_dtk25.04_cp310_e2s

module purge
module load compiler/dtk/25.04
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_SDMA_GANG=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export DGL_GRAPHBOLT=0
export DGL_USE_GRAPHBOLT=0
export DGL_LOAD_GRAPHBOLT=0
export OMP_NUM_THREADS=16
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH || true

# ---- 环境校验 ----
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true

cd "${WORKDIR}"

# ---- 构建参数 ----
ARGS=()
ARGS+=(--data-source "${DATA_SOURCE}")
ARGS+=(--date "${DATE}")
ARGS+=(--hour "${HOUR}")
ARGS+=(--num-steps "${NUM_STEPS}")
ARGS+=(--device "${DEVICE}")
ARGS+=(--output-root "${OUTPUT_ROOT}")

# 模型列表
if [ "${MODELS}" = "all" ]; then
    ARGS+=(--models all)
else
    ARGS+=(--models ${MODELS})
fi

# 变量
if [ "${ALL_SURFACE}" = "1" ]; then
    ARGS+=(--all-surface)
elif [ -n "${VARIABLES}" ]; then
    ARGS+=(--variables ${VARIABLES})
fi

# 气压层
if [ -n "${PRESSURE_VARS}" ]; then
    ARGS+=(--pressure-vars ${PRESSURE_VARS})
fi

if [ "${SKIP_PLOTS}" = "1" ]; then
    ARGS+=(--skip-plots)
fi

echo "[info] CMD: python ZK_Models/run_verify.py ${ARGS[*]}"
echo "[info] 开始时间: $(date)"

python ZK_Models/run_verify.py "${ARGS[@]}"

echo "[info] 完成时间: $(date)"
