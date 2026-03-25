#!/bin/bash
# =============================================================
# 功能二：滚动推理（含可选定量评估）Slurm 提交脚本
#
# 默认行为（如无参数覆盖）：
#   - 全部启用模型
#   - gundong_20260324 数据源
#   - 20260308 单日，12h 起报，6h步长，240h
#   - 全部地表变量
#   - 不开启评估（需要则设 ENABLE_EVAL=1）
#
# 用法示例：
#   sbatch scripts/submit_rolling.sh
#
#   # 自定义参数
#   MODELS="fengwu fuxi" DATE_RANGE="20260301:20260318" \
#   ENABLE_EVAL=1 SAVE_DIFF=1 \
#   sbatch scripts/submit_rolling.sh
#
#   # 多卡并行（通过 torchrun 或 srun --ntasks 分片日期）
#   WORLD_SIZE=8 sbatch --ntasks=8 scripts/submit_rolling.sh
# =============================================================
#SBATCH -J zk_rolling
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH -o logs/rolling_%j.out
#SBATCH -e logs/rolling_%j.err

set -euo pipefail

# ---- 工作目录 ----
WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
ZK_ROOT="${WORKDIR}/ZK_Models"
LOG_DIR="${ZK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---- 可配置参数 ----
MODELS="${MODELS:-all}"
DATA_SOURCE="${DATA_SOURCE:-gundong_20260324}"
DATE_RANGE="${DATE_RANGE:-20260308}"
INIT_HOUR="${INIT_HOUR:-12}"
LEAD_STEP="${LEAD_STEP:-6}"
MAX_LEAD="${MAX_LEAD:-240}"
VARIABLES="${VARIABLES:-}"                # 空=模型默认全地表变量
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/share/aciwgvx1jd/GunDong_Infer_result_12h}"
DEVICE="${DEVICE:-auto}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
SAVE_NC="${SAVE_NC:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"           # 1=开启内嵌评估
SAVE_DIFF="${SAVE_DIFF:-0}"              # 1=保存 diff npy（需 ENABLE_EVAL=1）
SAVE_DIFF_NC="${SAVE_DIFF_NC:-0}"        # 1=保存 diff nc（需 ENABLE_EVAL=1）
METRICS="${METRICS:-W-MAE W-RMSE}"

# ---- 环境 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] models=${MODELS}"
echo "[info] data_source=${DATA_SOURCE}"
echo "[info] date_range=${DATE_RANGE}"
echo "[info] init_hour=${INIT_HOUR}  lead_step=${LEAD_STEP}  max_lead=${MAX_LEAD}"
echo "[info] output=${OUTPUT_ROOT}"
echo "[info] enable_eval=${ENABLE_EVAL}"
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

# ---- 多卡分片：若需要多进程并行日期，使用 torchrun ----
# 默认单进程；若 WORLD_SIZE>1 则用 torchrun
WORLD_SIZE="${WORLD_SIZE:-1}"

# ---- 环境校验 ----
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true

cd "${WORKDIR}"

# ---- 构建参数 ----
ARGS=()
ARGS+=(--data-source "${DATA_SOURCE}")
ARGS+=(--date-range "${DATE_RANGE}")
ARGS+=(--init-hour "${INIT_HOUR}")
ARGS+=(--lead-step "${LEAD_STEP}")
ARGS+=(--max-lead "${MAX_LEAD}")
ARGS+=(--device "${DEVICE}")
ARGS+=(--output-root "${OUTPUT_ROOT}")
ARGS+=(--metrics ${METRICS})

if [ "${MODELS}" = "all" ]; then
    ARGS+=(--models all)
else
    ARGS+=(--models ${MODELS})
fi

if [ -n "${VARIABLES}" ]; then
    ARGS+=(--variables ${VARIABLES})
fi
if [ "${SKIP_PLOTS}" = "1" ]; then
    ARGS+=(--skip-plots)
fi
if [ "${SAVE_NC}" = "1" ]; then
    ARGS+=(--save-nc)
fi
if [ "${ENABLE_EVAL}" = "1" ]; then
    ARGS+=(--enable-eval)
fi
if [ "${SAVE_DIFF}" = "1" ]; then
    ARGS+=(--save-diff)
fi
if [ "${SAVE_DIFF_NC}" = "1" ]; then
    ARGS+=(--save-diff-nc)
fi

echo "[info] 开始时间: $(date)"

if [ "${WORLD_SIZE}" -gt "1" ]; then
    echo "[info] 多进程模式: WORLD_SIZE=${WORLD_SIZE}"
    torchrun \
        --nproc_per_node="${WORLD_SIZE}" \
        --master_port=29500 \
        ZK_Models/run_rolling.py "${ARGS[@]}"
else
    python ZK_Models/run_rolling.py "${ARGS[@]}"
fi

echo "[info] 完成时间: $(date)"
