#!/bin/bash
# =============================================================
# 独立评估：对已有 NPY 文件计算 W-RMSE/W-MAE Slurm 提交脚本
#
# 适用场景：
#   1. 对历史 NPY 存档补充评估
#   2. run_rolling.py 未开启 --enable-eval 时事后补算
#
# 用法：
#   TIME_TAG=20260308T12 sbatch scripts/submit_evaluate.sh
#
#   # 自定义参数
#   TIME_TAG=20260308T12 MODELS="FengWu FuXi" SAVE_DIFF=1 \
#   sbatch scripts/submit_evaluate.sh
# =============================================================
#SBATCH -J zk_evaluate
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:1
#SBATCH -o logs/evaluate_%j.out
#SBATCH -e logs/evaluate_%j.err

set -euo pipefail

WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
ZK_ROOT="${WORKDIR}/ZK_Models"
LOG_DIR="${ZK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---- 可配置参数 ----
TIME_TAG="${TIME_TAG:-20260308T12}"
MODELS="${MODELS:-FengWu GraphCast FuXi PanGu}"
VARIABLES="${VARIABLES:-u10 v10 t2m}"
PRED_BASE_DIR="${PRED_BASE_DIR:-/public/share/aciwgvx1jd/GunDong_Infer_result_12h}"
ERA5_DIR="${ERA5_DIR:-/public/share/aciwgvx1jd/20260324/surface}"
STEP_INTERVAL="${STEP_INTERVAL:-6}"
EXPECTED_STEPS="${EXPECTED_STEPS:-40}"
METRICS="${METRICS:-W-MAE W-RMSE}"
SAVE_DIFF="${SAVE_DIFF:-0}"
SAVE_DIFF_NC="${SAVE_DIFF_NC:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"              # 空=使用默认路径

# ---- 环境 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] time_tag=${TIME_TAG}"
echo "[info] models=${MODELS}"
echo "[info] variables=${VARIABLES}"
echo "=========================================="

if [ -f /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh ]; then
    source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
fi
conda activate torch2.4_dtk25.04_cp310_e2s

module purge
module load compiler/dtk/25.04
export OMP_NUM_THREADS=16
unset PYTHONPATH || true

cd "${WORKDIR}"

# ---- 构建参数 ----
ARGS=()
ARGS+=(--time-tag "${TIME_TAG}")
ARGS+=(--models ${MODELS})
ARGS+=(--variables ${VARIABLES})
ARGS+=(--pred-base-dir "${PRED_BASE_DIR}")
ARGS+=(--era5-dir "${ERA5_DIR}")
ARGS+=(--step-interval "${STEP_INTERVAL}")
ARGS+=(--expected-steps "${EXPECTED_STEPS}")
ARGS+=(--metrics ${METRICS})

if [ "${SAVE_DIFF}" = "1" ]; then
    ARGS+=(--save-diff)
fi
if [ "${SAVE_DIFF_NC}" = "1" ]; then
    ARGS+=(--save-diff-nc)
fi
if [ -n "${OUTPUT_DIR}" ]; then
    ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

echo "[info] 开始时间: $(date)"
python ZK_Models/run_evaluate.py "${ARGS[@]}"
echo "[info] 完成时间: $(date)"
