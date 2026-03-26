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
#
#   # 指定不同 conda 环境
#   CONDA_ENV=my_env DTK_VERSION=25.04 sbatch scripts/submit_evaluate.sh
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
ZK_LEGACY="${WORKDIR}/ZK_Models"
UNIFIED_ROOT="${ZK_LEGACY}/unified_pipeline"
ZK_ROOT="${ZK_LEGACY}"
LOG_DIR="${ZK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ==============================================================
# ---- 环境选择（通过环境变量覆盖）----
# CONDA_ENV:   conda 环境名称，默认 torch2.4_dtk25.04_cp310_e2s
# CONDA_BASE:  miniconda/anaconda 根目录，默认自动探测
# DTK_VERSION: 编译器/DTK 版本，默认 25.04（留空则跳过 module load）
# USE_CUDA:    1 = 使用 CUDA 模式而非 ROCm/DCU（默认 0）
# ==============================================================
CONDA_ENV="${CONDA_ENV:-torch2.4_dtk25.04_cp310_e2s}"
CONDA_BASE="${CONDA_BASE:-}"
DTK_VERSION="${DTK_VERSION:-25.04}"
USE_CUDA="${USE_CUDA:-0}"

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

# ---- 打印运行信息 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] conda_env=${CONDA_ENV}"
echo "[info] dtk_version=${DTK_VERSION:-none}"
echo "[info] time_tag=${TIME_TAG}"
echo "[info] models=${MODELS}"
echo "[info] variables=${VARIABLES}"
echo "=========================================="

# ==============================================================
# ---- 激活 conda 环境 ----
# ==============================================================
_activate_conda() {
    if [ -n "${CONDA_BASE}" ]; then
        local init_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
        if [ -f "${init_sh}" ]; then
            source "${init_sh}"
            return 0
        fi
    fi
    for candidate in \
        "/public/home/aciwgvx1jd/miniconda3" \
        "/public/home/aciwgvx1jd/anaconda3" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "/opt/miniconda3" \
        "/opt/conda"
    do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            source "${candidate}/etc/profile.d/conda.sh"
            CONDA_BASE="${candidate}"
            return 0
        fi
    done
    echo "[warn] 未找到 conda 初始化脚本，尝试直接激活..."
    return 1
}

_activate_conda || true
conda activate "${CONDA_ENV}"
echo "[info] 已激活 conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

# ==============================================================
# ---- 加载 DTK / CUDA 模块 ----
# ==============================================================
module purge
if [ "${USE_CUDA}" = "1" ]; then
    echo "[info] USE_CUDA=1，跳过 DTK module，使用 CUDA 模式"
else
    if [ -n "${DTK_VERSION}" ]; then
        module load "compiler/dtk/${DTK_VERSION}"
        echo "[info] 已加载 compiler/dtk/${DTK_VERSION}"
    else
        echo "[warn] DTK_VERSION 为空，跳过 module load"
    fi
    export LD_LIBRARY_PATH=${ROCM_PATH:+$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:}${LD_LIBRARY_PATH:-}
fi

export OMP_NUM_THREADS=16
unset PYTHONPATH || true

echo "[info] Python: $(python --version 2>&1)"

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
python "${UNIFIED_ROOT}/run_evaluate.py" "${ARGS[@]}"
echo "[info] 完成时间: $(date)"
