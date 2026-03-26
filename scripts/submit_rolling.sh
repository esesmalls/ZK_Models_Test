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
# 仅基于已有 NPY 重跑合并评估（无 ONNX），在计算节点手动执行示例：
#   cd /public/home/.../graphcast
#   conda activate torch2.4_dtk25.04_cp310_e2s
#   python ZK_Models/run_eval_npy.py --data-source gundong_20260324 \\
#     --date-range 20260310 --init-hour 12 --max-lead 240 \\
#     --models pangu fengwu fuxi graphcast --output-root /path/to/GunDong_Infer_result_12h
#
# 用法示例：
#   sbatch scripts/submit_rolling.sh
#
#   # 自定义参数
#   MODELS="fengwu fuxi" DATE_RANGE="20260301:20260318" \
#   ENABLE_EVAL=1 SAVE_DIFF=1 \
#   sbatch scripts/submit_rolling.sh
#
#   # 指定不同 conda 环境
#   CONDA_ENV=my_env DTK_VERSION=25.04 sbatch scripts/submit_rolling.sh
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

# ---- 多卡并行策略 ----
# WORLD_SIZE:      并行进程数（默认 1=单进程）
# PARALLEL_MODE:   auto | date | model
#   auto  — 日期数 >= WORLD_SIZE 时按日期分片，否则按模型分片
#   date  — 多日任务：18天 × WORLD_SIZE=8 → 每卡 2~3 天
#   model — 单日任务：1天 × 4模型 × WORLD_SIZE=4 → 每卡 1 个模型
WORLD_SIZE="${WORLD_SIZE:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-auto}"

# ---- DCU / CPU 说明（Slurm 配额）----
# 逐模型串行 + WORLD_SIZE=1 时，同一时间只有 1 张 DCU 在跑推理，其余卡空闲属正常；
# rocm-smi 固定间隔采样也可能在步间采到 0% 利用率。
# 多卡并行：WORLD_SIZE>1 且 PARALLEL_MODE=model（按模型分卡）或 date（按日期分卡）时，
# 须同时降低 #SBATCH --cpus-per-task 或提高节点 CPU 配额，使
#   ntasks * cpus-per-task <= 节点可用核数
# 例如 8 进程时可设：  sbatch --ntasks-per-node=8 --cpus-per-task=8 ...

# ---- 打印运行信息 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] conda_env=${CONDA_ENV}"
echo "[info] dtk_version=${DTK_VERSION:-none}"
echo "[info] models=${MODELS}"
echo "[info] data_source=${DATA_SOURCE}"
echo "[info] date_range=${DATE_RANGE}"
echo "[info] init_hour=${INIT_HOUR}  lead_step=${LEAD_STEP}  max_lead=${MAX_LEAD}"
echo "[info] output=${OUTPUT_ROOT}"
echo "[info] enable_eval=${ENABLE_EVAL}"
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
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    if [ -n "${DTK_VERSION}" ]; then
        module load "compiler/dtk/${DTK_VERSION}"
        echo "[info] 已加载 compiler/dtk/${DTK_VERSION}"
    else
        echo "[warn] DTK_VERSION 为空，跳过 module load"
    fi
    export LD_LIBRARY_PATH=${ROCM_PATH:+$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:}${LD_LIBRARY_PATH:-}
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export HSA_ENABLE_SDMA=0
    export HSA_ENABLE_SDMA_GANG=0
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export DGL_GRAPHBOLT=0
export DGL_USE_GRAPHBOLT=0
export DGL_LOAD_GRAPHBOLT=0
export OMP_NUM_THREADS=16
unset PYTHONPATH || true

# ---- 环境校验 ----
echo "[info] Python: $(python --version 2>&1)"
python -c "import torch; print('[info] torch', torch.__version__, '| cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('[info] ORT providers:', ort.get_available_providers())" || true

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

ARGS+=(--parallel-mode "${PARALLEL_MODE}")

echo "[info] parallel_mode=${PARALLEL_MODE}  world_size=${WORLD_SIZE}"
echo "[info] 开始时间: $(date)"

if [ "${WORLD_SIZE}" -gt "1" ]; then
    echo "[info] 多进程模式: WORLD_SIZE=${WORLD_SIZE}  PARALLEL_MODE=${PARALLEL_MODE}"
    torchrun \
        --nproc_per_node="${WORLD_SIZE}" \
        --master_port=29500 \
        ZK_Models/run_rolling.py "${ARGS[@]}"
else
    python ZK_Models/run_rolling.py "${ARGS[@]}"
fi

echo "[info] 完成时间: $(date)"
