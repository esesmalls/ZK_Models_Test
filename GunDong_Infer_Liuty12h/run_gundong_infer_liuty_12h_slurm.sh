#!/bin/bash
#SBATCH -J gundong_liuty_12h
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH -o logs/gundong_liuty_12h_%j.out
#SBATCH -e logs/gundong_liuty_12h_%j.err

set -euo pipefail

WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
SCRIPT_DIR="${WORKDIR}/ZK_Models/GunDong_Infer_Liuty12h"
PY_SCRIPT="${SCRIPT_DIR}/run_gundong_infer_liuty_12h.py"

mkdir -p "${SCRIPT_DIR}/logs"

INPUT_ROOT="${INPUT_ROOT:-/public/share/aciwgvx1jd/20260324}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/share/aciwgvx1jd/GunDong_Infer_result_12h}"
START_DATETIME="${START_DATETIME:-20260308T12}"
MAX_LEAD_HOURS="${MAX_LEAD_HOURS:-240}"
ONLY_MODELS="${ONLY_MODELS:-pangu,graphcast}"
SURFACE_ONLY="${SURFACE_ONLY:-1}"

NPROC=${NPROC_PER_NODE:-8}

echo "[info] job=${SLURM_JOB_ID:-na}"
echo "[info] input=${INPUT_ROOT}"
echo "[info] output=${OUTPUT_ROOT}"
echo "[info] start=${START_DATETIME}"
echo "[info] max_lead_hours=${MAX_LEAD_HOURS}"
echo "[info] only_models=${ONLY_MODELS}"

source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
conda activate torch2.4_dtk25.04_cp310_e2s

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

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"

# Avoid accidentally importing a pre-unpacked onnxruntime ROCm site
# (it may be built against different glibc versions).
unset PYTHONPATH || true
unset ORT_ROCM_SITE || true
export PYTHONNOUSERSITE=1

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__); print('providers', ort.get_available_providers())" || true

cd "${WORKDIR}"

# Extra CLI args.
EXTRA_ARGS=()
if [ "${SURFACE_ONLY}" = "1" ]; then
  EXTRA_ARGS+=(--surface-only)
fi

srun --nodes=1 --ntasks=1 torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${PY_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --start-datetime "${START_DATETIME}" \
  --max-lead-hours "${MAX_LEAD_HOURS}" \
  --only-models "${ONLY_MODELS}" \
  "${EXTRA_ARGS[@]}"

