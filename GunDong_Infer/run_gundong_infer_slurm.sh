#!/bin/bash
#SBATCH -J gundong_240h
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH -o logs/gundong_240h_%j.out
#SBATCH -e logs/gundong_240h_%j.err

set -euo pipefail

WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
SCRIPT_DIR="${WORKDIR}/ZK_Models/GunDong_Infer"
PY_SCRIPT="${SCRIPT_DIR}/run_gundong_infer.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

INPUT_ROOT="${INPUT_ROOT:-/public/share/aciwgvx1jd/20260324}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/share/aciwgvx1jd/GunDong_Infer_result}"
MAX_LEAD="${MAX_LEAD:-240}"
LEAD_STEP="${LEAD_STEP:-6}"
START_HOUR="${START_HOUR:-0}"
DEVICE="${DEVICE:-auto}"
ONLY_MODELS="${ONLY_MODELS:-pangu,graphcast}"
DATE_FILTER="${DATE_FILTER:-}"

echo "[info] job=${SLURM_JOB_ID:-na}"
echo "[info] input=${INPUT_ROOT}"
echo "[info] output=${OUTPUT_ROOT}"
echo "[info] models=${ONLY_MODELS}"

if [ -f /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh ]; then
  source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
fi
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
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true

cd "${WORKDIR}"

NPROC=${NPROC_PER_NODE:-8}
srun --nodes=1 --ntasks=1 torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${PY_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --start-hour "${START_HOUR}" \
  --max-lead-hours "${MAX_LEAD}" \
  --lead-step-hours "${LEAD_STEP}" \
  --device "${DEVICE}" \
  --only-models "${ONLY_MODELS}" \
  --date-filter "${DATE_FILTER}"

