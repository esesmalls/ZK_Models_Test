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
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true

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
