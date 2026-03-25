#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J pangu_two_interps
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_pangu_two_interps.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_pangu_two_interps.err

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
else
  SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
ZK_DIR="${SUBMIT_DIR}/ZK_Models"
mkdir -p "${ZK_DIR}/logs"

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

cd "${ZK_DIR}"
python run_pangu_two_interps.py

echo "END TIME: $(date)"
