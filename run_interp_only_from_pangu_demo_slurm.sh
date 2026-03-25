#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH -J pangu_interp_only
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_pangu_interp_only.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_pangu_interp_only.err

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
ZK_DIR="${ROOT_DIR}/ZK_Models"
mkdir -p "${ZK_DIR}/logs"

source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
conda activate torch2.4_dtk25.04_cp310_e2s

module purge
module load compiler/dtk/25.04
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8

cd "${ROOT_DIR}"
python ZK_Models/run_pangu_two_interps_noxarray.py \
  --raw-dir ZK_Models/results/pangu_2step_demo \
  --num-steps 2 \
  --date 19790101 \
  --hour 0

echo "END TIME: $(date)"
