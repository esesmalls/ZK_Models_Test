#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:8
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J fw_fx_infer_plot
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_fw_fx_infer_plot.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_fw_fx_infer_plot.err

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ZK_DIR="${SLURM_SUBMIT_DIR}"
else
  ZK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

GRAPHCAST_ROOT="$(cd "${ZK_DIR}/.." && pwd)"
mkdir -p "${ZK_DIR}/logs"

source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
conda activate torch2.4_dtk25.04_cp310_e2s

module purge
module load compiler/dtk/25.04 2>/dev/null || true

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export ORT_INTRA_OP_NUM_THREADS=${ORT_INTRA_OP_NUM_THREADS:-4}
export ORT_INTER_OP_NUM_THREADS=${ORT_INTER_OP_NUM_THREADS:-1}
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

DATE=${DATE:-20260301}
HOUR=${HOUR:-0}
NUM_STEPS=${NUM_STEPS:-4}
TEST_DATA=${TEST_DATA:-${GRAPHCAST_ROOT}/test_era5_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ZK_DIR}/results_fengwu_fuxi}
MODELS=${MODELS:-fengwu_v1,fengwu_v2,fuxi_short,fuxi_medium}
VARS=${VARS:-u10,v10,t2m,msl,z500,t850,q850}
FUXI_STATS_DIR=${FUXI_STATS_DIR:-/public/home/aciwgvx1jd/newh5/stats}
FENGWU_STATS_DIR=${FENGWU_STATS_DIR:-${ZK_DIR}/fengwu}

cd "${GRAPHCAST_ROOT}"

echo "RUN: date=${DATE} hour=${HOUR} steps=${NUM_STEPS} models=${MODELS}"
echo "TEST_DATA=${TEST_DATA}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

python ZK_Models/run_fengwu_fuxi_infer_plot.py \
  --test-data "${TEST_DATA}" \
  --date "${DATE}" \
  --hour "${HOUR}" \
  --num-steps "${NUM_STEPS}" \
  --device dcu \
  --models "${MODELS}" \
  --variables "${VARS}" \
  --fuxi-stats-dir "${FUXI_STATS_DIR}" \
  --fengwu-stats-dir "${FENGWU_STATS_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

echo "END TIME: $(date)"
