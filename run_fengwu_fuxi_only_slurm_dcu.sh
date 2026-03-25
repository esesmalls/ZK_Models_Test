#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH -J fw_fuxi_only
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_fw_fuxi.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_fw_fuxi.err
#
# 仅 Fengwu + Fuxi（跳过 GraphCast/Pangu）。时间语义见 run_four_models_test_era5 文档字符串。
# 追加参数示例：
#   sbatch run_fengwu_fuxi_only_slurm_dcu.sh --date 20260301 --hour 0 --num-steps 2

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
export OMP_NUM_THREADS=8
export ORT_INTRA_OP_NUM_THREADS=${ORT_INTRA_OP_NUM_THREADS:-4}

cd "${GRAPHCAST_ROOT}"
python ZK_Models/run_four_models_test_era5.py \
  --device dcu \
  --only-models fengwu,fuxi \
  "$@"
echo "END TIME: $(date)"
