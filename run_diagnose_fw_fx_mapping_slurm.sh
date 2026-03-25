#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -J diag_map_fwfx
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_diag_map_fwfx.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_diag_map_fwfx.err

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
export ORT_LOG_SEVERITY_LEVEL=3
export OMP_NUM_THREADS=4

cd "${GRAPHCAST_ROOT}"
python ZK_Models/diagnose_fw_fx_mapping.py --device dcu --report "ZK_Models/logs/${SLURM_JOB_ID}_fw_fx_mapping_report.md" "$@"

echo "END TIME: $(date)"
