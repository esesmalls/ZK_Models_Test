#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
# 申请 8 卡；当前 Python 脚本仍以单进程为主（GraphCast 用 cuda:0，ONNX 多为单 EP）。多卡需后续 torchrun/并行改造。
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:8
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH -J four_models_era5
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_four_models.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_four_models.err
#
# GraphCast + Pangu + Fengwu + Fuxi，使用 graphcast/test_era5_data 多步推理并写 PNG。
#
# 提交（建议在 ZK_Models 目录，与 zk_infer 脚本一致）:
#   cd .../graphcast/ZK_Models && sbatch run_four_models_test_era5_slurm_dcu.sh
#   sbatch run_four_models_test_era5_slurm_dcu.sh -- --num-steps 4 --date 20260301 --hour 0
#
# 默认参数在下方 EXTRA_ARGS；命令行追加会传给 python（需 -- 分隔 sbatch 与脚本选项时见示例）。

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

# 计算节点上执行的是 spool 里的脚本副本，勿仅用 BASH_SOURCE 定位目录
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

which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true

cd "${GRAPHCAST_ROOT}"
# 默认：DCU 上 GraphCast 用 HIP；ONNX 用 pick_providers(dcu)
EXTRA_ARGS=(--device dcu --num-steps 3)
python ZK_Models/run_four_models_test_era5.py "${EXTRA_ARGS[@]}" "$@"

echo "END TIME: $(date)"
