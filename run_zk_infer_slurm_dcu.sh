#!/bin/bash
#SBATCH -p kshkexclu01
#SBATCH -N 1
# 本集群 kshkexclu01 上实测需至少申请 2 卡 DCU 才能通过调度（单卡会报 configuration not available）
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH -J zk_infer_dcu
# 使用绝对路径，避免在计算节点工作目录下无法创建 logs/ 导致作业立刻失败
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_zk_infer.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/logs/%j_zk_infer.err
#
# ZK_Models ONNX 推理 — DCU 计算节点（对齐 graphcast/run_inference_slurm_1gpu.sh）
#
# 日志写入本目录下 logs/（与 #SBATCH -o logs/... 一致）
#
# 务必在 ZK_Models 目录下提交: cd .../graphcast/ZK_Models && sbatch run_zk_infer_slurm_dcu.sh ...
#
# 用法:
#   sbatch run_zk_infer_slurm_dcu.sh pangu
#   sbatch run_zk_infer_slurm_dcu.sh fengwu
#   sbatch run_zk_infer_slurm_dcu.sh fuxi_short
#   sbatch run_zk_infer_slurm_dcu.sh pangu --date 20200102 --hour 12 --num-steps 6
#
# 环境: conda activate torch2.4_dtk25.04_cp310_e2s
# onnxruntime 需带 ROCm/MIGraphX EP 才能在 DCU 上加速；否则脚本仍会用 --device dcu 并回退 CPU。

set -Eeuo pipefail

echo "START TIME: $(date)"
echo "JOB_ID=${SLURM_JOB_ID:-}"

# Slurm 在计算节点执行的是 spool 里的脚本副本，勿用 BASH_SOURCE 定位工程目录
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

MODEL="${1:-pangu}"
shift || true

# 部分集群上 srun 会错误解析相对路径日志目录；批处理内直接执行即可
python infer_cepri_onnx.py --model "${MODEL}" --device dcu "$@"

echo "END TIME: $(date)"
