#!/bin/bash
#SBATCH -J ort_dcu_test
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:K100-PCIE-64GB-LS:1
#SBATCH -o /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/GunDong_Infer/logs/%j_ort_dcu_test.out
#SBATCH -e /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/GunDong_Infer/logs/%j_ort_dcu_test.err

set -euo pipefail

# ROCm wheel 已解压到登录节点目录（计算节点通过共享 $HOME 读取）
export ORT_ROCM_SITE="${ORT_ROCM_SITE:-/public/home/aciwgvx1jd/opt/onnxruntime_rocm_py310_site}"
export PYTHONPATH="${ORT_ROCM_SITE}:${PYTHONPATH:-}"

source /public/home/aciwgvx1jd/miniconda3/etc/profile.d/conda.sh
conda activate torch2.4_dtk25.04_cp310_e2s

module purge
module load compiler/dtk/25.04
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:$LD_LIBRARY_PATH

echo "ROCM_PATH=$ROCM_PATH"
echo "PYTHONPATH=$PYTHONPATH"
which python
python -c "import sys; print('python', sys.version)"
python - <<'PY'
import os
import onnxruntime as ort
print("onnxruntime", ort.__version__)
print("available_providers", ort.get_available_providers())
PY

ZK="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models"
ONNX6="${ZK}/pangu/pangu_weather_6.onnx"
if [[ -f "$ONNX6" ]]; then
  python - <<PY
import sys
from pathlib import Path

ZK = Path("${ZK}")
sys.path.insert(0, str(ZK))
sys.path.insert(0, str(ZK / "GunDong_Infer"))
import onnxruntime as ort  # noqa: F401
from infer_cepri_onnx import create_session, pick_providers, pangu_one_step
from cepri_loader import pack_pangu_onnx
from data_adapter_20260324 import load_time_blob

prov = pick_providers("dcu")
print("pick_providers(dcu)=", prov)
root = Path("/public/share/aciwgvx1jd/20260324")
blob = load_time_blob(root, "20260301", 0)
p_in, s_in = pack_pangu_onnx(blob)
sess = create_session(ZK / "pangu" / "pangu_weather_6.onnx", prov)
op, os_ = pangu_one_step(sess, p_in, s_in)
print("pangu_one_step ok", op.shape, os_.shape)
PY
else
  echo "skip pangu_one_step: missing $ONNX6"
fi
