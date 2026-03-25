#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal command chain for conda migration acceptance:
# 1) infer_cepri_onnx.py --model pangu
# 2) run_pangu_two_interps_noxarray.py
# 3) run_four_models_test_era5.py --only-models fengwu,fuxi
# 4) run_four_models_test_era5.py (all models)

usage() {
  cat <<'EOF'
Usage: run_minimal_chain.sh [options]

Options:
  --graphcast-root <path>  GraphCast project root (default: script auto-detect)
  --era5-root <path>       CEPRI ERA5 root for ONNX scripts (default: /public/share/aciwgvx1jd/CEPRI_ERA5)
  --test-data <path>       ERA5 test data dir for run_four_models_test_era5.py
                           (default: <graphcast-root>/test_era5_data)
  --date <YYYYMMDD>        Init date (default: 20260301)
  --hour <0-23>            Init hour (default: 0)
  --num-steps <int>        Num steps for pangu/four-models (default: 2)
  --device <mode>          auto|dcu|cuda|cpu (default: auto)
  --result-root <path>     Output root (default: <graphcast-root>/ZK_Models/results/conda_migrate_<ts>)
  --skip-all-models        Run only steps 1-3
  --dry-run                Print commands only
  -h, --help               Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GRAPHCAST_ROOT="$(cd "${ZK_ROOT}/.." && pwd)"
ERA5_ROOT="/public/share/aciwgvx1jd/CEPRI_ERA5"
TEST_DATA="${GRAPHCAST_ROOT}/test_era5_data"
DATE="20260301"
HOUR=0
NUM_STEPS=2
DEVICE="auto"
TS="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${ZK_ROOT}/results/conda_migrate_${TS}"
RUN_ALL_MODELS=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graphcast-root)
      GRAPHCAST_ROOT="${2:-}"
      ZK_ROOT="${GRAPHCAST_ROOT}/ZK_Models"
      TEST_DATA="${GRAPHCAST_ROOT}/test_era5_data"
      RESULT_ROOT="${ZK_ROOT}/results/conda_migrate_${TS}"
      shift 2
      ;;
    --era5-root)
      ERA5_ROOT="${2:-}"
      shift 2
      ;;
    --test-data)
      TEST_DATA="${2:-}"
      shift 2
      ;;
    --date)
      DATE="${2:-}"
      shift 2
      ;;
    --hour)
      HOUR="${2:-}"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --result-root)
      RESULT_ROOT="${2:-}"
      shift 2
      ;;
    --skip-all-models)
      RUN_ALL_MODELS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${RESULT_ROOT}"
LOG_FILE="${RESULT_ROOT}/minimal_chain.log"
PANGU_RAW_DIR="${RESULT_ROOT}/pangu_raw"
PANGU_INTERP_DIR="${RESULT_ROOT}/pangu_interp"
FOUR_MODELS_FWFX_OUT="${RESULT_ROOT}/four_models_fwfx"
FOUR_MODELS_ALL_OUT="${RESULT_ROOT}/four_models_all"

run_cmd() {
  local cmd="$1"
  echo ">>> ${cmd}" | tee -a "${LOG_FILE}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  eval "${cmd}" 2>&1 | tee -a "${LOG_FILE}"
}

echo "Result root: ${RESULT_ROOT}" | tee "${LOG_FILE}"
echo "GraphCast root: ${GRAPHCAST_ROOT}" | tee -a "${LOG_FILE}"
echo "ZK root: ${ZK_ROOT}" | tee -a "${LOG_FILE}"

if [[ ! -d "${GRAPHCAST_ROOT}" || ! -d "${ZK_ROOT}" ]]; then
  echo "Invalid graphcast/zk directory." >&2
  exit 1
fi

if [[ ! -d "${ERA5_ROOT}" ]]; then
  echo "Warning: ERA5 root does not exist now: ${ERA5_ROOT}" | tee -a "${LOG_FILE}"
fi

if [[ ! -d "${TEST_DATA}" ]]; then
  echo "Warning: test-data does not exist now: ${TEST_DATA}" | tee -a "${LOG_FILE}"
fi

cd "${GRAPHCAST_ROOT}"

# Step 1
run_cmd "python \"${ZK_ROOT}/infer_cepri_onnx.py\" --model pangu --era5-root \"${ERA5_ROOT}\" --date \"${DATE}\" --hour ${HOUR} --num-steps ${NUM_STEPS} --device \"${DEVICE}\" --output-dir \"${PANGU_RAW_DIR}\""

# Step 2
run_cmd "python \"${ZK_ROOT}/run_pangu_two_interps_noxarray.py\" --raw-dir \"${PANGU_RAW_DIR}\" --num-steps ${NUM_STEPS} --date \"${DATE}\" --hour ${HOUR}"

# Collect interpolation output hint
if [[ "${DRY_RUN}" -eq 0 ]]; then
  LATEST_BUNDLE="$(ls -dt "${ZK_ROOT}"/results/pangu_interp_bundle_* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${LATEST_BUNDLE}" ]]; then
    mkdir -p "${PANGU_INTERP_DIR}"
    echo "Latest interpolation bundle: ${LATEST_BUNDLE}" | tee -a "${LOG_FILE}"
  fi
fi

# Step 3
run_cmd "python \"${ZK_ROOT}/run_four_models_test_era5.py\" --test-data \"${TEST_DATA}\" --date \"${DATE}\" --hour ${HOUR} --num-steps ${NUM_STEPS} --device \"${DEVICE}\" --only-models fengwu,fuxi"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ -d "${GRAPHCAST_ROOT}/result/four_models_test_era5" ]]; then
    mkdir -p "${FOUR_MODELS_FWFX_OUT}"
    cp -r "${GRAPHCAST_ROOT}/result/four_models_test_era5/." "${FOUR_MODELS_FWFX_OUT}/"
  fi
fi

# Step 4
if [[ "${RUN_ALL_MODELS}" -eq 1 ]]; then
  run_cmd "python \"${ZK_ROOT}/run_four_models_test_era5.py\" --test-data \"${TEST_DATA}\" --date \"${DATE}\" --hour ${HOUR} --num-steps ${NUM_STEPS} --device \"${DEVICE}\""
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ -d "${GRAPHCAST_ROOT}/result/four_models_test_era5" ]]; then
      mkdir -p "${FOUR_MODELS_ALL_OUT}"
      cp -r "${GRAPHCAST_ROOT}/result/four_models_test_era5/." "${FOUR_MODELS_ALL_OUT}/"
    fi
  fi
fi

echo
echo "Minimal chain completed. See log: ${LOG_FILE}"
