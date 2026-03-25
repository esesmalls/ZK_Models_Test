#!/usr/bin/env bash
set -Eeuo pipefail

# Export a runnable conda environment and optionally pack it.
# Usage:
#   bash export_pack_conda_env.sh --env-name torch2.4_dtk25.04_cp310_e2s
#   bash export_pack_conda_env.sh --env-name myenv --out-dir /tmp/conda_migrate --skip-pack

usage() {
  cat <<'EOF'
Usage: export_pack_conda_env.sh --env-name <name> [options]

Options:
  --env-name <name>      Conda env name to export and pack (required)
  --out-dir <path>       Output directory for artifacts
                         (default: ZK_Models/conda_migration_artifacts)
  --skip-pack            Skip conda-pack step
  --with-explicit-lock   Export explicit lock file via `conda list --explicit`
  -h, --help             Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUT_DIR="${ZK_ROOT}/conda_migration_artifacts"

ENV_NAME=""
OUT_DIR="${DEFAULT_OUT_DIR}"
DO_PACK=1
WITH_EXPLICIT_LOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --skip-pack)
      DO_PACK=0
      shift
      ;;
    --with-explicit-lock)
      WITH_EXPLICIT_LOCK=1
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

if [[ -z "${ENV_NAME}" ]]; then
  echo "--env-name is required" >&2
  usage
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
YAML_FILE="${OUT_DIR}/environment.${ENV_NAME}.${TS}.yml"
PACK_FILE="${OUT_DIR}/conda_env.${ENV_NAME}.${TS}.tar.gz"
LOCK_FILE="${OUT_DIR}/explicit.${ENV_NAME}.${TS}.txt"
META_FILE="${OUT_DIR}/export_meta.${ENV_NAME}.${TS}.txt"

echo "[1/4] Check environment exists: ${ENV_NAME}"
conda env list | awk '{print $1}' | rg -x "${ENV_NAME}" >/dev/null

echo "[2/4] Export environment YAML -> ${YAML_FILE}"
conda env export -n "${ENV_NAME}" --no-builds > "${YAML_FILE}"

if [[ "${WITH_EXPLICIT_LOCK}" -eq 1 ]]; then
  echo "[3/4] Export explicit lock -> ${LOCK_FILE}"
  conda list -n "${ENV_NAME}" --explicit > "${LOCK_FILE}"
else
  echo "[3/4] Skip explicit lock (enable via --with-explicit-lock)"
fi

if [[ "${DO_PACK}" -eq 1 ]]; then
  echo "[4/4] Pack env tarball -> ${PACK_FILE}"
  conda run -n "${ENV_NAME}" python -c "import conda_pack" >/dev/null 2>&1 || {
    echo "conda-pack is missing in env '${ENV_NAME}'. Install with: conda install -n ${ENV_NAME} -c conda-forge conda-pack" >&2
    exit 1
  }
  conda pack -n "${ENV_NAME}" -o "${PACK_FILE}"
else
  echo "[4/4] Skip conda-pack (--skip-pack)"
fi

{
  echo "timestamp=${TS}"
  echo "env_name=${ENV_NAME}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "yaml=${YAML_FILE}"
  echo "pack=${PACK_FILE}"
  echo "explicit_lock=${LOCK_FILE}"
} > "${META_FILE}"

echo
echo "Artifacts ready:"
echo "  ${YAML_FILE}"
[[ "${WITH_EXPLICIT_LOCK}" -eq 1 ]] && echo "  ${LOCK_FILE}"
[[ "${DO_PACK}" -eq 1 ]] && echo "  ${PACK_FILE}"
echo "  ${META_FILE}"
echo
echo "Next: sync artifacts + code + models + sample data to target server."
