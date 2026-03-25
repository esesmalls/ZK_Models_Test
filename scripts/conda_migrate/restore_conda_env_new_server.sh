#!/usr/bin/env bash
set -Eeuo pipefail

# Restore a packed conda env on a new server and run basic dependency checks.

usage() {
  cat <<'EOF'
Usage: restore_conda_env_new_server.sh --pack <conda_pack_tar.gz> [options]

Options:
  --pack <path>          Path to conda-pack tar.gz (required)
  --prefix <path>        Target extraction directory (default: /srv/graphcast/conda_env)
  --check-device <name>  Device hint for ONNX provider check: auto|dcu|cuda|cpu (default: auto)
  --skip-checks          Skip python import/provider checks
  -h, --help             Show this help
EOF
}

PACK_PATH=""
PREFIX="/srv/graphcast/conda_env"
CHECK_DEVICE="auto"
DO_CHECKS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pack)
      PACK_PATH="${2:-}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --check-device)
      CHECK_DEVICE="${2:-}"
      shift 2
      ;;
    --skip-checks)
      DO_CHECKS=0
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

if [[ -z "${PACK_PATH}" ]]; then
  echo "--pack is required" >&2
  usage
  exit 2
fi

if [[ ! -f "${PACK_PATH}" ]]; then
  echo "conda-pack file not found: ${PACK_PATH}" >&2
  exit 1
fi

mkdir -p "${PREFIX}"
echo "[1/3] Extract conda env to ${PREFIX}"
tar -xzf "${PACK_PATH}" -C "${PREFIX}"

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  echo "Python not found in ${PREFIX}/bin/python after extraction." >&2
  exit 1
fi

echo "[2/3] Run conda-unpack"
"${PREFIX}/bin/conda-unpack"

if [[ "${DO_CHECKS}" -eq 1 ]]; then
  echo "[3/3] Validate runtime imports/providers"
  "${PREFIX}/bin/python" - <<PY
import importlib
import sys

mods = ["onnxruntime", "torch", "netCDF4", "xarray", "matplotlib", "ruamel.yaml"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        failed.append((m, repr(e)))

if failed:
    print("Import check failed:")
    for m, err in failed:
        print(f"  - {m}: {err}")
    sys.exit(2)

import onnxruntime as ort
import torch
print("onnxruntime providers:", ort.get_available_providers())
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version:", torch.__version__)
print("device_hint:", "${CHECK_DEVICE}")
PY
fi

echo
echo "Conda env restored."
echo "Activate with: source \"${PREFIX}/bin/activate\""
