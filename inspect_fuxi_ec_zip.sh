#!/usr/bin/env bash
# Wait for FuXi_EC.zip (Zenodo 10401602), verify size/md5, unzip, inventory.
set -euo pipefail
ZK="$(cd "$(dirname "$0")" && pwd)"
cd "$ZK"
ZIP="FuXi_EC.zip"
EXPECTED_SIZE=8633905248
EXPECTED_MD5="310e6d5de0a421d7f253c68a886431e0"
OUTDIR="FuXi_EC_extracted"
REPORT="FuXi_EC_inspect_report.txt"

{
  echo "=== FuXi_EC.zip post-download check ==="
  echo "Started: $(date -Is)"
  while true; do
    sz=$(stat -c%s "$ZIP")
    echo "$(date -Is) size=${sz} (expect ${EXPECTED_SIZE})"
    if [ "$sz" -ge "$EXPECTED_SIZE" ]; then
      break
    fi
    if ! pgrep -af "FuXi_EC[.]zip" >/dev/null 2>&1; then
      echo "No active download process; if size < expected, download may have failed."
    fi
    sleep 120
  done

  echo "=== md5 ==="
  md5sum "$ZIP"

  got=$(md5sum "$ZIP" | awk '{print $1}')
  if [ "$got" != "$EXPECTED_MD5" ]; then
    echo "WARNING: md5 mismatch (got $got, expected $EXPECTED_MD5)"
  else
    echo "OK: md5 matches Zenodo record."
  fi

  echo "=== unzip -> ${OUTDIR} ==="
  rm -rf "$OUTDIR"
  unzip -q "$ZIP" -d "$OUTDIR"

  echo "=== directory tree (depth 3) ==="
  find "$OUTDIR" -maxdepth 3 -print | sort

  echo "=== files: onnx and weight blobs ==="
  find "$OUTDIR" -type f \( -iname '*.onnx' -o -name 'short' -o -name 'medium' -o -name 'long' \) -print | sort

  echo "=== search: mean / std / stats / npy / npz (likely empty for FuXi_EC) ==="
  find "$OUTDIR" -type f \( \
    -iname '*mean*' -o -iname '*std*' -o -iname '*stat*' \
    -o -iname '*.npy' -o -iname '*.npz' \
    \) -print | sort || true

  echo "=== file count & total bytes (unpacked) ==="
  find "$OUTDIR" -type f | wc -l
  du -sh "$OUTDIR"

  echo "Finished: $(date -Is)"
} | tee "$REPORT"

echo "Wrote $ZK/$REPORT"
