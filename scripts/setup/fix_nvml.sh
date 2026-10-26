#!/usr/bin/env bash
# Repoint libnvidia-ml.so.1 to match the loaded kernel driver.
# apt cuda-toolkit can install a newer user-space NVML than the running driver,
# which breaks nvidia-smi and NCCL (nvmlInit_v2 failed: Driver/library version mismatch).
set -euo pipefail

if [[ ! -r /proc/driver/nvidia/version ]]; then
  exit 0
fi

DRIVER_VER="$(sed -n 's/.*  \([0-9.][0-9.]*\)  Release.*/\1/p' /proc/driver/nvidia/version | head -1)"
if [[ -z "$DRIVER_VER" ]]; then
  exit 0
fi

_repoint() {
  local libdir="$1"
  local target="${libdir}/libnvidia-ml.so.${DRIVER_VER}"
  local link="${libdir}/libnvidia-ml.so.1"
  if [[ -f "$target" && ( -L "$link" || ! -e "$link" ) ]]; then
    local current
    current="$(readlink -f "$link" 2>/dev/null || true)"
    if [[ "$current" != "$target" ]]; then
      ln -sfn "libnvidia-ml.so.${DRIVER_VER}" "$link"
      echo "fix_nvml: ${link} -> libnvidia-ml.so.${DRIVER_VER}"
    fi
  fi
}

for libdir in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
  _repoint "$libdir"
done

# ldconfig can restore the apt/CUDA package symlink (often newer than the
# kernel driver). Re-apply after cache refresh so NCCL sees a matching NVML.
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig 2>/dev/null || true
fi
for libdir in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
  _repoint "$libdir"
done
