#!/usr/bin/env bash
# Detect visible CUDA GPU count. Prefer torch (respects CUDA_VISIBLE_DEVICES),
# then nvidia-smi, then fall back to 1.
#
# Usage:
#   source scripts/setup/detect_gpus.sh
#   detect_gpus   # sets DETECTED_GPUS

detect_gpus() {
  local count=""
  if command -v python3 >/dev/null 2>&1; then
    count="$(
      python3 - <<'PY' 2>/dev/null || true
import os
try:
    import torch
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
    else:
        print(0)
except Exception:
    print(0)
PY
    )"
  fi
  if [[ -z "${count}" || "${count}" == "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      # Count comma-separated visible device IDs (ignore blanks).
      count="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}')"
    else
      count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
  fi
  if [[ -z "${count}" || "${count}" -lt 1 ]]; then
    count=1
  fi
  DETECTED_GPUS="${count}"
  export DETECTED_GPUS
}
