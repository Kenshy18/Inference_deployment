#!/usr/bin/env bash
set -euo pipefail

readonly PRODUCTION_PYTHON="${MASK_PIPELINE_PRODUCTION_PYTHON:-/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10}"
readonly TARGET="${MASK_PIPELINE_CUDA_EXPERIMENT_SITE:-/home/kenshin/.local/share/mask-pipeline-cuda-experiment}"

if [[ ! -x "${PRODUCTION_PYTHON}" ]]; then
  echo "Production Python was not found: ${PRODUCTION_PYTHON}" >&2
  exit 1
fi

mkdir -p "${TARGET}"
"${PRODUCTION_PYTHON}" -m pip install \
  --disable-pip-version-check \
  --target "${TARGET}" \
  --upgrade \
  'cupy-cuda12x==13.6.0' \
  'fastrlock==0.8.3'

MASK_PIPELINE_CUDA_EXPERIMENT_SITE="${TARGET}" \
  "${PRODUCTION_PYTHON}" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["MASK_PIPELINE_CUDA_EXPERIMENT_SITE"])
import cupy

device = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"CuPy {cupy.__version__}; CUDA device: {device}")
PY
