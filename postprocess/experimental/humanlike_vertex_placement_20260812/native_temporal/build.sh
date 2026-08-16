#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BUILD_DIR="${SOURCE_DIR}/build"
readonly BUILD_ENV="${MASK_PIPELINE_NATIVE_BUILD_ENV:-/home/kenshin/.local/share/mask-pipeline-native-build/env}"

"${BUILD_ENV}/bin/cmake" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BUILD_ENV}" \
  -DPython_EXECUTABLE="/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10"
"${BUILD_ENV}/bin/cmake" --build "${BUILD_DIR}" --parallel
