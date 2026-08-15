#!/usr/bin/env bash
set -euo pipefail

readonly BUILD_ROOT="${MASK_PIPELINE_NATIVE_ROOT:-/home/kenshin/.local/share/mask-pipeline-native-build}"
readonly MAMBA_BIN="${BUILD_ROOT}/bin/micromamba"
readonly MAMBA_ROOT="${BUILD_ROOT}/mamba-root"
readonly ENV_PREFIX="${BUILD_ROOT}/env"
readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BUILD_DIR="${SOURCE_DIR}/build"
readonly MAMBA_VERSION="2.9.0"
readonly MAMBA_ARCHIVE="${BUILD_ROOT}/downloads/micromamba-${MAMBA_VERSION}.tar.bz2"
readonly MAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION}"
readonly MAMBA_SHA256="8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  mkdir -p "${BUILD_ROOT}/bin" "${BUILD_ROOT}/downloads"
  curl --fail --location --retry 3 --output "${MAMBA_ARCHIVE}" "${MAMBA_URL}"
  echo "${MAMBA_SHA256}  ${MAMBA_ARCHIVE}" | sha256sum --check --status
  python3 - "${MAMBA_ARCHIVE}" "${MAMBA_BIN}" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
with tarfile.open(archive, "r:bz2") as package:
    member = package.getmember("bin/micromamba")
    source = package.extractfile(member)
    if source is None:
        raise RuntimeError("micromamba archive has no bin/micromamba payload")
    destination.write_bytes(source.read())
destination.chmod(0o755)
PY
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${MAMBA_BIN}" create \
    --yes \
    --prefix "${ENV_PREFIX}" \
    --file "${SOURCE_DIR}/environment.yml" \
    --strict-channel-priority
fi

"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release
"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  cmake --build "${BUILD_DIR}" --parallel
"${MAMBA_BIN}" run --prefix "${ENV_PREFIX}" \
  env PYTHONPATH="${BUILD_DIR}" \
  python "${SOURCE_DIR}/test_native_interval_metrics.py"
