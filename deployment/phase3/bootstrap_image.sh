#!/usr/bin/env bash
set -euo pipefail

stage_root=
release_commit=
asset_commit=
profile=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-root) stage_root=$2; shift 2 ;;
    --release-commit) release_commit=$2; shift 2 ;;
    --asset-commit) asset_commit=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $(id -u) -ne 0 ]]; then
  echo "bootstrap_image.sh must run as root" >&2
  exit 1
fi
if [[ -z "$stage_root" || -z "$release_commit" || -z "$asset_commit" ]]; then
  echo "--stage-root, --release-commit and --asset-commit are required" >&2
  exit 2
fi
if [[ "$profile" != core && "$profile" != all ]]; then
  echo "--profile must be core or all" >&2
  exit 2
fi

if ! id kenshin >/dev/null 2>&1; then
  useradd --create-home --uid 1000 --shell /bin/bash kenshin
fi
if [[ $(id -u kenshin) -ne 1000 ]]; then
  echo "distribution user kenshin must have uid 1000" >&2
  exit 1
fi
install -m 0644 "$stage_root/wsl.conf" /etc/wsl.conf

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git libegl1 libgl1 libglib2.0-0 libgomp1 \
  libsqlite3-0 python3 sudo tar xz-utils

"$stage_root/prepare_image.sh" \
  --stage-root "$stage_root" \
  --release-commit "$release_commit" \
  --asset-commit "$asset_commit" \
  --profile "$profile"
