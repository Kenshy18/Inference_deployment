#!/usr/bin/env bash
set -euo pipefail

target=${1:-}
case "$target" in
  /mnt/?/MaskPipelineDeployment/work/mask-pipeline-*) ;;
  *) echo "refusing unsafe release work cleanup: $target" >&2; exit 2 ;;
esac
[[ -d "$target" ]] || exit 0
find "$target" -mindepth 1 -depth -delete
rmdir "$target"
