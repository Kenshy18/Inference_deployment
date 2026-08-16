#!/usr/bin/env bash
set -euo pipefail

target=${1:-}
allowed_root=${2:-}
case "$allowed_root" in
  /mnt/?/*) ;;
  *) echo "refusing unsafe release work root: $allowed_root" >&2; exit 2 ;;
esac
root_resolved=$(realpath -m -- "$allowed_root")
target_resolved=$(realpath -m -- "$target")
case "$root_resolved" in
  /mnt/?|/mnt/?/) echo "refusing broad release work root: $root_resolved" >&2; exit 2 ;;
esac
if [[ $(dirname -- "$target_resolved") != "$root_resolved" ]] || \
   [[ $(basename -- "$target_resolved") != mask-pipeline-* ]]; then
  echo "refusing unsafe release work cleanup: $target_resolved (root=$root_resolved)" >&2
  exit 2
fi
[[ -d "$target_resolved" ]] || exit 0
find "$target_resolved" -mindepth 1 -depth -delete
rmdir "$target_resolved"
