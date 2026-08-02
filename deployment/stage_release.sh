#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "$script_dir/.." && pwd)
runtime_env_root=/home/kenshin/.local/share/video-mask-runtime/envs
stage_root=
profile=all

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-root) stage_root=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$stage_root" ]]; then
  echo "--stage-root is required" >&2
  exit 2
fi
if [[ "$profile" != core && "$profile" != all ]]; then
  echo "--profile must be core or all" >&2
  exit 2
fi
if [[ -e "$stage_root" ]]; then
  echo "refusing to replace existing stage root: $stage_root" >&2
  exit 1
fi
if [[ -n $(git -C "$repository_root" status --porcelain) ]]; then
  echo "release staging requires a clean Git worktree" >&2
  exit 1
fi

release_commit=$(git -C "$repository_root" rev-parse HEAD)
runtime_root="$runtime_env_root/production"
runtime_sources="$runtime_env_root/src"
[[ -x "$runtime_root/bin/python3.10" ]] || {
  echo "production runtime is unavailable: $runtime_root" >&2
  exit 1
}
[[ -d "$runtime_sources" ]] || {
  echo "production runtime sources are unavailable: $runtime_sources" >&2
  exit 1
}

mkdir -p "$stage_root"
cleanup_on_failure=1
cleanup() {
  if [[ $cleanup_on_failure -eq 1 ]]; then
    find "$stage_root" -mindepth 1 -depth -delete 2>/dev/null || true
    rmdir "$stage_root" 2>/dev/null || true
  fi
}
trap cleanup EXIT

git -C "$repository_root" bundle create \
  "$stage_root/inference-backend2.bundle" HEAD
"$runtime_root/bin/python3.10" "$script_dir/export_assets.py" \
  "$stage_root/production-assets" --root "$repository_root" --profile "$profile"
tar -cpf "$stage_root/production-runtime.tar" \
  -C "$runtime_env_root" production
tar -cpf "$stage_root/runtime-sources.tar" \
  -C "$runtime_env_root" src
cp "$script_dir/phase3/bootstrap_image.sh" "$stage_root/bootstrap_image.sh"
cp "$script_dir/phase3/prepare_image.sh" "$stage_root/prepare_image.sh"

cleanup_on_failure=0
echo "[PASS] release stage: root=$stage_root commit=$release_commit profile=$profile"
