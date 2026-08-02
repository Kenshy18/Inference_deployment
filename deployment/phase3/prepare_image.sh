#!/usr/bin/env bash
set -euo pipefail

stage_root=/mnt/d/MaskPipelineDeployment/staging
repository_bundle="$stage_root/inference-backend2.bundle"
runtime_archive="$stage_root/production-runtime.tar"
runtime_sources_archive="$stage_root/runtime-sources.tar"
asset_root="$stage_root/production-assets"
repository_root=/home/kenshin/inference_backend2
runtime_root=/home/kenshin/.local/share/video-mask-runtime/envs/production
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
repository_bundle="$stage_root/inference-backend2.bundle"
runtime_archive="$stage_root/production-runtime.tar"
runtime_sources_archive="$stage_root/runtime-sources.tar"
asset_root="$stage_root/production-assets"

if [[ $(id -u) -ne 0 ]]; then
  echo "prepare_image.sh must run as root inside the build distribution" >&2
  exit 1
fi
if [[ -z "$release_commit" || -z "$asset_commit" ]]; then
  echo "--release-commit and --asset-commit are required" >&2
  exit 2
fi
if [[ "$profile" != core && "$profile" != all ]]; then
  echo "--profile must be core or all" >&2
  exit 2
fi
for path in "$repository_bundle" "$runtime_archive" "$runtime_sources_archive" "$asset_root/ASSET_PACK.json"; do
  [[ -e "$path" ]] || { echo "missing staged payload: $path" >&2; exit 1; }
done
[[ ! -e "$repository_root" ]] || {
  echo "refusing to replace existing repository: $repository_root" >&2
  exit 1
}
[[ ! -e "$runtime_root" ]] || {
  echo "refusing to replace existing runtime: $runtime_root" >&2
  exit 1
}

install -d -o kenshin -g kenshin /home/kenshin/.local/share/video-mask-runtime/envs
tar -xf "$runtime_archive" -C /home/kenshin/.local/share/video-mask-runtime/envs
tar -xf "$runtime_sources_archive" -C /home/kenshin/.local/share/video-mask-runtime/envs
chown -R kenshin:kenshin /home/kenshin/.local/share/video-mask-runtime

runuser -u kenshin -- git clone "$repository_bundle" "$repository_root"
runuser -u kenshin -- git -C "$repository_root" checkout --detach "$asset_commit"
runuser -u kenshin -- \
  python3 "$repository_root/deployment/install_assets.py" "$asset_root" --root "$repository_root"
runuser -u kenshin -- git -C "$repository_root" checkout --detach "$release_commit"
install -m 0644 "$repository_root/deployment/phase3/wsl.conf" /etc/wsl.conf

# bootstrap_runtime.sh downloads fixed FFmpeg and the Ubuntu sqlite development
# package into the repository-local native runtime. A freshly minimized rootfs
# has no APT index, so refresh it for image construction and remove it again at
# the end. The deployed runtime itself remains offline-capable.
apt-get update
runuser -u kenshin -- env \
  INFERENCE_RUNTIME_PYTHON="$runtime_root/bin/python3.10" \
  "$repository_root/deployment/setup_phase2.sh" \
  --profile "$profile" --full-hash --skip-windows-check

"$repository_root/deployment/phase3/finalize_image.sh" \
  --release-commit "$release_commit" --asset-commit "$asset_commit" \
  --profile "$profile"
