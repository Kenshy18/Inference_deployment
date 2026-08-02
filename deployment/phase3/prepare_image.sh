#!/usr/bin/env bash
set -euo pipefail

stage_root=/mnt/d/MaskPipelineDeployment/staging
repository_bundle="$stage_root/inference-backend2.bundle"
runtime_archive="$stage_root/production-runtime.tar"
asset_root="$stage_root/production-assets"
repository_root=/home/kenshin/inference_backend2
runtime_root=/home/kenshin/.local/share/video-mask-runtime/envs/production
release_commit=
asset_commit=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-root) stage_root=$2; shift 2 ;;
    --release-commit) release_commit=$2; shift 2 ;;
    --asset-commit) asset_commit=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
repository_bundle="$stage_root/inference-backend2.bundle"
runtime_archive="$stage_root/production-runtime.tar"
asset_root="$stage_root/production-assets"

if [[ $(id -u) -ne 0 ]]; then
  echo "prepare_image.sh must run as root inside the build distribution" >&2
  exit 1
fi
if [[ -z "$release_commit" || -z "$asset_commit" ]]; then
  echo "--release-commit and --asset-commit are required" >&2
  exit 2
fi
for path in "$repository_bundle" "$runtime_archive" "$asset_root/ASSET_PACK.json"; do
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
chown -R kenshin:kenshin /home/kenshin/.local/share/video-mask-runtime

runuser -u kenshin -- git clone "$repository_bundle" "$repository_root"
runuser -u kenshin -- git -C "$repository_root" checkout --detach "$asset_commit"
runuser -u kenshin -- \
  python3 "$repository_root/deployment/install_assets.py" "$asset_root" --root "$repository_root"
runuser -u kenshin -- git -C "$repository_root" checkout --detach "$release_commit"

runuser -u kenshin -- env \
  INFERENCE_RUNTIME_PYTHON="$runtime_root/bin/python3.10" \
  "$repository_root/deployment/setup_phase2.sh" \
  --profile all --full-hash --skip-windows-check

fixture_root=/opt/mask-pipeline/fixtures
report_root=/opt/mask-pipeline/release
install -d -o kenshin -g kenshin "$fixture_root" "$report_root"
ffmpeg_binary="$repository_root/overlay/native/.runtime/ffmpeg/bin/ffmpeg"
runuser -u kenshin -- "$ffmpeg_binary" -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=8" \
  -f lavfi -i "sine=frequency=880:sample_rate=48000:duration=8" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -shortest -movflags +faststart \
  "$fixture_root/deployment-smoke.mp4"

cat > "$report_root/release.json" <<EOF
{
  "schema_version": 1,
  "release_commit": "$release_commit",
  "asset_commit": "$asset_commit",
  "distribution": "MaskPipelineProduction",
  "backend_root": "$repository_root",
  "runtime_python": "$runtime_root/bin/python3.10",
  "fixture": "$fixture_root/deployment-smoke.mp4"
}
EOF
chown -R kenshin:kenshin /opt/mask-pipeline

runuser -u kenshin -- "$runtime_root/bin/python3.10" \
  "$repository_root/deployment/preflight.py" \
  --root "$repository_root" --profile all \
  --runtime-python "$runtime_root/bin/python3.10" --full-hash \
  > "$report_root/preflight.json"

# Secrets and developer state are never part of the release image.
for forbidden in /home/kenshin/.codex /home/kenshin/.ssh /root/.ssh; do
  [[ ! -e "$forbidden" ]] || {
    echo "forbidden developer state found in image: $forbidden" >&2
    exit 1
  }
done
find /home/kenshin -maxdepth 2 -type f \
  \( -name '.bash_history' -o -name '.python_history' \) -delete
find "$repository_root/output" -mindepth 1 -depth -delete 2>/dev/null || true
mkdir -p "$repository_root/output"
chown -R kenshin:kenshin "$repository_root/output"

echo "[PASS] distribution image prepared: release=$release_commit assets=$asset_commit"
