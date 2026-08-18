#!/usr/bin/env bash
set -euo pipefail

repository_root=/home/kenshin/inference_backend2
runtime_root=/home/kenshin/.local/share/video-mask-runtime/envs/production
release_commit=
asset_commit=
profile=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-commit) release_commit=$2; shift 2 ;;
    --asset-commit) asset_commit=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $(id -u) -ne 0 ]]; then
  echo "finalize_image.sh must run as root" >&2
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

fixture_root=/opt/mask-pipeline/fixtures
report_root=/opt/mask-pipeline/release
install -d -o kenshin -g kenshin "$fixture_root" "$report_root"
ffmpeg_binary="$repository_root/overlay/native/.runtime/ffmpeg/bin/ffmpeg"
ffmpeg_library="$repository_root/overlay/native/.runtime/ffmpeg/lib"
runuser -u kenshin -- env LD_LIBRARY_PATH="$ffmpeg_library" \
  "$ffmpeg_binary" -hide_banner -loglevel error -y \
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
  "profile": "$profile",
  "distribution": null,
  "distribution_policy": "assigned-by-versioned-wsl-import",
  "backend_root": "$repository_root",
  "runtime_python": "$runtime_root/bin/python3.10",
  "fixture": "$fixture_root/deployment-smoke.mp4"
}
EOF
chown -R kenshin:kenshin /opt/mask-pipeline
runuser -u kenshin -- "$runtime_root/bin/python3.10" \
  "$repository_root/deployment/preflight.py" \
  --root "$repository_root" --profile "$profile" \
  --runtime-python "$runtime_root/bin/python3.10" --full-hash \
  > "$report_root/preflight-build.json"

# The build clone is also the runtime source tree.  Keep comparison notebooks,
# test suites and retired policy modules out of the deployed distribution after
# all image-construction tests have passed.  Production imports are guarded by
# postprocess/tests/test_engine_imports.py before this point.
pruned_paths=(
  "$repository_root/postprocess/tentative"
  "$repository_root/postprocess/tests"
  "$repository_root/orchestration/tests"
  "$repository_root/overlay/native/tests"
  "$repository_root/postprocess/approximation/polygon/production.py"
  "$repository_root/postprocess/approximation/polygon/rdp.py"
  "$repository_root/postprocess/approximation/polygon/stages.py"
  "$repository_root/postprocess/approximation/ellipse"
  "$repository_root/postprocess/keyframes/polygon"
  "$repository_root/postprocess/keyframes/ellipse"
  "$repository_root/postprocess/gap_fill/polygon"
  "$repository_root/postprocess/gap_fill/ellipse"
  "$repository_root/postprocess/models/k2_v5"
  "$repository_root/postprocess/nms/adaptive.py"
  "$repository_root/postprocess/nms/component_aware.py"
  "$repository_root/postprocess/nms/stages.py"
)
pruned_manifest="$report_root/pruned-development-paths.txt"
: > "$pruned_manifest"
for target in "${pruned_paths[@]}"; do
  if [[ -d "$target" ]]; then
    printf '%s\n' "${target#"$repository_root/"}" >> "$pruned_manifest"
    find "$target" -mindepth 1 -depth -delete
    rmdir "$target"
  elif [[ -f "$target" ]]; then
    printf '%s\n' "${target#"$repository_root/"}" >> "$pruned_manifest"
    find "$target" -maxdepth 0 -type f -delete
  fi
done
find "$repository_root" -type d \
  \( -name __pycache__ -o -name .pytest_cache \) -prune -exec sh -c '
    for directory do
      find "$directory" -mindepth 1 -depth -delete
      rmdir "$directory"
    done
  ' sh {} +
chown kenshin:kenshin "$pruned_manifest"

# Test the tree that users actually receive.  The construction-time preflight
# above proves assets and native dependencies before pruning; this second gate
# proves that the runtime-only source tree remains importable and contains no
# retired Production implementation.
runuser -u kenshin -- "$runtime_root/bin/python3.10" \
  "$repository_root/deployment/preflight.py" \
  --root "$repository_root" --profile "$profile" \
  --runtime-python "$runtime_root/bin/python3.10" --full-hash \
  > "$report_root/preflight.json"

for forbidden in /home/kenshin/.codex; do
  [[ ! -e "$forbidden" ]] || {
    echo "forbidden developer state found in image: $forbidden" >&2
    exit 1
  }
done
for ssh_root in /home/kenshin/.ssh /root/.ssh; do
  if [[ -d "$ssh_root" ]] && [[ -n $(find "$ssh_root" -mindepth 1 -print -quit) ]]; then
    echo "forbidden SSH state found in image: $ssh_root" >&2
    exit 1
  fi
  rmdir "$ssh_root" 2>/dev/null || true
done
find /home/kenshin -maxdepth 2 -type f \
  \( -name '.bash_history' -o -name '.python_history' \) -delete
find "$repository_root/output" -mindepth 1 -depth -delete 2>/dev/null || true
mkdir -p "$repository_root/output"
printf '\n' > "$repository_root/output/.gitkeep"
chown -R kenshin:kenshin "$repository_root/output"
apt-get clean
find /var/lib/apt/lists -mindepth 1 -depth -delete

echo "[PASS] distribution image prepared: release=$release_commit assets=$asset_commit"
