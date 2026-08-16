#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
runtime_python=${INFERENCE_RUNTIME_PYTHON:-/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10}
native_interval_root=/home/kenshin/.local/share/video-mask-runtime/native-interval
profile=core
full_hash=0
skip_overlay_bootstrap=0
skip_windows_check=0
skip_tests=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile=$2; shift 2 ;;
    --runtime-python) runtime_python=$2; shift 2 ;;
    --full-hash) full_hash=1; shift ;;
    --skip-overlay-bootstrap) skip_overlay_bootstrap=1; shift ;;
    --skip-windows-check) skip_windows_check=1; shift ;;
    --skip-tests) skip_tests=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$profile" != core && "$profile" != all ]]; then
  echo "--profile must be core or all" >&2
  exit 2
fi
if [[ ! -x "$runtime_python" ]]; then
  echo "production runtime Python is unavailable: $runtime_python" >&2
  echo "Phase 2 reuses the existing runtime; phase 3 will provision it." >&2
  exit 1
fi

MASK_PIPELINE_NATIVE_ROOT="$native_interval_root" \
INFERENCE_RUNTIME_PYTHON="$runtime_python" \
  "$repo_root/postprocess/production/polygon/runtime/native_interval/bootstrap_and_build.sh"

verify=("$runtime_python" "$script_dir/verify_assets.py" --root "$repo_root" --profile "$profile" --stage runtime)
if [[ $full_hash -eq 1 ]]; then verify+=(--full-hash); fi
"${verify[@]}"

inference_root="$repo_root/InstanceSegmentation/inference"
"$runtime_python" "$inference_root/dinov3_codino/setup_environment.py" --runtime-python "$runtime_python"
"$runtime_python" "$inference_root/dinov3_codino_mh0/setup_environment.py" --python "$runtime_python" --extract-only
"$runtime_python" "$inference_root/face_dino_v2/setup_environment.py" --runtime-python "$runtime_python" --verify engines
if [[ "$profile" == all ]]; then
  "$runtime_python" "$inference_root/rtdetr_head_face/setup_environment.py" --runtime-python "$runtime_python"
  "$runtime_python" "$inference_root/dinov3_cascade/setup_environment.py" --runtime-python "$runtime_python"
  "$runtime_python" "$inference_root/eva02_cascade/setup_environment.py" --runtime-python "$runtime_python"
fi

if [[ $skip_overlay_bootstrap -eq 0 ]]; then
  "$repo_root/overlay/bootstrap_fast_runtime.sh"
  "$repo_root/overlay/native/bootstrap_runtime.sh"
fi
"$repo_root/overlay/native/build.sh"

if [[ $skip_tests -eq 0 ]]; then
  PYTHONPATH="$repo_root/orchestration/tests:$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$runtime_python" -m pytest -q \
    "$repo_root/orchestration/tests" \
    "$repo_root/overlay/native/tests/test_modes.py"
fi

preflight=("$runtime_python" "$script_dir/preflight.py" --root "$repo_root" --profile "$profile" --runtime-python "$runtime_python")
if [[ $full_hash -eq 1 ]]; then preflight+=(--full-hash); fi
"${preflight[@]}"

if [[ $skip_windows_check -eq 0 ]] && command -v powershell.exe >/dev/null 2>&1; then
  windows_check=$(wslpath -w "$repo_root/gui/windows/Check-WindowsRuntime.ps1")
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$windows_check" \
    -BackendRoot "$repo_root" -RuntimePython "$runtime_python"
fi

echo "[PASS] phase-2 clone setup: profile=$profile root=$repo_root"
