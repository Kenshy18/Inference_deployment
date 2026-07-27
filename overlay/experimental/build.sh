#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
zig="$script_dir/.runtime/zig/zig"
ffmpeg_root="$script_dir/.runtime/ffmpeg"
sqlite_root="$script_dir/.runtime/sqlite-dev"
output_dir="$script_dir/build"
production_runtime="/home/kenshin/.local/share/video-mask-runtime/envs/production"
nvcc="$production_runtime/bin/nvcc"
cuda_include="$production_runtime/targets/x86_64-linux/include"
cuda_lib="$production_runtime/targets/x86_64-linux/lib"
host_compiler="$production_runtime/bin/x86_64-conda-linux-gnu-g++"

if [[ ! -x "$zig" ]]; then
  echo "missing Zig C++ toolchain: $zig" >&2
  exit 1
fi
if [[ ! -f "$ffmpeg_root/include/libavcodec/avcodec.h" ]]; then
  echo "missing FFmpeg shared SDK: $ffmpeg_root" >&2
  exit 1
fi

mkdir -p "$output_dir"
if [[ ! -x "$nvcc" || ! -x "$host_compiler" ]]; then
  echo "CUDA 12.9 compiler/runtime is unavailable: $production_runtime" >&2
  exit 1
fi
"$nvcc" \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -arch=sm_120 \
  -ccbin "$host_compiler" \
  -I"$script_dir/src" \
  -c "$script_dir/src/cuda_overlay.cu" \
  -o "$output_dir/cuda_overlay.o"
"$zig" c++ \
  -std=c++20 \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Wextra \
  -Wpedantic \
  -I"$ffmpeg_root/include" \
  -I"$sqlite_root/usr/include" \
  -I"$cuda_include" \
  "$script_dir/src/main.cpp" \
  "$output_dir/cuda_overlay.o" \
  -L"$ffmpeg_root/lib" \
  -lavformat \
  -lavcodec \
  -lavutil \
  -Wl,--no-as-needed \
  -lswresample \
  -Wl,--as-needed \
  -L"$cuda_lib" \
  -lcudart \
  /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 \
  -Wl,--disable-new-dtags \
  -Wl,-rpath,'$ORIGIN/../.runtime/ffmpeg/lib' \
  -Wl,-rpath,"$cuda_lib" \
  -o "$output_dir/overlay_lowlevel"

echo "$output_dir/overlay_lowlevel"
