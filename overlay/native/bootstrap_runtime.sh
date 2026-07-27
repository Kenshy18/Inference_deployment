#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$script_dir/.runtime"
download_dir="$runtime_dir/downloads"
ffmpeg_archive="$download_dir/ffmpeg-shared.tar.xz"
zig_archive="$download_dir/zig.tar.xz"
sqlite_download_dir="$download_dir/sqlite"

mkdir -p "$download_dir" "$sqlite_download_dir"

if [[ ! -f "$ffmpeg_archive" ]]; then
  curl -fL --retry 3 \
    -o "$ffmpeg_archive" \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz"
fi
if [[ ! -f "$zig_archive" ]]; then
  curl -fL --retry 3 \
    -o "$zig_archive" \
    "https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz"
fi

if [[ ! -x "$runtime_dir/ffmpeg/bin/ffmpeg" ]]; then
  mkdir -p "$runtime_dir/ffmpeg"
  tar -xJf "$ffmpeg_archive" \
    -C "$runtime_dir/ffmpeg" \
    --strip-components=1
fi
if [[ ! -x "$runtime_dir/zig/zig" ]]; then
  mkdir -p "$runtime_dir/zig"
  tar -xJf "$zig_archive" \
    -C "$runtime_dir/zig" \
    --strip-components=1
fi
if [[ ! -f "$runtime_dir/sqlite-dev/usr/include/sqlite3.h" ]]; then
  (
    cd "$sqlite_download_dir"
    apt download libsqlite3-dev
  )
  mkdir -p "$runtime_dir/sqlite-dev"
  for package in "$sqlite_download_dir"/*.deb; do
    dpkg-deb -x "$package" "$runtime_dir/sqlite-dev"
  done
fi

echo "runtime ready: $runtime_dir"

