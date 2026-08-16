#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$script_dir/.runtime"
download_dir="$runtime_dir/downloads"
ffmpeg_archive="$download_dir/ffmpeg-shared.tar.xz"
zig_archive="$download_dir/zig.tar.xz"
sqlite_download_dir="$download_dir/sqlite"
sqlite_archive="$sqlite_download_dir/libsqlite3-dev_3.45.1-1ubuntu2.7_amd64.deb"
ffmpeg_sha256=069d8d27c96c9a86a6b2a074fe52cbbd71ce5d7e5a687230d5ae56c2288c8630
zig_sha256=c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05
sqlite_sha256=6b8dfabce91ae021ff957f81dc4a4377c1bfa6eb9332cbf6d38e3e66b5198abb

verify_archive() {
  local path=$1 expected=$2 label=$3 observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$observed" != "$expected" ]]; then
    echo "$label hash mismatch: expected=$expected observed=$observed path=$path" >&2
    exit 1
  fi
}

download_archive() {
  local path=$1 url=$2 label=$3 partial="$1.partial"
  [[ ! -e "$partial" ]] || {
    echo "refusing stale partial $label archive: $partial" >&2
    exit 1
  }
  if ! curl -fL --retry 3 -o "$partial" "$url"; then
    find "$partial" -maxdepth 0 -type f -delete 2>/dev/null || true
    exit 1
  fi
  mv "$partial" "$path"
}

mkdir -p "$download_dir" "$sqlite_download_dir"

if [[ ! -x "$runtime_dir/ffmpeg/bin/ffmpeg" ]]; then
  if [[ ! -f "$ffmpeg_archive" ]]; then
    download_archive "$ffmpeg_archive" \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz" \
      "native FFmpeg"
  fi
  verify_archive "$ffmpeg_archive" "$ffmpeg_sha256" "native FFmpeg"
  mkdir -p "$runtime_dir/ffmpeg"
  tar -xJf "$ffmpeg_archive" \
    -C "$runtime_dir/ffmpeg" \
    --strip-components=1
fi
if [[ ! -x "$runtime_dir/zig/zig" ]]; then
  if [[ ! -f "$zig_archive" ]]; then
    download_archive "$zig_archive" \
      "https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz" \
      "Zig"
  fi
  verify_archive "$zig_archive" "$zig_sha256" "Zig"
  mkdir -p "$runtime_dir/zig"
  tar -xJf "$zig_archive" \
    -C "$runtime_dir/zig" \
    --strip-components=1
fi
if [[ ! -f "$runtime_dir/sqlite-dev/usr/include/sqlite3.h" ]]; then
  if [[ ! -f "$sqlite_archive" ]]; then
    (
      cd "$sqlite_download_dir"
      apt download libsqlite3-dev=3.45.1-1ubuntu2.7
    )
  fi
  verify_archive "$sqlite_archive" "$sqlite_sha256" "SQLite development package"
  mkdir -p "$runtime_dir/sqlite-dev"
  dpkg-deb -x "$sqlite_archive" "$runtime_dir/sqlite-dev"
fi

echo "runtime ready: $runtime_dir"
