#!/usr/bin/env bash
set -euo pipefail

overlay_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime_root="$overlay_root/.runtime"
archive="$runtime_root/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
target="$runtime_root/ffmpeg-nvenc-btbn-8.1"
archive_url=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-24-13-32/ffmpeg-n8.1.2-31-g8c9502e9b0-linux64-gpl-8.1.tar.xz
archive_sha256=67b4011d82c67586c3703c81d3c967a837319f75e5e61551f22a91588e440660

if [[ -x "$target/bin/ffmpeg" && -x "$target/bin/ffprobe" ]]; then
  echo "fast overlay FFmpeg ready: $target"
  exit 0
fi
mkdir -p "$runtime_root"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 -o "$archive.partial" "$archive_url"
  mv "$archive.partial" "$archive"
fi
observed=$(sha256sum "$archive" | awk '{print $1}')
if [[ "$observed" != "$archive_sha256" ]]; then
  echo "fast overlay FFmpeg hash mismatch: expected=$archive_sha256 observed=$observed" >&2
  exit 1
fi
[[ ! -e "$target" ]] || {
  echo "refusing to replace incomplete fast overlay runtime: $target" >&2
  exit 1
}
temporary=$(mktemp -d "$runtime_root/.ffmpeg-fast.XXXXXX")
cleanup() {
  find "$temporary" -mindepth 1 -depth -delete 2>/dev/null || true
  rmdir "$temporary" 2>/dev/null || true
}
trap cleanup EXIT
source_root="$temporary/extracted"
mkdir "$source_root"
tar -xJf "$archive" -C "$source_root" --strip-components=1
[[ -x "$source_root/bin/ffmpeg" && -x "$source_root/bin/ffprobe" ]] || {
  echo "fast overlay FFmpeg archive is incomplete" >&2
  exit 1
}
mv "$source_root" "$target"
printf '%s  %s\n' "$archive_sha256" "$archive" > "$target/archive.sha256"
if [[ -e "$runtime_root/ffmpeg-nvenc" && ! -L "$runtime_root/ffmpeg-nvenc" ]]; then
  echo "refusing to replace non-symlink runtime alias" >&2
  exit 1
fi
ln -sfn "$(basename "$target")" "$runtime_root/ffmpeg-nvenc"
echo "fast overlay FFmpeg ready: $target"
