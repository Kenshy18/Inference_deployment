#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
work_root=${DEPLOYMENT_TEST_WORK_ROOT:-/mnt/d/GUI_frontend/deployment-test/work}
fixture_root="$work_root/fixtures"
runtime_root=${MASK_RUNTIME_ROOT:-/home/kenshin/.local/share/video-mask-runtime}
ffmpeg="$runtime_root/tools/ffmpeg/bin/ffmpeg"
ffprobe="$runtime_root/tools/ffmpeg/bin/ffprobe"
source_short="$repository_root/data/codino_trt_3min_simple150_input.mp4"
source_15m="$repository_root/data/新しいフォルダー/HEYZO-3545_30分-45分.mp4"
source_alt="$repository_root/data/新しいフォルダー/HEYZO-3549 浜田希 はまたのそみ 激しめイラマか好き - 無修正アタルト動画 HEYZO -.mp4"
source_long="$repository_root/data/新しいフォルダー/連結済み_長時間動画.mp4"

mkdir -p "$fixture_root"
for required in "$ffmpeg" "$ffprobe" "$source_short" "$source_15m" "$source_alt" "$source_long"; do
  if [[ ! -e "$required" ]]; then
    printf 'missing fixture dependency: %s\n' "$required" >&2
    exit 1
  fi
done

make_fixture() {
  local target=$1
  shift
  if [[ -s "$fixture_root/$target" ]]; then
    printf '[fixture] reuse %s\n' "$target"
    return
  fi
  printf '[fixture] create %s\n' "$target"
  "$ffmpeg" -hide_banner -loglevel error -y "$@" "$fixture_root/$target"
}

make_fixture golden_1080p2997_h264_aac.mp4 \
  -i "$source_short" -t 176 -map 0:v:0 -map '0:a?' -c copy
make_fixture golden_short.mp4 \
  -ss 60 -i "$source_short" -t 20 -map 0:v:0 -map '0:a?' -c copy
make_fixture real_720p24_45s.mp4 \
  -ss 120 -i "$source_15m" -t 45 -map 0:v:0 -map '0:a?' -c copy
make_fixture landscape_720p24_h265.mkv \
  -ss 240 -i "$source_15m" -t 30 -map 0:v:0 -map '0:a?' \
  -c:v libx265 -preset faster -crf 22 -pix_fmt yuv420p10le -c:a aac -b:a 128k
make_fixture portrait_720x1280_30_h264.mp4 \
  -ss 360 -i "$source_15m" -t 30 -map 0:v:0 -map '0:a?' \
  -vf 'scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=30' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -b:a 128k
make_fixture real_portrait_720x1280_20s.mkv \
  -ss 420 -i "$source_15m" -t 20 -map 0:v:0 -map '0:a?' \
  -vf 'scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280' \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -b:a 128k
make_fixture uhd_2160p24_h265_noaudio.mp4 \
  -ss 480 -i "$source_15m" -t 20 -map 0:v:0 -an -vf scale=3840:2160 \
  -c:v hevc_nvenc -preset p4 -tune hq -cq 22 -pix_fmt yuv420p
make_fixture real_4k24_20s_noaudio.mp4 \
  -ss 540 -i "$source_15m" -t 20 -map 0:v:0 -an -vf scale=3840:2160 \
  -c:v hevc_nvenc -preset p4 -tune hq -cq 22 -pix_fmt yuv420p
make_fixture vfr_pts_gap_h264.mp4 \
  -ss 600 -i "$source_15m" -t 30 -map 0:v:0 -an \
  -vf "select='if(lt(mod(n,120),60),1,not(mod(n,2)))'" -fps_mode vfr \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p
make_fixture real_vfr_pts_gap_30s.mp4 \
  -ss 660 -i "$source_15m" -t 30 -map 0:v:0 -an \
  -vf "select='if(lt(mod(n,96),48),1,not(mod(n,2)))'" -fps_mode vfr \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p
make_fixture long_gop_bframes.mov \
  -ss 720 -i "$source_15m" -t 30 -map 0:v:0 -map '0:a?' \
  -c:v libx264 -preset veryfast -crf 18 -g 250 -bf 3 -pix_fmt yuv420p \
  -c:a aac -b:a 128k
make_fixture short_60fps.mp4 \
  -ss 900 -i "$source_alt" -t 20 -map 0:v:0 -map '0:a?' \
  -vf 'scale=1280:720,fps=60' -c:v libx264 -preset veryfast -crf 18 \
  -pix_fmt yuv420p -c:a aac -b:a 128k
make_fixture 'unicode_日本語 space.mp4' \
  -ss 1100 -i "$source_alt" -t 20 -map 0:v:0 -map '0:a?' -c copy
make_fixture h264_noaudio.mp4 \
  -ss 1200 -i "$source_alt" -t 20 -map 0:v:0 -an -c copy

if [[ ! -s "$fixture_root/invalid_truncated.mp4" ]]; then
  cp -- "$fixture_root/golden_short.mp4" "$fixture_root/invalid_truncated.mp4"
  size=$(stat -c %s "$fixture_root/invalid_truncated.mp4")
  truncate -s "$((size / 2))" "$fixture_root/invalid_truncated.mp4"
fi

make_fixture codino_120m_mixed.mp4 \
  -i "$source_long" -t 7200 -map 0:v:0 -map '0:a?' -c copy -movflags +faststart

"$repository_root/deployment_tests/scripts/probe_fixtures.py" \
  --fixtures "$fixture_root" \
  --output "$work_root/fixtures-report.json"
