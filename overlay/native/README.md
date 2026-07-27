# Native overlay engine

`overlay-render --execution-mode fast`が内部で使う本番用native実装です。
利用者向けの入口は統合CLIであり、このディレクトリのコマンドはbuild、回帰試験、
性能調査向けです。

## 構成

```text
GPU worker:
MP4 -> libavformat -> NVDEC CUDA frame -> CUDA NV12 span描画
    -> NVENC zero-copy -> MP4

CPU worker:
MP4 -> libavformat -> software decode -> YUV420P span描画
    -> libx264 -> MP4

SQLite ------------------------------------------------------^
```

Python frame loop、OpenCV、BGR変換、rawvideo pipeを使いません。postprocess
mask SQLiteとunified inference SQLiteを読み、BT.709 limited-range
YUV420P/NV12へ直接描画します。

対応機能:

- `raw`、`tracked`、`final`、`faces`
- `final`への顔box追加
- ASCII label、mask塗り、輪郭、顔box
- libx264またはNVENC出力
- AAC等の入力音声stream copy
- atomic動画出力とJSON summary
- フレーム区間の並列実行と再encodeなしの連結

通常OpenCV版とフォント、アンチエイリアス、境界pixelは完全一致しません。
GPUラベルは速度優先の組み込みASCII fontです。

## Build

toolchainとshared libraryは`native/.runtime`に配置します。未導入時だけ:

```bash
./bootstrap_runtime.sh
```

build:

```bash
./build.sh
```

生成物は`build/overlay_native`です。CUDA 12.9 compiler/runtimeの場所は現環境の
production runtimeに固定しています。

## 直接実行

通常は統合CLIを使ってください。単一workerの内部調査例:

```bash
build/overlay_native \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output output/final.mp4 \
  --encoder h264_nvenc \
  --preset p1 \
  --bitrate-mbps 8 \
  --gpu-pipeline
```

分割runner:

```bash
python3 segmented.py \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/work \
  --workers 6 \
  --cpu-workers 0 \
  --bitrate-mbps 8 \
  --gpu-pipeline
```

`--cpu-workers 3`ならCPU 3＋NVENC 3です。各workerは担当開始位置へseekし、
PTSから正確なframe番号を復元します。最後はFFmpeg concat demuxerで連結します。

## Test

```bash
./run_tests.sh
```

小型H.264＋AAC動画と両SQLite schemaを一時生成し、4 overlay種別、顔併合、
ラベル、音声、manifest、atomic出力を回帰確認します。

採用検証は[`../docs/ADOPTION_VALIDATION.md`](../docs/ADOPTION_VALIDATION.md)、
性能履歴は[`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md)を参照してください。
