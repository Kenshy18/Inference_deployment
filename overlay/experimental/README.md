# Experimental low-level overlay renderer

このディレクトリは現行`overlay_renderer`を変更せず、低レイヤー化による速度上限を
測るための独立実装です。productionの主要4モードとCLI名へ対応し、
repository orchestrationから任意の高速backendとして選択できます。任意codecや
pixel formatまで含む標準実装の完全置換ではありません。

## Architecture

```text
CPU path:
MP4 -> libavformat -> software decode -> YUV420P span blend
    -> libx264 / NVENC -> libavformat -> MP4

GPU path:
MP4 -> libavformat -> NVDEC CUDA frame -> custom CUDA NV12 span kernel
    -> NVENC (zero-copy) -> libavformat -> MP4

SQLite -----------------------------------------------^
```

Python frame loop、OpenCV、BGR変換、rawvideo pipeを使用しません。postprocess
SQLiteとunified inference SQLiteを読み、BT.709 limited-range YUV420P/NV12へ
直接合成します。GPU経路では復号後の映像をCPUへdownloadせず、同じCUDA
hardware-frame poolをNVENCへ渡します。

現在の試作範囲:

- `raw`、`tracked`、`final`、`faces`の4モード
- final maskへの顔box追加
- track/class/scoreのASCIIラベル
- AAC等の入力音声stream copy
- atomic動画出力とJSON manifest
- H.264 yuv420p入力
- libx264またはNVENC出力
- mask塗り、輪郭、顔box
- software decode + CPU span blend、およびNVDEC + CUDA + NVENC

現在もproduction OpenCV版とpixel完全一致するフォント／アンチエイリアスでは
ありません。GPU版のラベルは速度優先の組み込みseven-segment風ASCII fontです。

## Build

ポータブルZig C++ toolchain、FFmpeg shared SDK、SQLite headerを
`experimental/.runtime`へ配置してから実行します。CUDA kernelのbuildには現在、
production runtimeのCUDA 12.9 (`nvcc`, header, `libcudart`)を使用します。

```bash
./bootstrap_runtime.sh
./build.sh
```

## Run

```bash
build/overlay_lowlevel \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output output/final.mp4 \
  --encoder h264_nvenc \
  --preset p1 \
  --bitrate-mbps 8 \
  --mask-alpha 0.32 \
  --outline-thickness 2 \
  --decoder-threads 0
```

完全GPU経路は`--gpu-pipeline`を追加します。`--encoder h264_nvenc`専用です。

```bash
build/overlay_lowlevel \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output output/final_gpu.mp4 \
  --encoder h264_nvenc \
  --preset p1 \
  --bitrate-mbps 8 \
  --gpu-pipeline
```

顔と音声を含む最終overlay:

```bash
build/overlay_lowlevel \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --include-faces \
  --face-sqlite inference.sqlite \
  --output output/final_faces.mp4 \
  --manifest output/final_faces.json \
  --codec h264_nvenc \
  --nvenc-preset p1 \
  --target-bitrate-mbps 8 \
  --gpu-pipeline \
  --copy-audio
```

`--hw-decode`だけを指定する経路はNVDEC後にCPUへdownloadして合成し、再びGPUへ
uploadする比較用です。通常はCPU経路か`--gpu-pipeline`を選びます。

品質基準用のlibx264 lossless出力は`--encoder libx264 --crf 0`で生成できます。

## Segmented benchmark

```bash
python benchmark_segmented.py \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/hybrid \
  --workers 4 \
  --cpu-workers 2 \
  --end-frame 5289 \
  --bitrate-mbps 8
```

各C++ workerは担当開始frame付近のkeyframeへlibavformatでseekし、PTSから正確な
frame番号を復元します。最後はFFmpeg concat demuxerで再encodeなしに結合します。

完全GPUベンチマーク例:

```bash
python benchmark_segmented.py \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/gpu \
  --workers 6 \
  --cpu-workers 0 \
  --end-frame 5289 \
  --bitrate-mbps 8 \
  --gpu-pipeline
```

分割処理でも`--include-faces --face-sqlite ... --copy-audio`を指定できます。
音声はsegmentへ重複させず、映像concat後に元動画の該当時間範囲をstream copy
します。

今回のRTX 5090実測上の推奨値は1080pで6 worker、4Kで4 workerです。最重量の
final＋顔＋ラベル＋音声でも、軸平行span圧縮後は1080p 6 workerが最速でした。
入力長、codec、ストレージ、同時GPU負荷が変わる場合は再探索してください。

`--gpu-pipeline`と`--cpu-workers N`を併用すると、NVENC workerだけを完全GPU経路、
libx264 workerをCPU経路で実行できます。ただし今回の実測ではメモリ帯域競合に
よりGPU専用構成より遅かったため、比較・将来検証用です。

速度優先の既定ではMP4の`moov` atomを末尾に置きます。Web配信などで先頭配置が
必要な場合は、単体rendererまたはsegmented benchmarkへ`--faststart`を追加
してください。画質は変わりませんが最終ファイル移動の固定費が増えます。

詳細な実測結果は[REPORT.md](REPORT.md)を参照してください。

## Orchestration

既存のPython/OpenCV backendを既定値のまま残し、設定で明示的に切り替えます。
`workers: 6, cpu_workers: 0`は純GPU、`workers: 6, cpu_workers: 3`は
CPU 3＋NVENC 3です。

```json
{
  "overlay": {
    "backend": "experimental_cpp",
    "codec": "h264_nvenc",
    "workers": 6,
    "cpu_workers": 0,
    "target_bitrate_mbps": 8.0,
    "nvenc_preset": "p1",
    "copy_audio": true
  }
}
```

4モードの実データ検証設定は
[`../../orchestration/configs/experimental_cpp_4modes_300_20260726.json`](../../orchestration/configs/experimental_cpp_4modes_300_20260726.json)
にあります。

## Test

```bash
./run_tests.sh
```

小型H.264＋AAC動画と両SQLite schemaを一時生成し、4モード、顔併合、ラベル、
音声、manifest、atomic出力をCPU経路で回帰確認します。
