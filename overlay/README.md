# overlay

`inference`または`postprocess`が出力したSQLiteと元動画から、確認用の
オーバーレイ動画を生成する独立リポジトリです。既存リポジトリのコードを
importせず、公開SQLite契約だけに依存します。

## 対応する4種類

1. `raw`: inference直後のAI生出力polygon mask
2. `tracked`: NMS、カット検出、tracking、短命track削除後のmask
3. `final`: 最終後処理後のmask。任意で顔検出boxを合成可能
4. `faces`: 顔・頭部検出boxだけ

## Setup

```bash
cd /home/kenshin/inference_backend2/overlay
python3 -m venv .venv
.venv/bin/pip install -e .
```

標準の`overlay-render`経路ではGPUを使用しません。動画のdecode、描画、encodeは
OpenCVのCPU処理です。C++/CUDA高速経路は後述の`experimental/`を
orchestrationから明示的に選択できます。

## Usage

### AI生出力マスク

```bash
overlay-render \
  --mode raw \
  --video input.mp4 \
  --sqlite after_inference.sqlite \
  --output output/raw.mp4
```

### 最低限の後処理後

`postprocess`のtracking stageが出力した`tracked.sqlite`を指定します。この
SQLiteの`masks`にはNMS、カット分割、tracking、短命track削除が反映されています。

```bash
overlay-render \
  --mode tracked \
  --video input.mp4 \
  --sqlite 04_tracking/tracked.sqlite \
  --output output/tracked.mp4
```

### 最終後処理後

```bash
overlay-render \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output output/final.mp4
```

顔検出も重ねる場合は、`face_detection`を含むunified inference SQLiteを追加します。

```bash
overlay-render \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --include-faces \
  --face-sqlite after_inference_with_faces.sqlite \
  --output output/final_with_faces.mp4
```

### 顔検出のみ

```bash
overlay-render \
  --mode faces \
  --video input.mp4 \
  --sqlite after_inference_with_faces.sqlite \
  --output output/faces.mp4
```

## Useful options

```text
--start-frame N          出力開始フレーム
--end-frame N            出力終了フレーム（含む）
--mask-alpha 0.32        マスク塗りの透明度
--no-labels              track ID、クラス、score表示を無効化
--codec mp4v             OpenCV FourCC
--codec h264             FFmpeg/libx264によるH.264出力
--codec h264_nvenc       NVIDIA NVENCによるH.264出力
--h264-crf 18            H.264品質（小さいほど高画質）
--h264-preset veryfast   H.264速度・圧縮率preset
--nvenc-cq 18            NVENC品質（小さいほど高画質）
--nvenc-preset p5        NVENC preset（p1最速、p7最高品質）
--target-bitrate-mbps N  CPU/NVENC共通の制約付き目標bitrate（CRF/CQより優先）
--manifest result.json   入力契約と処理結果をJSON保存
--overwrite              既存出力の置換を許可
```

非ASCIIのクラス名はOpenCV標準フォントで正しく描けないため、ラベル文字列を
省略してtrack IDを表示します。マスクの色と形状には影響しません。

現在の出力動画は映像のみで、元動画の音声はコピーしません。確認用overlayとして
元動画と同じFPS、幅、高さを維持します。

### H.264出力

```bash
overlay-render \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --codec h264 \
  --h264-crf 18 \
  --h264-preset veryfast \
  --output output/final_h264.mp4
```

H.264は`imageio-ffmpeg`に同梱されたFFmpeg/libx264を使用します。システムの
FFmpegを使う場合は`--ffmpeg-bin /path/to/ffmpeg`を指定できます。`mp4v`は従来どおり
OpenCVだけで出力されます。どちらのモードも音声は含みません。

### NVIDIA NVENC出力

```bash
overlay-render \
  --mode final \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --codec h264_nvenc \
  --nvenc-cq 18 \
  --nvenc-preset p5 \
  --output output/final_nvenc.mp4
```

NVENCには`h264_nvenc`を含むFFmpegとNVIDIA driverが必要です。既定では
`overlay/.runtime/ffmpeg-nvenc/bin/ffmpeg`、`PATH`、`imageio-ffmpeg`の順で
対応encoderを探索します。別のFFmpegは`--ffmpeg-bin`または
`OVERLAY_FFMPEG_BIN`で指定できます。

複数区間を同時encodeして最後に再encodeなしで連結する比較には、ベンチマーク
runnerを使用できます。

```bash
python benchmarks/benchmark_segmented.py \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/nvenc-3way \
  --ffmpeg-bin .runtime/ffmpeg-nvenc/bin/ffmpeg \
  --codec h264_nvenc \
  --workers 3
```

`--workers`には1以上の任意の並列数を指定できます。`--codec h264`ではCPU
x264並列、`--codec h264_nvenc`ではNVENC並列になります。CPUとNVENCを混成する
場合は、総worker数とCPU worker数を指定します。

```bash
python benchmarks/benchmark_segmented.py \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/hybrid-6way \
  --ffmpeg-bin .runtime/ffmpeg-nvenc/bin/ffmpeg \
  --codec hybrid \
  --workers 6 \
  --hybrid-cpu-workers 3
```

この例ではCPU x264を3 worker、NVENCを3 worker同時実行します。encoder種別は
動画区間の難易度が片側へ偏りにくいよう交互に割り当てます。分割方式はGOP境界が
増えるため、通常CLIとは分けて実験用として提供しています。各workerは担当区間の
開始フレームへ直接seekし、seekできないbackendだけ先頭から順次grabします。

同じ画質・容量を狙ってCPU/NVENCを比較する場合は
`--target-bitrate-mbps`を指定します。この指定時はlibx264とNVENCの両方を同じ
制約付きbitrateにし、NVENCはsingle-pass CBRを使います。`--cpu-weight`と
`--nvenc-weight`はworker 1個あたりの担当frame比で、値が大きい側へ多く割り当てます。

2026-07-26にCore Ultra 9 285K、RTX 5090、5290 framesの入力で測定した
ハードウェア固有の高速候補は次のとおりです。

```bash
# 1080p: 8 Mbps、CPU 3 + NVENC 5
python benchmarks/benchmark_segmented.py \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/fast-1080p \
  --ffmpeg-bin .runtime/ffmpeg-nvenc/bin/ffmpeg \
  --codec hybrid \
  --workers 8 \
  --hybrid-cpu-workers 3 \
  --h264-preset veryfast \
  --nvenc-preset p1 \
  --target-bitrate-mbps 8 \
  --cpu-weight 1 \
  --nvenc-weight 1.08

# 4K: 28 Mbps、CPU 7 + NVENC 3
python benchmarks/benchmark_segmented.py \
  --video input.mp4 \
  --sqlite predictions.sqlite \
  --output-dir output/fast-4k \
  --ffmpeg-bin .runtime/ffmpeg-nvenc/bin/ffmpeg \
  --codec hybrid \
  --workers 10 \
  --hybrid-cpu-workers 7 \
  --h264-preset ultrafast \
  --nvenc-preset p1 \
  --target-bitrate-mbps 28 \
  --cpu-weight 1.12 \
  --nvenc-weight 1
```

実測中央値は1080pが約455 fps、4Kが約135 fpsでした。worker最適値はCPU、
GPU、解像度、mask密度、ストレージで変わるため、別環境では近傍を再測定して
ください。4K測定は実動画を4K化し、同じmaskを座標変換した負荷再現用入力です。
モデルが実際に4Kで生成したmaskの精度を評価する試験ではありません。

2本の独立動画をCPU worker群とNVENC worker群で同時処理する比較には、次を使用
できます。`--video-b`と`--sqlite-b`を省略すると、同じ入力を独立した2ジョブとして
処理します。

```bash
python benchmarks/benchmark_two_videos.py \
  --video-a input-a.mp4 \
  --sqlite-a predictions-a.sqlite \
  --video-b input-b.mp4 \
  --sqlite-b predictions-b.sqlite \
  --output-dir output/two-videos \
  --ffmpeg-bin .runtime/ffmpeg-nvenc/bin/ffmpeg \
  --cpu-workers 3 \
  --nvenc-workers 3
```

FFmpeg直接経路、OpenCV/rawvideo経路、mask描画込みのI/O分解には
`benchmarks/benchmark_io_paths.py`を使用します。FFmpeg直接値はmask描画を含まない
下限値であり、完成overlayの速度としては扱いません。

詳しい入力境界は[CONTRACT.md](CONTRACT.md)を参照してください。

OpenCV/BGR/rawvideo pipeを使わないC++/libav/CUDAの速度上限試作は
[`experimental/`](experimental/)に分離しています。productionの主要4モード、
顔、ASCIIラベル、音声copyへ対応し、orchestrationの`experimental_cpp`
backendとして選択できます。
NVDECからcustom CUDA合成、NVENCまでGPU内で完結する経路も含みますが、任意codec/
pixel formatを含む完全なproduction置換ではありません。

## Test

テストは小さな動画と両SQLite schemaを一時生成するため、実モデル、GPU、
既存リポジトリを必要としません。

```bash
make test
```
