# DINOv3 ViT-S+ + compact Co-DINO MH0

`dinov3_codino_mh0`は、DINOv3 ViT-S+/16 backboneとcompact Co-DINO
MH0 headを使う1クラスのインスタンスセグメンテーションモデルです。
モデル固有コード、PyTorchチェックポイント、TensorRT runtime、再ビルド手順を
このフォルダ内に閉じています。`tentative_folder`や
`inference/dinov3_codino`を実行時に参照しません。

## Backends

- `tensorrt-fast`: RTX 5090（SM120）、TensorRT 10.13、736x1280、
  固定B16。既定値です。
- `pytorch`: 同じcheckpointを使う固定B2の品質参照・フォールバックです。

TensorRT版は次の4区間と専用MSDA CUDA pluginで構成されます。

1. BF16 ViT-S+ backbone + P2-P5 SFP neck
2. mixed-FP16 query encoder
3. FP32 decoder
4. FP32 N16 mask refinement head

未使用P6の削除、固定query幾何のキャッシュ、バッチbbox後処理、
ROI-local semantic convolution、union-cropped mask pasteに加えて、
BGR letterbox・RGB変換・ImageNet正規化を1 kernelで行うCUDA pluginを
含みます。モデル推論中には、前バッチのmask polygon化と共通契約への変換を
1 workerで重ねます。キューは有界で、SQLiteへの書き込み順序は元フレーム順を
維持します。

## Repository-wide inference

共通CLIは動画を読み、後方互換性のある共通schema v3のSQLiteを生成します。

```bash
cd inference
python run_inference.py \
  --mode segmentation \
  --segmentation-model dinov3_codino_mh0 \
  --segmentation-backend tensorrt-fast \
  --input /path/to/input.mp4 \
  --output /path/to/result.sqlite
```

モデル単体のCLIも同じSQLite契約を使用します。

```bash
python infer.py \
  --backend tensorrt-fast \
  --input /path/to/input.mp4 \
  --output /path/to/result.sqlite \
  --fast-sqlite
```

PyTorch参照:

```bash
python infer.py \
  --backend pytorch \
  --input /path/to/input.mp4 \
  --output /path/to/reference.sqlite
```

## Three-class classifier

MH0 detector自体は1クラス`foreground`です。その後、mask ROIの
`192x14x14`特徴をMH0専用Spatial-GAP分類器へ渡し、
`女性器`、`男性器`、`結合部分`へ分類します。検出器クラスと分類器クラスは
SQLite内で分離され、既存の共通schema v3を変更しません。

既定checkpointは次です。

```text
artifacts/classifier/best.pt
```

現在のローカルartifactは接続確認用の暫定重みです。巨大Co-DINO分類器の
Spatial-GAP構造と学習済みブロックを維持し、256チャネル入力射影だけを
192チャネルへ決定論的に変換しています。分類精度は未検証です。学習後は、
同一のクラス契約・入力shape・checkpoint schemaで作成した`best.pt`へ
置き換えるだけで、CLI、SQLite schema、後処理を変更せず差し替えられます。

暫定artifactを再生成する場合:

```bash
python tools/create_bootstrap_classifier.py --overwrite
```

## Artifacts

既定checkpoint:

```text
artifacts/detector/video_pseudo_mh0_epoch6_ema_deploy.pth
SHA-256 755345eb1d5d252f630371c7d7895397e03a6aca0ee7f3e3d323d0b7bdc0bd9c
```

これはepoch 6のEMA tensorsを選び、optimizerを除いたdeploy checkpointです。
重みの量子化はしていません。元checkpointからの変換情報は
`artifacts/detector/checkpoint_provenance.json`にあります。

TensorRT bundle:

```text
artifacts/trt/fast-sm120-fixed-b16-v1/manifest.json
```

起動時にはengine/pluginのサイズとSHA-256を検証します。engine、checkpoint、
shared objectは`.gitignore`対象なので、モデルフォルダを配布するときは別途
artifact storeまたは同梱コピーが必要です。

## Environment

vendor archiveからCo-DINO、DINOv3、共通推論契約を展開します。

```bash
python setup_environment.py --extract-only
```

依存パッケージもインストールする場合:

```bash
python setup_environment.py --python /path/to/python3.10
```

MMCV 1.7.2はCUDA opsを有効にしたビルドが必要です。検証環境は
Python 3.10.18、PyTorch 2.11.0+cu129、CUDA 12.9、
TensorRT 10.13.0.35、RTX 5090です。

## Convert the original training checkpoint

The training checkpoint produced by `ExpMomentumEMAHook` stores the evaluated
EMA weights in the ordinary model keys and the swapped raw weights in `ema_*`
buffers. Convert it to a compact deploy checkpoint first:

```bash
python tools/convert_ema_checkpoint.py \
  /path/to/video_pseudo_mh0_epoch6.pth \
  artifacts/detector/video_pseudo_mh0_epoch6_ema_deploy.pth
```

The converter removes optimizer state and `ema_*` backup buffers, normalizes
`torch.compile` `_orig_mod` keys, omits known non-persistent tensors, and records
the source SHA-256. It rejects a checkpoint whose EMA provenance cannot be
established.

## Rebuild TensorRT from checkpoint

新しい出力先を指定すると、checkpointからONNX export、SM120 plugin compile、
4 engine build、hash付きmanifest生成まで一括実行します。

```bash
python trt/build_fast_engines.py \
  --runtime-python /path/to/python3.10 \
  --config artifacts/detector/resolved_config.py \
  --checkpoint artifacts/detector/video_pseudo_mh0_epoch6_ema_deploy.pth \
  --output-dir artifacts/trt/fast-sm120-fixed-b16-rebuild
```

生成bundleを検証して使うには:

```bash
python infer.py \
  --backend tensorrt-fast \
  --trt-bundle artifacts/trt/fast-sm120-fixed-b16-rebuild/manifest.json \
  --input /path/to/input.mp4 \
  --output /path/to/result.sqlite
```

TensorRT engineはTensorRT versionとGPU architectureに依存します。異なるGPUでは
engineをその環境で再生成してください。ビルド後はPyTorch版との動画parity検証を
実施してから本番採用します。

## Validated performance

RTX 5090、1920x1080、5,290フレームの動画では、CUDA融合前処理と
CPU契約変換のoverlapを有効にしたモデル推論部分が149.734 FPS、
decode・polygon化・SQLite enqueueを含む推論ループが146.201 FPSでした。
プロセス起動とengine検証を含む全体は44.15秒です。
固定入力100 iterationでは153.452 images/sです。
動画全域から5フレーム間隔で1,058フレームをPyTorch版と比較した結果:

- detection-count agreement: 98.866%
- mean bbox IoU: 0.987689
- mean mask IoU: 0.988174

この比較はPyTorchとの数値parityであり、教師ラベルに対するmAPではありません。

上記5,290フレームでは最適化前の同一出力SQLiteと全テーブル・全行が一致し、
プロセス全体は46.67秒から44.15秒へ5.4%短縮しました。
