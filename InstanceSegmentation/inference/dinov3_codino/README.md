# DINOv3 Co-DINO standalone inference

このフォルダだけで、同一のCo-DINOチェックポイントを次の2方式で実行できます。
どちらも動画を入力し、共通契約に変換したインスタンスセグメンテーション結果を
SQLiteだけに保存します。

- `tensorrt-fast`（既定）: RTX 5090 / SM120向けの25 fps級最適化版
- `pytorch`: TensorRTを使わない、元の安定したPyTorch実装

## Layout

- `infer.py`: 2バックエンドを選択する動画推論CLI
- `model.py`: 共通のCo-DINOモデル構築。PyTorch版はここから直接実行
- `preprocessing.py`, `postprocessing.py`, `classifier.py`: 両方式で共有する処理
- `adapter.py`: PyTorch固有出力から共有セグメンテーション契約への変換
- `optimized/`: fast版の固定B2実行、CUDA Graph、先読み、core/tail並列化
- `trt/runtime.py`: 4 engineを一体として組み込むTensorRTランタイム
- `trt/sm120_msda.py`: fast query engineに必要なSM120 native plugin登録
- `trt/build_fast_engines.py`: checkpointからRTX 5090向けfast bundleを一括生成
- `trt/build_runtime_checkpoint.py`: optimizerとTensorRT置換済みmoduleを除いた
  実行専用checkpointを生成
- `trt/fast_engine_build.py`, `trt/native/`: 精度制約付きTensorRT buildとSM120 CUDA実装
- `trt/build_engines.py`: portableな4 engineを一括生成する保守用入口
- `trt/bundle.py`: profile、engine、plugin、サイズ、SHA-256の検証
- `setup_environment.py`: 同梱ソースの展開、依存確認、fast bundleの完全検証
- `artifacts/`: 設定、チェックポイント、分類器、検証済みfast bundle
- `vendor/`: 固定したCo-DINO/DINOv3ソースと共有推論層
- `.runtime/`: セットアップ時に生成されるソースと環境ロック

`optimized/`は高速実行のまとまり、`trt/`はengineの読込・作成・検証のまとまりです。
4つのTensorRT partitionを独立したモデルとして公開せず、必ず1つのCo-DINO
バックエンドとして扱います。

## Setup

既に必要な依存関係を持つPythonを指定します。

```bash
python setup_environment.py --runtime-python /path/to/python
```

依存パッケージもそのPythonへ導入する場合は`--install`を付けます。セットアップは
同梱ソースを`.runtime/`へ展開し、チェックポイント、4 engine、SM120 pluginを
SHA-256まで検証します。

## Optimized TensorRT

```bash
/path/to/python infer.py \
  --backend tensorrt-fast \
  --input input.mp4 \
  --output codino_fast.sqlite \
  --overwrite
```

fast版は固定batch 2、入力tensor 736x1280（有効画像720x1280）です。4 engine、
native plugin、CUDA Graph、2つの出力slot、worker CUDA stream、bounded
decode/preprocessを一体として使用します。起動時はbundle内の実行専用checkpointと
軽量deployment shellを使い、直後に置換するViT-L backboneや学習用補助headを
構築しません。分類器も最適化スケジュールに含まれるため無効化できません。

同梱bundleはRTX 5090 / compute capability 12.0用です。他GPUや別batch sizeでは
実行せず、その環境向けに検証されたbundleが必要です。

## Stable PyTorch

```bash
/path/to/python infer.py \
  --backend pytorch \
  --batch-size 1 \
  --input input.mp4 \
  --output codino_pytorch.sqlite \
  --overwrite
```

PyTorch版は同じconfig、検出checkpoint、分類器manifestを直接読み込みます。
TensorRT engineやSM120 pluginには依存しないため、fast版の比較基準および安全な
fallbackとして維持します。GPUメモリに余裕があればbatch sizeは変更できます。

## Performance

2026-07-23、RTX 5090、1920x1080 / 29.97 fps動画で測定した値です。

| backend | frames | inference loop | measured compute | detections |
|---|---:|---:|---:|---:|
| `tensorrt-fast` | 300 | 19.92 fps | 24.43 img/s | 267 |
| `tensorrt-fast` | 1,800 | 23.08 fps | 23.94 img/s | 1,636 |
| `tensorrt-fast`（deployment shell） | 5,290 | 23.29 fps | 23.58 img/s | 4,069 |
| `tensorrt-fast`（checkpointから再生成） | 300 | 18.35 fps | 22.48 img/s | 267 |
| `pytorch` | 10 | 2.20 fps | 3.16 img/s | 0 |

2026-08-01のepoch 6検出器とbackbone ROI分類器への更新後、同一動画・同一
1,500フレームの比較では旧版23.36 img/s、新版22.34 img/sでした。検出数は
814から822へ変化しています。検出後処理を除く固定core 150反復では旧24.15、
新24.06 img/sで、モデル本体の差は0.4%です。

「25 fps」は主にGPU計算区間の目標値です。動画デコード、起動時のモデル構築、
契約変換、SQLite保存を含むプロセス全体で常に25 fpsを保証する意味ではありません。
入力動画、検出数、ストレージ、warmup、GPU状態によって変動します。
再生成版は、同じマシン状態で既存fast bundleと比較した元実装の検証でも計算速度差
0.62%でした。上表の測定セッションでは両者とも過去の25.15 img/s記録を下回ったため、
engine生成だけで25 img/sを保証せず、動画ごとの品質・速度gateを必須とします。

同じ5,290フレームでは、deployment shell導入前後でプロセス全体が
252.10秒から244.99秒へ2.8%短縮し、最大RSSは6,946,724 KiBから
3,896,916 KiBへ43.9%減少しました。定常推論の数値処理やTensorRT engineは
変更していません。

`--fast-sqlite`は異常終了時の耐久性と引き換えにSQLite書込みを速くします。
通常は指定しないでください。

## TensorRT engine maintenance

### RTX 5090 fast bundle

`trt/build_fast_engines.py`が、checkpointから25 fps級の構成をクリーン再生成する
正式な入口です。次を1コマンドで行います。

1. backbone、query encoder、decoder、mask headの固定shape ONNXをexport
2. optimizerと置換対象weightを除いた実行専用checkpointを生成
3. SM120専用MSDA CUDA extensionを同梱ソースからコンパイル
4. 空のtiming cacheから4つのTensorRT engineを別プロセスでbuild
5. engine、plugin、実行専用checkpoint、元checkpoint、分類器、実行Python、
   builder sourceをSHA-256検証
6. 完全なbundleだけをatomicに公開

RTX 5090（compute capability 12.0）、TensorRT 10.13、CUDA compiler、G++、Ninjaが
必要です。分類器manifestもbundleの再現性に含めるため必須です。既存engineや
既存timing cacheは入力として受け取りません。出力先には存在しないディレクトリを
指定します。

```bash
/path/to/python trt/build_fast_engines.py \
  --runtime-python /path/to/python \
  --config artifacts/detector/resolved_config.py \
  --checkpoint artifacts/detector/teacher_vitl_codino_epoch6_deploy.pth \
  --classifier-checkpoint artifacts/classifier/backbone/manifest.json \
  --environment-lock .runtime/environment-lock.json \
  --output-dir /path/to/new-fast-sm120-fixed-b2-v1
```

生成したbundleは、同梱bundleと同じ推論CLIへ明示的に渡せます。

```bash
/path/to/python infer.py \
  --backend tensorrt-fast \
  --trt-bundle /path/to/new-fast-sm120-fixed-b2-v1/manifest.json \
  --trt-verify full \
  --input input.mp4 \
  --output rebuilt_fast.sqlite \
  --overwrite
```

manifestの`production_registered`と`e2e_25fps_claimed`は、生成直後は`false`です。
新しいcheckpointやengineは、PyTorch版との結果比較と実動画での長尺速度測定を
通してから本番bundleと置き換えてください。

### Portable bundle

`trt/build_engines.py`は、一般的な環境で保守しやすいportable profileを生成します。
portable engineを生成する場合も、4 engineは一括で作成します。

```bash
/path/to/python trt/build_engines.py \
  --runtime-python /path/to/python \
  --config artifacts/detector/resolved_config.py \
  --checkpoint artifacts/detector/teacher_vitl_codino_epoch6_deploy.pth \
  --classifier-checkpoint artifacts/classifier/backbone/manifest.json \
  --environment-lock .runtime/environment-lock.json \
  --output-dir artifacts/trt/new-portable-fixed-b2-v1
```

portable bundleは`--backend tensorrt-fast`には渡せません。公開する実行方式は
検証済みfast版とPyTorch版の2つに限定しています。

## Output

推論結果はSQLiteだけです。主なテーブルは`frames`、`detections`、
`classification_probabilities`、`segmentations`、`segmentation_polygons`、
`segmentation_points`です。bundle内の`manifest.json`は推論結果ではなく、
engine・pluginの混在や破損を防ぐ内部管理情報です。
