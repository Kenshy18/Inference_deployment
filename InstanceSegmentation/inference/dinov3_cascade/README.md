# DINOv3 Cascade standalone inference

このフォルダだけでDINOv3 Cascade、TensorRTバックボーン、ROI分類器の動画推論を実行します。

## Layout

- `infer.py`: 動画推論CLI
- `adapter.py`: DINOv3固有出力から共有セグメンテーション契約への変換
- `runtime.py`: モデル固有の1バッチ推論
- `setup_environment.py`: 同梱ソースの展開、依存確認、成果物検証
- `instance_segmentation/`: 検出・マスク推論とTensorRT処理
- `classifier/`: ROI分類
- `artifacts/`: 検証済み重みとTensorRTエンジン（Git管理対象外）
- `vendor/`: 固定したDetectron2/DINOv3ソースと共有推論層のスナップショット
- `.runtime/`: セットアップ時に生成される実行用ソースと環境ロック

## Run

```bash
python setup_environment.py --runtime-python /path/to/python
python infer.py --input input.mp4 --output detections.sqlite \
  --max-frames 1 --overwrite
```

依存パッケージも現在のPythonへ導入する場合は、セットアップに`--install`を付けます。
標準配置を使う限り、チェックポイント、バックボーン、TensorRTエンジン、フレームワークのパス指定は不要です。
結果はSQLiteだけに保存されます。

## TensorRT engine

`instance_segmentation/trt/`がDINOv3バックボーンのONNX export、engine作成、
実行をまとめて所有します。

```bash
python instance_segmentation/trt/export_backbone.py --help
python instance_segmentation/trt/build_engine.py \
  --onnx backbone.onnx --engine backbone.engine
```
