# EVA-02 Cascade standalone inference

このフォルダだけでEVA-02 CascadeとROI分類器の動画推論、および
PyTorchチェックポイントからTensorRTエンジンの再生成を行います。

## Layout

- `infer.py`: 動画推論CLI
- `adapter.py`: EVA-02固有出力から共有セグメンテーション契約への変換
- `runtime.py`: モデル固有の1バッチ推論
- `setup_environment.py`: 同梱フレームワークの展開、依存確認、成果物検証
- `instance_segmentation/`: 検出・マスク推論
- `classifier/`: ROI分類
- `trt/`: ONNX export、TensorRT build、PyTorchとの差分検証、bundle読込
- `artifacts/`: 検証済みチェックポイント（Git管理対象外）
- `vendor/`: 固定したDetectron2ソースと共有推論層のスナップショット
- `.runtime/`: セットアップ時に生成される実行用ソースと環境ロック

## Setup and engine build

```bash
python setup_environment.py --runtime-python /path/to/python
python trt/build_engine.py \
  --runtime-python /path/to/python
```

依存パッケージも現在のPythonへ導入する場合は、セットアップに`--install`を付けます。
標準配置を使う限り、チェックポイントやフレームワークのパス指定は不要です。

`build_engine.py`は次の処理を別プロセスで順番に実行します。

1. `model_final.pth`から、枝刈り済みEVA-02 ViTを動的batch ONNXへexport
2. batch 1/12/20、1280×1280、FP16のTensorRT engineをbuild
3. 同じcheckpointのPyTorch出力とbatch 1/12/20で数値比較
4. 検証合格時だけ、hash付きmanifestとengineをatomicに公開

既存の出力先は上書きしません。再生成時は古いbundleを退避または削除してから
実行してください。ONNXは中間生成物であり、checkpointから再生成できるため
最終bundleには保存しません。

## Run

```bash
# 高速版（既定）
python infer.py \
  --backend tensorrt-backbone \
  --input input.mp4 \
  --output detections.sqlite

# 元の安定版
python infer.py \
  --backend pytorch \
  --input input.mp4 \
  --output detections.sqlite
```

どちらも同じ前処理、Cascade ROIヘッド、ROI分類器、SQLite schemaを使います。
TensorRT化する範囲は枝刈り済みEVA-02 ViTバックボーンだけです。
SimpleFeaturePyramid、Cascade Mask R-CNNのproposal/ROI/maskヘッド、
ROI分類器はPyTorchで動作します。したがってこれはEVA-02全体の
純TensorRT化ではなく、精度差を狭く保ちながら重いバックボーンを高速化する
hybrid backendです。

## Verified baseline

2026-07-24、RTX 5090、TensorRT 10.13、1280×1280で検証した基準値です。

- TensorRT backbone検証: batch 1/12/20の全条件で合格
- PyTorch backboneとのcosine類似度: 0.999986以上
- 1800フレームの動画＋分類器＋SQLite: 22.13 fps
- 同じ現行コードのPyTorch版: 9.85 fps（700フレーム）
- 比較区間640検出: フレーム別検出件数・分類クラスが全一致
- SQLite: 1611検出すべてに分類結果と3クラス確率、maskは最大150点

速度はGPU、TensorRT/PyTorch/CUDA、動画内容、I/Oで変動します。上記は
engine単体の理論性能ではなく、動画decode、Cascade head、分類器、
SQLite保存を含むend-to-end値です。
