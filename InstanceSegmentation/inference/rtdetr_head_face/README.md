# RT-DETR Head/Face standalone inference

このフォルダだけでRT-DETR Head/Face動画推論を実行し、結果をSQLiteへ保存します。

## Layout

- `infer.py`: 動画推論CLI
- `adapter.py`: RT-DETR固有出力から共有物体検出契約への変換
- `model.py`: モデル構築と1バッチ推論
- `preprocessing.py`: letterboxと入力tensor作成
- `postprocessing.py`: スコア、クラス、面積、NMSによる検出絞り込み
- `setup_environment.py`: 同梱フレームワークの展開、依存確認、成果物検証
- `artifacts/`: 検証済みチェックポイント（Git管理対象外）
- `vendor/`: 固定したRT-DETRv4ソース、ネイティブ依存、共有推論層のスナップショット
- `.runtime/`: 実行用ソース、任意のローカル依存、環境ロック

## Run

```bash
python setup_environment.py --runtime-python /path/to/python
python infer.py --input input.mp4 --output detections.sqlite \
  --max-frames 1 --warmup-iterations 0 --overwrite
```

依存を現在のPythonへ導入する場合は`--install`、フォルダ内の
`.runtime/site-packages`へ導入する場合は`--install-local`を使います。
結果はSQLiteだけに保存されます。現在はPyTorchで検出までを行い、追跡や
顔検出後の時系列処理は行いません。このモデルフォルダには、推論から利用されない
名目上だけのTensorRT engine builderは置いていません。
