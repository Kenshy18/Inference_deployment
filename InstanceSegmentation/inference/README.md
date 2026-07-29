# Inference architecture

`inference/`は、モデル固有実装を共通の入出力契約へ接続する層です。モデルの
交換や最適化では、利用側のパイプラインとの後方互換を維持します。

## Unified run

公開入口は`run_inference.py`です。動画を入力し、次の3モードを同じSQLite
schema v3で出力します。既存schema v2は読み取り互換として残します。

```bash
# インスタンスセグメンテーション＋分類器
python run_inference.py \
  --mode segmentation \
  --segmentation-model dinov3_codino \
  --input input.mp4 \
  --output result.sqlite

# インスタンスセグメンテーション＋分類器＋顔検出
python run_inference.py \
  --mode segmentation-face \
  --segmentation-model eva02_cascade \
  --input input.mp4 \
  --output result.sqlite

# 顔検出
python run_inference.py \
  --mode face \
  --input input.mp4 \
  --output result.sqlite
```

選択可能なセグメンテーションモデルは`dinov3_codino`、
`dinov3_codino_mh0`、`dinov3_cascade`、`eva02_cascade`です。
Co-DINOとcompact Co-DINO MH0では
`--segmentation-backend tensorrt-fast|pytorch`、EVA-02では
`--segmentation-backend tensorrt-backbone|pytorch`を選択できます。
顔検出は既定でRT-DETRの`Face`と`Head`を保存します。第2モデルとして
`--face-model face_dino_v2`を選択でき、DINOv3 ViT-S+、compact Co-DINO、
Face楕円・キーポイント・occlusion branchの固定B8 TensorRT推論を利用できます。
RT-DETRで`--face-classes`へ値を渡さなければ、`VisibleBody`を含む全クラスを
保存します。`face_dino_v2`が持つクラスは`Head`と`Face`です。

各`<model>/infer.py`はモデルフォルダ単体の保守・検証用入口として残します。
統一pipelineはモデルを隔離プロセスで実行し、成功した全結果だけを最後に
1つのSQLiteへatomicに公開します。これによりDetectron2、Co-DINO、RT-DETRの
依存関係とGPU初期化を上位層で混在させません。`segmentation-face`では現在、
安全なモデル分離を優先して動画をモデルごとに読み込みます。既定は逐次実行です。
`dinov3_codino_mh0`と`face_dino_v2`の組合せに限り、
`--parallel-models`で両モデルを同時実行できます。巨大`dinov3_codino`、
旧顔検出、片方だけの推論ではこのオプションを指定できません。
GPUの電力上限で完全同時実行が遅くなる環境では、
`--parallel-model-stagger-seconds N`により顔モデルを先に起動してピーク競合を
調整できます。

Face DINO v2は既定の固定B8 bundleに加え、build済みmanifestを
`--face-trt-bundle /path/to/manifest.json`で選択できます。B16 profileは
`trt/build_engines.py --batch-size 16`で構築できますが、B8と数値が完全一致する
保証はないため、精度評価後に明示選択してください。

## 処理の境界

```text
run_inference.py
    -> orchestration/pipeline.py
    -> registered standalone model process(es)
    -> DetectionFrame | SegmentationFrame SQLite
    -> unified schema-v3 SQLite
```

- `contracts/`: フレーム、分類、物体検出、インスタンスセグメンテーションの
  安定した入出力定義
- `video/`: モデル非依存の動画メタデータ取得とデコード
- `persistence/`: 契約オブジェクトからSQLiteへの保存
- `pipelines/`: デコード、推論、保存を接続するタスク非依存の制御
- `orchestration/`: モード選択、モデルプロセス起動、atomic統合
- `registry.py`: モデルID、タスク、Adapterの軽量な登録情報
- `<model>/adapter.py`: モデル固有値を共通契約へ変換する唯一の境界
- `<model>/infer.py`: 引数解釈、モデル構築、共通パイプライン呼び出しだけを行うCLI

モデル固有の前処理、TensorRT、分類器、後処理は各モデルフォルダ内に閉じます。
共有化は、2モデル以上で同じ意味と変更理由を持つことが確認できた処理に限ります。

## 契約

入力画像はデコード済みの`Frame`です。座標は元動画の画素座標、矩形は
half-openの`[x1, y1, x2, y2)`、スコアは`0.0`から`1.0`です。結果は入力と同じ
フレーム順・件数で返します。

- インスタンスセグメンテーション:
  `SegmentationFrame`。検出矩形、検出スコア、任意の分類結果、元画像座標の
  polygon maskを持ちます。
- 顔・頭部検出:
  `DetectionFrame`。従来モデルは検出矩形とスコアを持ちます。
  `face_dino_v2`では同じHead/Face検出行に加え、HeadとFaceの対応、顔確率、
  楕円、64×64確率マスク、5点キーポイント、visible/occluded、validと各確率を
  `FaceObservation`として持ちます。追跡は現在の推論範囲に含めません。
- 分類:
  現時点では独立パイプラインではなく、各検出の任意フィールド
  `Classification`として保持します。

契約の破壊的変更では`CONTRACT_VERSION`を更新します。モデル内部の変更だけなら
更新しません。

## モデルの交換・追加

既存モデルを最適化するときは、そのモデルの`adapter.py`より内側だけを変更します。
たとえばRT-DETRをYOLOへ交換するときは、同じ`DetectionFrame`を返すAdapterを
追加すれば、動画デコードとSQLite出力を再利用できます。

新規モデルは次の順に追加します。

1. タスクに対応するAdapter protocolを実装する
2. `registry.py`へモデルIDを登録する
3. 1フレームの実推論と契約テストを通す
4. モデルフォルダだけを別の場所へコピーし、`setup_environment.py`実行後に
   同じ推論を通す

各モデルの`vendor/inference_common.tar.gz`には、この共有層のスナップショットが
含まれます。これによりモデルフォルダ単体でも`.runtime/shared`へ展開して実行
できます。

## Unified SQLite output

CLIの`--output`はSQLiteファイルのパスです。既存ファイルは誤って追記せず、
`--overwrite`を明示した場合だけ置き換えます。

- `schema_info`: schema名とversion
- `videos`, `runs`: 入力動画と実行モード
- `run_metadata`: 統一CLIの実行設定
- `model_executions`: モデル、backend、役割、タスク
- `model_metadata`: 各モデル固有の設定
- `frames`: モデルに依存しないフレーム番号、時刻、元動画サイズ
- `detections`: 実行モデル、クラス、スコア、元動画座標の矩形
- `classifications`: セグメンテーション検出に対応する分類器結果
- `classification_probabilities`: 分類器のクラス別確率
- `segmentations`: 検出とマスクの対応
- `segmentation_polygons`: マスク内のpolygon
- `segmentation_points`: polygonを構成する元動画座標の点
- `face_observations`: Head/Face行の対応、顔有無、顔スコア、正確な楕円
- `face_masks`: 元画像上の対応boxと`zlib-u8-probability-v1`顔確率マスク
- `face_keypoints`: 5点の座標、意味クラス、visible/occluded、valid、confidence
- `face_keypoint_class_probabilities`: background/eye/nose/mouth確率
- `face_keypoint_state_probabilities`: occluded/visible確率

schema v3でも従来の`detections`とFace外接boxは残るため、schema v2相当のreaderは
列追加を許容すれば従来表示を継続できます。rich情報を利用するreaderは
`face_observations`からHeadとFaceを対応付けます。顔楕円の角度はradian、
座標と半径は元動画画素座標です。顔マスクは0〜255へ量子化した確率をzlib圧縮し、
`face_masks`のboxへ写像します。

モデル内部ではバイナリマスクを生成し、SQLite保存直前に
`CHAIN_APPROX_SIMPLE`で輪郭polygonへ変換してから形状を簡略化します。
輪郭近似方式はモデル別のオプションにせず、全モデルでこの処理に統一します。
1検出マスクに属する全polygonの頂点数は合計150点以下です。複数の輪郭がある場合も
各輪郭150点ではなく、マスク全体で150点の予算を分配します。

全モードで全テーブルを作成します。使用しない結果テーブルが空になるだけなので、
読む側はモードによってschemaを切り替える必要がありません。各検出は
`model_execution_id`を持つため、同じフレームのセグメンテーションと顔検出を
区別できます。

分類器を持つセグメンテーションモデルでは、`detections.score`に検出器のスコア、
`classifications.score`に分類器が選択したクラスの最大確率、
`classification_probabilities`に全クラスの確率分布を保存します。

JSON/JSONLの推論結果は生成しません。既定では異常終了時の耐久性を優先します。
`--fast-sqlite`を指定した場合だけ、耐久性と引き換えに保存速度を優先します。
モデル個別SQLiteから最終SQLiteへの公開は引き続き一時ファイルからのatomic
renameで行われます。

## Test

共有層のテストにはGPUやチェックポイントは不要です。

```bash
PYTHONPATH=inference python3 -m unittest discover -s inference/tests -v
```
