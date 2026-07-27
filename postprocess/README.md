# Mask postprocess

インスタンスセグメンテーションの生出力から最終マスクSQLiteまでを処理する、
スタンドアロンの後処理リポジトリです。推論リポジトリには依存しません。

実行時コードは次の3種類だけです。

```text
postprocess/
  CONTRACT.md              入出力・ステージ交換規約
  run_pipeline.py          唯一の一気通貫CLI
  contracts/               実装に依存しない型と成果物I/O
  common/                  設定、登録、パイプライン実行

  preprocessing/           正規化、スコアポリシー
  nms/                     NMS
  cut_detection/           カット検出
  tracking/                トラッキング
  approximation/
    ellipse/               楕円近似
    polygon/               ポリゴン近似
  keyframes/
    ellipse/               楕円キーフレーム
    polygon/               ポリゴンキーフレーム
  gap_fill/
    ellipse/               楕円マスク補完
    polygon/               ポリゴン補完
  evaluation/              品質評価
  artifacts/               最終SQLite生成・検証
  visualization/           可視化
```

各機能ディレクトリがアルゴリズムと、その機能をパイプラインへ接続する
`stages.py`を所有します。機能パッケージ同士はimportしません。データは
`contracts`で定義された名前付き成果物だけで受け渡されます。

## 実行

Python 3.10以上と、`pyproject.toml`に記載された依存関係が必要です。

```bash
python -m pip install -e .
```

AIの生出力JSONLからポリゴン後処理を行う例です。カット検出を有効にする場合、
入力動画も必要です。

```bash
python run_pipeline.py \
  --input-jsonl input/detections.jsonl \
  --input-video input/video.mp4 \
  --output-dir output/polygon \
  --shape-mode polygon
```

動画がなく、カットを検出しない場合:

```bash
python run_pipeline.py \
  --input-jsonl input/detections.jsonl \
  --output-dir output/polygon \
  --shape-mode polygon \
  --no-cut-detect
```

既存の追跡済みSQLiteから楕円後処理を開始する場合:

```bash
python run_pipeline.py \
  --input-sqlite input/tracked.sqlite \
  --output-dir output/ellipse \
  --shape-mode ellipse
```

未追跡の検出SQLiteと元動画から開始する場合:

```bash
python run_pipeline.py \
  --input-sqlite input/video_raw_detections.sqlite \
  --input-video input/video.mp4 \
  --output-dir output/ellipse \
  --shape-mode ellipse
```

`--input-sqlite`は次の形式を自動判別します。

- `masks.track_id`を持つ追跡済みSQLite
- `metadata`、`frames`、未追跡の`masks.mask_id`を持つDINOv3
  `raw_mask_sqlite_v1`
- `segmentation_polygons`と`segmentation_points`を持つ
  InstanceSegmentation unified inference schema v2/v3

未追跡SQLiteの場合だけ、動画をカット検出に使用した後、スコア方針、NMS、
トラッキングから実行します。

インストール後は、同じCLIを`postprocess`コマンドでも実行できます。モデルの
配置を変える場合は`--model-root`または`POSTPROCESS_MODEL_ROOT`を使います。

出力ディレクトリには各ステージのサブディレクトリ、最終成果物、および
`pipeline_manifest.json`が生成されます。manifestには使用実装、要求・提供
成果物、実行時間、全成果物のパスが記録されます。

raw入力からカット検出を実行した場合、最終`predictions.sqlite`には
`cuts`（新シーン先頭フレーム）と`cut_detection_metadata`（方式、所要時間、
件数、フレーム意味論）が保持されます。マスク変換後にも件数とメタデータの
整合性を最終validatorが確認します。

旧`Dinov3_postprocess`の`AI後処理最終.sqlite`を読むソフトウェア向けには、
現行成果物を残したまま互換SQLiteを追加出力できます。

```bash
python run_pipeline.py \
  --input-sqlite input/inference.sqlite \
  --input-video input/video.mp4 \
  --output-dir output/postprocess \
  --export-legacy-sqlite
```

`pipeline_manifest.json`の`legacy_predictions_sqlite`が互換成果物です。
このSQLiteは旧契約と同じ`masks`、`tracks`、`cuts`だけを持ちます。旧形式にも
カット位置`cuts(frame)`は含まれますが、現行の監査用`raw_tracked_masks`、
`raw_tracks`、`cut_detection_metadata`は含めません。現行
`predictions_sqlite`が正本であり、互換版は追加成果物です。

変換だけを行うtentative CLIもあります。

```bash
python -m tentative.export_legacy_sqlite \
  --input-sqlite output/postprocess/current.sqlite \
  --output-sqlite output/postprocess/legacy.sqlite
```

## モジュールの交換・追加

標準構成は`configs/pipelines/*.json`で確認できます。`--pipeline-config`に
別JSONを指定すると、ステージの順序、実装、オプションを変更できます。
pipeline JSONの値が基準値になり、CLIで該当オプションを明示した場合だけ
上書きされます。

新実装は`contracts.PostprocessStage`を満たし、`requires`で要求成果物、
`provides`で生成成果物を宣言します。runnerは既知成果物のschema、未宣言出力、
既存成果物の上書き、stageディレクトリ外への出力も検査します。設定の
`implementation`には組み込み名、
または`python.module:ClassName`を指定できます。

```json
{
  "id": "cut_detection",
  "implementation": "my_cut.detector:NewCutStage",
  "options": {"threshold": 0.8}
}
```

この場合、前段の`nms_jsonl`を受け取り、`cuts_json`を返す契約を維持すれば、
前後の実装を変更する必要はありません。詳しい規約は[CONTRACT.md](CONTRACT.md)、
構造の理由と追加手順は[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)を参照して
ください。

## 検証

```bash
make test
make smoke
```

`make smoke`はカット検出を別実装へ交換した状態で、生JSONLから最終SQLiteまで
実行します。
