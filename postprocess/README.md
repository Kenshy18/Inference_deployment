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
  classwise/               クラス別キーフレーム間隔ルーティング
  face_privacy/            顔楕円・Eye点から顔/目マスクを生成・統合
  evaluation/              品質評価
  artifacts/               最終SQLite生成・検証
  production/              昇格済み構成、CPU厳密DP、成果物統合
  visualization/           可視化
```

各機能ディレクトリがアルゴリズムと、その機能をパイプラインへ接続するstageを
所有します。後処理の正本は`production/polygon/`に集約され、NMS、入力幾何、
頂点方針、DP、pair-vote、topology検証、成果物materializeを別モジュールに分離
しています。データは`contracts`で定義された名前付き成果物だけで受け渡されます。

GUI用Liveプレビューは`common/live_preview.py`の単一非同期workerへ、各stageが
フレーム番号と軽量な図形だけを通知します。元動画のデコード・960×540描画・JPEG
生成はアルゴリズムの実行スレッド外で行い、ステージごとに最新1件だけを保持します。
構造化進捗はstage内経過秒を含む独立したheartbeatを出すため、画像が静止する
最適化stageでも停止と処理継続を区別できます。環境変数
`MASK_PIPELINE_PREVIEW_PATH`がない場合は完全に無効です。

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
  --output-dir output/polygon
```

動画がなく、カットを検出しない場合:

```bash
python run_pipeline.py \
  --input-jsonl input/detections.jsonl \
  --output-dir output/polygon \
  --no-cut-detect
```

既存の追跡済みSQLiteから開始する場合:

```bash
python run_pipeline.py \
  --input-sqlite input/tracked.sqlite \
  --output-dir output/postprocess \
  --keyframe-interval 6
```

未追跡の検出SQLiteと元動画から開始する場合:

```bash
python run_pipeline.py \
  --input-sqlite input/video_raw_detections.sqlite \
  --input-video input/video.mp4 \
  --output-dir output/postprocess
```

`--input-sqlite`は次の形式を自動判別します。

- `masks.track_id`を持つ追跡済みSQLite
- `metadata`、`frames`、未追跡の`masks.mask_id`を持つDINOv3
  `raw_mask_sqlite_v1`
- `segmentation_polygons`と`segmentation_points`を持つ
  InstanceSegmentation unified inference schema v2/v3

未追跡SQLiteの場合だけ、動画をカット検出に使用した後、スコア方針、NMS、
トラッキングから実行します。

ポリゴン構成の既定Productionは2026-08-15から次の構成です。

- `nms.production_v3`: 全穴埋め、所有本体比1%以下の島削除、仮想連結成分、
  Mask版Adaptive NMS、島対本体80%/50%判定
- `production.polygon_v3_cpu`: トラック面積に応じた14/16/18/20頂点ポリゴン、
  最小Recall制約付き多状態DP、
  2 sweep pair-vote、全補間フレームtopology検査、CPU `native_exact`区間評価
- 既定の努力目標キーフレーム間隔は6。`--keyframe-interval`で変更可能

旧NMS・旧ポリゴン実装と候補比較はGit履歴と保存済み監査成果物だけに残し、
開発ソースツリーからも撤去しました。本番経路は`nms.production_v3`と
`production.polygon_v3_cpu`だけです。未対応ラベルや凍結契約と異なる設定で旧実装へ
暗黙に戻ることはありません。昇格版の責務と凍結条件は
`production/README.md`を参照してください。

高精度カット検出は、連続したゼロ始まりのフレームではFFmpegで96×54へ直接
縮小デコードします。検出ロジックと閾値は従来と同じです。別処理と重ねる場合は
先に契約済みJSONを作り、後処理へ渡せます。

```bash
python precompute_cuts.py \
  --input-video input/video.mp4 \
  --output output/cuts.json

python run_pipeline.py \
  --input-sqlite input/inference.sqlite \
  --input-video input/video.mp4 \
  --precomputed-cuts-json output/cuts.json \
  --output-dir output/postprocess
```

`--max-frames`はprecompute側の上限です。動画が先に終了した場合は実フレーム数で
正常終了します。`cuts.json`は通常のカット検出stageと同じ契約で検証されます。

### クラス別にキーフレーム間隔を設定する

tracking後の確定クラスごとに、努力目標の`keyframe_interval`を独立して設定
できます。形状はProduction polygon、内部gap補完上限は15に固定されています。

```bash
python run_pipeline.py \
  --input-sqlite input/inference.sqlite \
  --input-video input/video.mp4 \
  --output-dir output/classwise \
  --keyframe-interval 3 \
  --class-postprocess-policy-json \
    configs/class_postprocess_policy.example.json
```

ポリシーは次の形式です。

```json
{
  "schema_version": 1,
  "default": {
    "keyframe_interval": 3
  },
  "classes": {
    "男性器": {
      "keyframe_interval": 2
    }
  }
}
```

優先順位はクラス設定、ポリシーの`default`、CLIの共通値です。同じ間隔を持つ
クラスは1グループへまとめ、SQLite走査を共有します。`keyframe_interval`は1以上で、
値が小さいほどキーフレームが密になります。

クラス判定にはtracking後のtrack確定ラベル（多数決）を使います。設定した
クラスが入力に存在しなくてもエラーにはならず、未指定クラスは`default`へ
進みます。旧policyの`shape_mode: polygon`と`max_gap: 15`は移行入力として受理
しますが、異なる値は明示的に拒否し、旧アルゴリズムへfallbackしません。

最終SQLiteには`class_postprocess_policies`と
`mask_postprocess_provenance`を追加します。後者は各性器マスクについて、
適用した固定形状契約、キーフレーム間隔、補完上限、クラス指定かdefaultか、
新規補完フレームかを記録します。`cuts`、`cut_detection_metadata`、
`raw_tracks`も最終統合後に保持されます。検出単位の追跡結果は
`raw_tracked_masks`を複製せず、`tracking_assignments`から生出力
`segmentations`を参照します。

任意のステージグラフを指定する`--pipeline-config`とは同時に使用できません。

### 顔・目マスクを性器マスクと統合する

Face DINO v2を含むunified inference schema-v3では、顔検出の後処理を追加
できます。目マスクは採用済みの余白設定で、回転楕円または回転長方形を
選択します。

```bash
python run_pipeline.py \
  --input-sqlite input/inference.sqlite \
  --input-video input/video.mp4 \
  --output-dir output/postprocess \
  --face-mask-target eyes \
  --eye-mask-shape ellipse \
  --minimum-eye-confidence 0.35
```

`--face-mask-target face`では検出器の正確な顔楕円を使用します。
`--face-mask-target eyes`では2つのvalidなEye点からマスクを導出し、不足・
低信頼・幾何的に不自然な場合は顔楕円ベースへフォールバックします。

生成される成果物は次の3つです。

- `face_masks_sqlite`: 顔または目だけのマスクと導出監査情報
- `combined_predictions_sqlite`: 性器と顔/目を同じ`masks`へ統合
- `combined_validation_report`: 統合SQLiteの検証結果

性器の`predictions_sqlite`は上書きしません。統合SQLiteでは
`track_id=face:eyes:<scene_id>:<track_number>`または
`face:face:<scene_id>:<track_number>`、
`label=Eyes`または`Face`となるため、ソフトウェアは性器と同じ`masks`
readerで読み込めます。`mask_provenance`には元の顔観測ID、直接Eye点か
fallbackか、confidence、アルゴリズムversionを保存します。

unified inference SQLiteを入力した標準実行では、これらの内部成果物を使って
最後に`result_sqlite`を生成します。これは推論の全生出力を保持したまま、
`tracking_assignments`、最終編集キーフレーム、カット、監査・provenanceを
追加した唯一の公開SQLiteです。下流ソフトウェアは`predictions_sqlite`や
`combined_predictions_sqlite`ではなく`result_sqlite`を使用してください。
後処理を行わない場合も`package_result.py`で同じ
`keyframe-primary-v3`契約に包装できます。未実行のtracking、最終mask、顔詳細、
classwise処理は欠落テーブルではなく空テーブルとなり、利用可否は
`result_capabilities`に記録されます。

公開SQLiteは`schema_version=3`、`contract_revision=5`で、編集用の
`mask_track_segments`、`mask_keyframes`、`keyframe_components`とtyped形状表を
常設します。ポリゴンは頂点列、楕円は`cx/cy/radius_x/radius_y/theta_radians`、
目元長方形は中心・half extent・角度として読み込めます。データがないモードでも
表は消えず0行です。全フレーム`masks`と`tracked_masks`は公開SQLiteへ重複保存
せず、`mask_keyframes`を唯一の最終形状正本とします。trackingは
`tracking_assignments.source_detection_id`から生出力`segmentations`を参照
します。overlayは必要な範囲だけ一時的な毎フレームcacheへ復元します。
Face DINO v2だけの入力でも`package_result.py --face-mask-target face/eyes`を
指定すれば、派生マスクを通常の最終キーフレームとして格納できます。

インストール後は、同じCLIを`postprocess`コマンドでも実行できます。

出力ディレクトリには各ステージのサブディレクトリ、最終成果物、および
`pipeline_manifest.json`が生成されます。manifestには使用実装、要求・提供
成果物、実行時間、全成果物のパスが記録されます。

raw入力からカット検出を実行した場合、公開`result.sqlite`には
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
カット位置`cuts(frame)`は含まれますが、現行の監査用
`tracking_assignments`、`raw_tracks`、`cut_detection_metadata`は含めません。現行
`predictions_sqlite`が正本であり、互換版は追加成果物です。

変換だけを行う境界アダプターCLIもあります。

```bash
python -m artifacts.export_legacy_cli \
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
