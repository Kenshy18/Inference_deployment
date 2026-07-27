# Postprocess contract

このファイルは、後処理モジュールを安全に変更・交換・追加するための公開規約
です。機能実装はこの規約だけに依存し、他機能の内部実装へ依存しません。

## 1. ステージ契約

すべての交換可能な処理は`contracts.PostprocessStage`を満たします。

```python
class PostprocessStage(Protocol):
    name: str
    requires: frozenset[str]
    provides: frozenset[str]

    def run(self, context: StageContext) -> StageResult: ...
```

- `requires`: 実行前に存在しなければならない名前付き成果物
- `provides`: 実行後に新たに存在しなければならない名前付き成果物
- `context.artifacts`: 前段までに作成された読み取り専用の成果物パス
- `context.stage_dir`: 当該ステージ専用の出力先
- `StageResult.artifacts`: `provides`を含む生成ファイルのパス
- 入力ファイルを上書きしない
- 宣言した出力がない、未宣言出力がある、またはschema違反の場合は失敗させる
- 既存成果物を同名で上書きしない
- 生成物は`context.stage_dir`内へ出力する

`common.PipelineRunner`が実行前後にこれらを検査します。組み込み成果物の
validatorは`contracts.artifacts`へ集約されています。新しい成果物名では
`register_artifact_contract(name, validator)`によりschemaも登録できます。

## 2. 一気通貫の成果物接続

ポリゴン構成の標準接続は次のとおりです。

| 機能 | 組み込み実装 | requires | provides |
| --- | --- | --- | --- |
| 正規化 | `preprocessing.normalize` | `input_jsonl` | `normalized_jsonl` |
| スコア方針 | `preprocessing.score_policy` | `normalized_jsonl` | `scored_jsonl` |
| NMS | `nms.adaptive` | `scored_jsonl` | `nms_jsonl` |
| カット検出 | `cut_detection.video` | `nms_jsonl` | `cuts_json` |
| tracking | `tracking.greedy` | `nms_jsonl`, `cuts_json` | `tracked_sqlite` |
| polygon近似 | `approximation.polygon.rdp` | `tracked_sqlite` | `approximated_sqlite` |
| keyframe | `keyframes.polygon.interval` | `approximated_sqlite` | `keyframes_sqlite` |
| gap fill | `gap_fill.polygon.linear` | `approximated_sqlite`, `keyframes_sqlite` | `predictions_sqlite` |
| 評価 | `evaluation.mask_iou` | `tracked_sqlite`, `predictions_sqlite` | `evaluation_summary` |
| 出力検証 | `artifacts.validate` | `predictions_sqlite` | `validation_report` |

楕円構成では、`tracked_sqlite`以降を次の接続へ交換します。

| 機能 | 組み込み実装 | requires | provides |
| --- | --- | --- | --- |
| ellipse近似 | `approximation.ellipse.production` | `tracked_sqlite` | `approximated_sqlite`, `approximation_metrics_csv` |
| keyframe | `keyframes.ellipse.dense` | `approximation_metrics_csv` | `keyframes_json`, `interpolated_union_json` |
| gap fill | `gap_fill.ellipse.linear` | `interpolated_union_json`, `approximation_metrics_csv` | `filled_union_json`, `filled_metrics_csv` |
| 評価 | `evaluation.ellipse.exact` | `filled_union_json`, `tracked_sqlite` | `evaluation_summary` |
| SQLite生成 | `artifacts.union_sqlite` | `filled_union_json`, `tracked_sqlite` | `predictions_sqlite` |
| 出力検証 | `artifacts.validate` | `predictions_sqlite` | `validation_report` |

`input_video`と`class_policy_json`は任意の補助成果物です。標準cut detectionを
有効にした場合のみ`input_video`が必要です。

Face DINO v2の顔後処理を有効にすると、通常の最終出力検証後に次を追加します。

| 機能 | 組み込み実装 | requires | provides |
| --- | --- | --- | --- |
| 顔/目mask生成 | `face_privacy.masks` | `input_raw_sqlite` | `face_masks_sqlite` |
| 性器maskとの統合 | `face_privacy.merge` | `predictions_sqlite`, `face_masks_sqlite` | `combined_predictions_sqlite` |
| 統合出力検証 | `artifacts.validate` | `combined_predictions_sqlite` | `combined_validation_report` |

入力推論SQLiteと性器のみの`predictions_sqlite`は読み取り専用であり、
上書きしません。

## 3. 入力JSONL

1行が1フレームのJSON objectです。入力時は次の別名を受理します。

- frame: `frame_index`または`frame_idx`
- detections: `detections`または`instances`
- label: `class_name`または`label`
- bbox: `bbox_xyxy=[x1,y1,x2,y2]`または`bbox=[x,y,width,height]`
- mask: 非空の`polygons`または`segmentation`

最小例:

```json
{"frame_idx":0,"instances":[{"label":"target","score":0.93,"bbox":[10,20,30,40],"segmentation":[[[10,20],[40,20],[40,60],[10,60]]]}]}
```

正規化後のcanonical JSONLは、`frame_index: int`と`detections: list`を必須と
します。各detectionは非空の`polygons`、`class_name`、`label`を持ちます。
score policyとNMSはこの形式を保ったまま検出だけを除外します。

### 未追跡の検出SQLite

`--input-sqlite`にはDINOv3の`raw_mask_sqlite_v1`を指定できます。次のテーブルを
要求します。

```text
metadata(key, value)
frames(frame, time_sec, width, height)
masks(frame, mask_id, detection_index, label, class_name, category_id,
      score, detector_score, class_score, bbox_xyxy, polygons, source_json)
```

InstanceSegmentationのunified inference schema v2/v3も指定できます。schema名は
`instance-segmentation-unified-inference`とし、少なくとも次のテーブルを
要求します。

```text
frames(id, run_id, frame_index, timestamp_sec, width, height)
detections(id, frame_id, model_execution_id, class_id, class_name, score,
           x1, y1, x2, y2)
classifications(detection_id, class_id, class_name, score)
segmentations(detection_id, encoding)
segmentation_polygons(id, detection_id, polygon_index)
segmentation_points(polygon_id, point_index, x, y)
```

どちらも`input_raw_sqlite`として識別され、`preprocessing.raw_sqlite`が
`normalized_jsonl`へ変換します。フラットなポリゴン座標もcanonicalな
`[[x, y], ...]`へ変換されます。標準構成では元動画をカット検出に使用し、
その後にtrackingを行います。`track_id`を持つSQLiteは従来どおり
`tracked_sqlite`として扱います。

## 4. cuts JSON

```json
{
  "schema_version": 1,
  "frames": [120, 450],
  "method": "high_precision",
  "elapsed_seconds": 0.42
}
```

`frames`はカット後のシーンが始まるフレーム番号です。cut detectionを交換する
場合は、`nms_jsonl`を読み、この形式の`cuts_json`を生成します。

## 5. マスクSQLite

`tracked_sqlite`、中間SQLite、`predictions_sqlite`は最低限`masks`テーブルと
次の列を持ちます。

| 列 | 型 | 意味 |
| --- | --- | --- |
| `frame` | INTEGER | 0以上のフレーム番号 |
| `track_id` | TEXT | 空でない追跡ID |
| `polygons` | TEXT | JSON list |
| `label` | TEXT | 任意のクラス名 |
| `shape_type` | TEXT | `polygon`または`ellipse`等 |

`tracks`テーブルは保持可能ですが、最終検証の必須条件ではありません。
共通の読み書きには`contracts.read_mask_rows`と
`contracts.write_mask_sqlite`を使用します。

`face_masks_sqlite`は同じ`masks`契約に加えて次を持ちます。

```text
mask_provenance(
  frame INTEGER,
  track_id TEXT,
  mask_kind TEXT,             -- face / eyes
  source_observation_id INTEGER,
  derivation TEXT,            -- face-ellipse / eye-keypoints / ellipse-fallback
  confidence REAL,
  algorithm_version TEXT,
  PRIMARY KEY(frame, track_id)
)
```

`combined_predictions_sqlite`は性器側SQLiteをbackupしてから、名前空間付き
`track_id`の顔マスクと`mask_provenance`を追加した成果物です。

標準のraw入力パイプラインが生成する`tracked_sqlite`と
`predictions_sqlite`には、カット検出結果も次の監査テーブルとして保持します。

```text
cuts(frame INTEGER PRIMARY KEY)
cut_detection_metadata(
  id INTEGER PRIMARY KEY,       -- 常に1
  schema_version INTEGER,       -- 現在1
  method TEXT,
  elapsed_seconds REAL,
  cut_count INTEGER,
  frame_semantics TEXT          -- first_frame_of_new_scene
)
```

`cuts.frame`は新しいシーンの先頭となる0-basedフレーム番号です。
`cut_detection_metadata`がある場合、最終validatorは1行だけであること、
`cut_count`と`cuts`の件数が一致することを検証します。旧来の追跡済みSQLiteを
入力する場合を考慮し、これらの監査テーブルがないSQLiteも読み込み可能です。
後処理の各マスク変換は参照SQLiteを安全にbackupしてから`masks`だけを書き換える
ため、カット情報はポリゴン・楕円のどちらでも最終SQLiteまで伝搬します。

### 旧Dinov3_postprocess互換SQLite

`--export-legacy-sqlite`を指定した場合、通常の`predictions_sqlite`に加えて
`legacy_predictions_sqlite`を生成します。この互換成果物のテーブルは
`masks`、`tracks`、`cuts`の3つだけで、旧`AI後処理最終.sqlite`と同じ列順・
主キーを持ちます。

```text
masks(
  frame INTEGER NOT NULL,
  track_id TEXT NOT NULL,
  polygons TEXT,
  shape_type TEXT,
  dilate_px INTEGER NOT NULL DEFAULT 0,
  feather_px INTEGER NOT NULL DEFAULT 0,
  mosaic_block INTEGER NOT NULL DEFAULT 0,
  mosaic_alias REAL NOT NULL DEFAULT 0,
  label TEXT,
  PRIMARY KEY(frame, track_id)
)
tracks(track_id TEXT PRIMARY KEY, label TEXT)
cuts(frame INTEGER PRIMARY KEY)
```

現行固有の`raw_tracked_masks`、`raw_tracks`、`cut_detection_metadata`は意図的に
除外します。互換出力はtentativeな境界アダプターであり、内部処理および新規
連携では監査情報を保持する`predictions_sqlite`を使用します。

## 6. 新実装の条件

1. 所有する機能ディレクトリ内に実装する。
2. `PostprocessStage`を実装する。
3. 既存接続へ交換する場合、置換対象と同じ`requires`/`provides`を使う。
4. 新機能を差し込む場合、新しい成果物名を宣言し、後段がそれを要求する。
5. 他の機能パッケージをimportしない。共有が必要な型・I/Oだけを`contracts`へ
   追加し、実行補助だけを`common`へ追加する。
6. 新成果物を追加する場合は、成果物validatorも`contracts`へ登録する。
7. pipeline JSONでは組み込み登録名、`python.module:attribute`、または
   `postprocess.stages` entry point名を指定する。
8. `tests/test_architecture.py`と新機能の単体テストを通す。

契約を壊す変更は、同じ成果物名を使って黙って導入せず、新しい成果物名または
schema versionとして明示します。
