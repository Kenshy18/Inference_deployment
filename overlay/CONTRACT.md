# Overlay input contract

`overlay`は入力SQLiteを読み取り専用で開き、`inference`や`postprocess`の
Pythonモジュールをimportしません。接続点は以下のSQLite列だけです。

## Unified inference SQLite

対応schema:

```text
schema_name = instance-segmentation-unified-inference
schema_version = 2 or 3
```

生マスクでは`model_executions.role = instance_segmentation`に属する
`detections`、`segmentations`、`segmentation_polygons`、
`segmentation_points`を読みます。分類結果がある場合は表示ラベルとスコアに
`classifications`を優先します。

顔では`model_executions.role = face_detection`に属する`detections`を読みます。
schema v2は従来どおり`[x1, y1, x2, y2)`を描画します。schema v3の
`face_observations`がある場合、通常CPU/NVENC rendererはHead box、正確なFace楕円、
顔確率マスク、validなキーポイントを描画します。高速native rendererはv3を受理し、
互換用のHead/Face boxを描画します。

フレーム番号と座標は0始まり、元動画画素座標です。

## Postprocess mask SQLite

tracked/finalモードは次の最小契約だけを要求します。

```text
masks(
    frame INTEGER,
    track_id TEXT,
    polygons TEXT,
    label TEXT optional
)
```

`polygons`は次のJSONです。

```json
[
  [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]]
]
```

- `tracked`: NMS、カット検出、tracking、短命track削除の直後に生成されたSQLite
- `final`: polygon/ellipse近似、keyframe選択、gap fillまで完了したSQLite

両者の物理schemaは同じです。処理段階を推測せず、利用者がモードと成果物を
明示します。

## Face privacy mask sidecar SQLite

`overlay-export-face-masks`はschema-v3の顔楕円・keypointから顔全体または目の
privacy polygonを派生し、入力SQLiteとは別のsidecarへ書きます。既存の最小
mask契約に加えて、由来を追跡する列を持ちます。

```text
schema_info(
    key TEXT PRIMARY KEY,
    value TEXT
)
masks(
    frame INTEGER,
    track_id TEXT,
    polygons TEXT,
    shape_type TEXT,             -- ellipse / rectangle
    label TEXT,                  -- Face / Eyes
    source_observation_id INTEGER,
    derivation TEXT,             -- face-ellipse / eye-keypoints / ellipse-fallback
    confidence REAL,
    PRIMARY KEY(frame, track_id)
)
```

`schema_name=face-privacy-mask-sqlite`、`schema_version=1`です。座標は元動画の
画素座標、フレームは0-basedです。`face_present=0`や有効な顔楕円がない観測は
マスクを出しません。`confidence`は顔楕円ではface score、Eye点採用時は2点の
低い方のconfidence、幾何fallbackでは`0.0`です。入力推論SQLiteはread-onlyで
開き、変更しません。

## Validation

- SQLiteの必須table/columnと`PRAGMA integrity_check`を確認します。
- inference SQLiteでは動画幅、高さ、FPS、最大フレーム番号を照合します。
- postprocess SQLiteでは最大フレーム番号を照合します。
- 入力SQLiteは変更しません。
- 出力動画は一時ファイルからatomicに置き換えます。
