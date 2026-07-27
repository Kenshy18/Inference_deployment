# Overlay input contract

`overlay`は入力SQLiteを読み取り専用で開き、`inference`や`postprocess`の
Pythonモジュールをimportしません。接続点は以下のSQLite列だけです。

## Unified inference SQLite

対応schema:

```text
schema_name = instance-segmentation-unified-inference
schema_version = 2
```

生マスクでは`model_executions.role = instance_segmentation`に属する
`detections`、`segmentations`、`segmentation_polygons`、
`segmentation_points`を読みます。分類結果がある場合は表示ラベルとスコアに
`classifications`を優先します。

顔では`model_executions.role = face_detection`に属する`detections`の
`[x1, y1, x2, y2)`を読みます。顔検出にはpolygonを要求しません。

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

## Validation

- SQLiteの必須table/columnと`PRAGMA integrity_check`を確認します。
- inference SQLiteでは動画幅、高さ、FPS、最大フレーム番号を照合します。
- postprocess SQLiteでは最大フレーム番号を照合します。
- 入力SQLiteは変更しません。
- 出力動画は一時ファイルからatomicに置き換えます。

