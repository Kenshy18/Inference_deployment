# クラス別後処理ベンチマーク（2026-07-28）

## 結論

クラス別の`shape_mode`、`keyframe_interval`、`max_gap`は、tracking後の
確定label単位で独立設定できる。最終SQLiteには適用policyと補完由来を残し、
既存のカット・tracking監査テーブルも保持する。

10分入力では、クラス別ルーティングを使って全クラスを従来と同じpolygon
K3/G0へ流した場合、従来出力とマスク行が完全一致した。固定オーバーヘッドは
stage合計中央値で0.242秒だった。

## 条件

- GPU: NVIDIA GeForce RTX 5090
- Python: production runtime Python 3.10
- 入力:
  `benchmark_10min_20260728`のgiant Co-DINO追跡済みSQLite
- 動画範囲: 14,400 frames（約10分）
- 入力マスク: 7,556
- track: 9
- クラス:
  - 男性器: 8 tracks / 7,542 masks
  - 結合部分: 1 track / 14 masks
- 楕円: CUDA、batch 128、prep workers 4、FP32、TF32 off
- 各構成3回。アルゴリズム差にはmanifestのstage合計中央値を使用

プロセスwall timeにはstage外で約3.2秒の断続的な待ちが混入したが、各stageの
時間は安定していた。そのため速度差の一次指標はstage合計とし、wall timeは
範囲を併記する。

## クラス別ルーティングの差

| 構成 | stage合計中央値 | process wall範囲 | 最終mask | 基準との差 |
| --- | ---: | ---: | ---: | ---: |
| 全polygon K3/G0（従来） | 2.344 s | 2.55–5.91 s | 7,556 | 基準 |
| 全polygon K3/G0（classwise） | 2.586 s | 2.75–2.79 s | 7,556 | +0.242 s / +10.3% |
| 全ellipse K2/G30（従来） | 18.632 s | 19.09–22.45 s | 7,595 | 基準 |
| 男性器ellipse K2/G30、結合部分polygon K3/G0 | 19.602 s | 20.21–23.59 s | 7,595 | +0.970 s / +5.2% |
| 男性器polygon K3/G0、結合部分ellipse K2/G30 | 2.844 s | 2.96–3.06 s | 7,556 | 対全polygon +0.500 s / +21.3% |

主要クラス楕円の最終内訳は、男性器7,581行（補完39）と結合部分polygon
14行だった。希少クラス楕円は結合部分14行だけを楕円処理し、男性器7,542行を
polygon処理した。楕円K2は観測数が少ない場合には重いforwardを実行しないため、
希少クラスだけの楕円化は小さい追加コストで済んだ。

## 各設定値の差

### Polygon

| 設定 | stage合計中央値 | process wall中央値 | 最終mask | IoU | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1/G0 | 0.938 s | 1.14 s | 7,556 | 0.9461 | 0.9481 | 0.9978 |
| K2/G0 | 2.010 s | 2.19 s | 7,556 | 0.9109 | 0.9178 | 0.9919 |
| K3/G0 | 2.344 s | 5.70 s（外れ待ち含む） | 7,556 | 0.8915 | 0.9030 | 0.9859 |
| K3/G30 | 2.373 s | 2.58 s | 7,597 | 0.8909 | 0.9030 | 0.9852 |

polygonではK1が最速かつ元maskとの画素一致も最高だった。K2/K3は中間
フレームを補間する計算が増える一方、最終SQLiteは全観測フレームを保持する。
したがって速度と元mask再現を優先する場合はK1が有利で、時間方向の平滑化を
意図する場合だけK2/K3を選ぶ。

K3でG0からG30へ変更した追加コストは0.029秒（約1.2%）で、41マスクを追加
した。補完対象が少ない今回のデータでは、補完上限の速度影響はほぼない。

### Ellipse

| 設定 | stage合計中央値 | process wall中央値 | 最終mask | Global IoU | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K2/G0 | 18.789 s | 22.30 s（外れ待ち含む） | 7,556 | 0.9377 | 0.9857 | 0.9506 |
| K2/G30 | 18.632 s | 19.17 s | 7,595 | 0.9377 | 0.9857 | 0.9506 |
| K3/G30 | 20.142 s | 20.61 s | 7,595 | 0.9332 | 0.9834 | 0.9482 |

K2/G0とK2/G30の品質評価は元の7,556観測に対して完全に同じで、39補完maskの
生成コストも測定揺らぎ以下だった。K3はkeyframe選択が約1.4秒増え、IoUも
0.0045低下したため、この入力ではK2を維持するのが妥当である。

## SQLite検証

- `PRAGMA integrity_check`: `ok`
- journal mode: `delete`としてatomic公開
- `tracks`: 9
- `cuts`: 2
- `cut_detection_metadata`: 1
- `raw_tracked_masks`: 7,584
- `raw_tracks`: 17
- 全polygon classwiseと従来polygonの
  `(frame, track_id, polygons, shape_type, label)`: 完全一致
- 各最終maskに`mask_postprocess_provenance`が対応
- 異なるグループから同じ`(frame, track_id)`が出た場合は統合を拒否

実測中、WAL modeの入力をグループ分割した際に未checkpointの変更が一時DB名の
WALへ残る問題を検出した。作業SQLiteを`DELETE` journalへ確定してからatomic
renameするよう修正し、WAL入力fixtureを使う回帰テストを追加した。

最終統合はPythonへ全maskを展開せず、SQLite間の`INSERT ... SELECT`とSQL上の
provenance判定で行う。したがって統合時メモリは動画長ではなく主にtrack数に
依存する。実測RSS中央値は全polygonで99,448KB、希少クラス楕円で120,844KB。
主要クラス楕円の2,176,952KBはK2モデル本体が支配している。

## 推奨

- polygonで元mask再現と速度を優先: K1/G0
- polygonで時間方向の平滑化を優先: クラスごとにK2またはK3
- ellipse: K2
- `max_gap`: 見逃しを埋めたいクラスだけ必要な上限を設定。今回規模では
  0と30の速度差は無視できる
- クラス別機構の固定コストは絶対値で小さいため、品質要件に合わせた混在設定を
  優先してよい
