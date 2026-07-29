# Keyframe-primary V3 30分ベンチマーク

実施日: 2026-07-29

## 結論

公開SQLiteを、生出力・追跡参照・編集可能キーフレームを正本とする
`keyframe-primary-v3`へ変更した。同じ生出力と最終表示内容を持つV2相当構成との
比較で、SQLiteは685.50 MiBから335.69 MiBへ51.03%縮小した。

高速overlayはV3のキーフレームを6つのworker範囲へ並列復元し、各worker専用の
一時shardをC++/libav rendererへ渡す。楕円・長方形は点列JSONへ展開せずtyped
parameterのまま渡す。30分・43,202フレームの最終overlayは26.46秒、
1,632.89 FPS（動画実時間の68.0倍）だった。同時期に再測定した展開済みmaskの
24.68秒、1,750.41 FPSとの差は1.78秒、7.20%に抑えた。

## 条件

- 入力: `HEYZO-3545 ... .mp4`の先頭約30分
- 動画: 1280x720、24 FPS、43,202フレーム、1,800.104秒
- 性器検出: `dinov3_codino_mh0`、TensorRT fast
- 顔検出: Face DINO v2
- 後処理: polygon、keyframe interval 3、gap 30、短命track 10以下を削除
- 顔privacy mask: eyes、ellipse
- cut検出: high precision
- overlay: fast、CPU 3 + NVENC 3、H.264 8 Mbps、音声なし

設定は
`orchestration/configs/benchmarks/v3_mh0_new_face_polygon_k3_30min_20260729.json`
に保存した。

## V3の公開契約

V3では次の3層だけを正本として公開する。

1. `detections`、`segmentations`、顔詳細などの変更しないAI生出力
2. `tracking_assignments`による生検出IDとraw/final trackの関連
3. `mask_keyframes`以下のtyped polygon/ellipse/rectangleによる編集可能最終形状

`masks`、`tracked_masks`、`raw_tracked_masks`は公開SQLiteへ保存しない。
最終形状はキーフレームから復元し、追跡表示は
`tracking_assignments.source_detection_id`から生polygonを参照する。
データがないモードでも固定テーブルは0行で存在し、機能の有無は
`result_capabilities`で表す。

30分結果の主要行数:

| 項目 | 行数 |
|---|---:|
| frames | 43,200 |
| detections | 129,307 |
| raw segmentations | 30,506 |
| face observations | 50,711 |
| tracking assignments | 26,873 |
| mask track segments | 48,230 |
| mask keyframes | 57,081 |
| polygon keyframe points | 102,011 |
| cuts | 10 |

## SQLiteサイズ

| 構成 | bytes | MiB |
|---|---:|---:|
| V3 keyframe-primary | 351,997,952 | 335.69 |
| V2相当・毎フレームmask併存 | 718,794,752 | 685.50 |
| 削減量 | 366,796,800 | 349.80 |

削減率は51.03%。V3はV2相当構成の48.97%である。

V2値は、同じV3結果へ旧構成の`masks` 76,302行、
`tracked_masks` 25,846行、`raw_tracked_masks` 26,873行を追加した
`v2_materialized_projection.sqlite`の実測値である。異なる推論結果同士を
比較した推定値ではない。

V3の公開ファイルには一時cacheを含めない。最終overlay用の6 shardは合計
16,203,776 bytes（15.45 MiB）で、処理後に自動削除される。最適化前の単一cache
226.06 MiBに対して93.16%小さい。

## Overlay速度

| 入力方式 | cache生成 | renderer | 合計 | FPS |
|---|---:|---:|---:|---:|
| V2相当dense・同時期再測定 | 0秒 | 24.51秒 | 24.68秒 | 1,750.41 |
| V3・旧直列cache | 8.21秒 | 24.58秒 | 32.96秒 | 1,310.85 |
| V3・並列shard＋typed形状 | 1.82秒 | 24.46秒 | 26.46秒 | 1,632.89 |

旧V3から6.50秒、19.72%短縮し、FPSは24.57%向上した。一時復元時間は
8.21秒から1.82秒へ77.87%短縮した。denseとのrenderer時間差は測定揺らぎの範囲で、
残る1.78秒はV3キーフレームの復元コストである。これを完全に0へするには、
C++ rendererがV3のpolygon補間まで直接実行する必要がある。

両出力とも43,202フレーム、1,800.083秒、24 FPSで、欠損フレームはなかった。
別々のH.264 encode結果のファイルサイズ差は約0.03%だった。

## 形状同等性

V2相当の最終`masks` 76,302件とV3から復元した76,302件を全件比較した。

- key欠落: 0
- label不一致: 0
- JSON座標列まで一致: 76,069
- polygon配列順のみ異なる: 233
- 1280x720 rasterで異なるmask: 0
- XOR差分pixel: 0
- 最小IoU: 1.0

233件は互いに離れた複数polygonの配列順だけが異なり、塗られる画素は同一だった。

追跡overlayについても、旧`tracked.sqlite` 25,846件と
`tracking_assignments`から復元した25,846件を全件比較し、key、label、
polygon JSONのすべてが完全一致した。

compact typed cacheについては、Pythonで展開した楕円polygonとC++でtyped
parameterから展開した結果をencode前後のraw YUV画素で比較し、完全一致を
回帰テストで確認した。30分実測でも全workerの描画mask件数はdense版と一致した。

## 処理速度の参考

一気通貫runのwall timeは9分48.87秒、最大RSSは約3.08 GiBだった。
orchestrator計測では推論487.93秒、後処理43.19秒。後処理の主要内訳は次の通り。

| 後処理phase | 秒 |
|---|---:|
| normalization | 4.849 |
| score policy | 0.809 |
| NMS | 1.141 |
| cut detection | 8.447 |
| tracking | 2.546 |
| polygon approximation | 1.053 |
| keyframe selection | 0.227 |
| gap fill | 5.379 |
| exact evaluation | 1.424 |
| face privacy masks | 5.116 |
| face merge | 1.195 |
| integrated V3 package | 7.387 |

## 成果物

- V3 SQLite:
  `orchestration/output/benchmark_v3_30min_20260729/run/02_postprocess/13_integrated_result_sqlite/result.sqlite`
- V2相当サイズ比較:
  `orchestration/output/benchmark_v3_30min_20260729/v2_materialized_projection.sqlite`
- V3 overlay:
  `orchestration/output/benchmark_v3_30min_20260729/overlay/v3_keyframe_compact/final.mp4`
- dense比較overlay:
  `orchestration/output/benchmark_v3_30min_20260729/overlay/v2_dense/final.mp4`

いずれのSQLiteも`PRAGMA integrity_check=ok`を確認した。
