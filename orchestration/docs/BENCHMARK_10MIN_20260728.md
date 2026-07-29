# 10分実動画パイプライン検証 — 2026-07-28

## 結論

10分・14,400フレームの実動画について、Cascade Mask CNNを除外し、以下を
実測した。

- 巨大Co-DINO / 高速Co-DINO MH0
- 新Face DINO v2あり / なし
- ポリゴン後処理 / GPU K2を使う楕円後処理
- raw / tracked / final / faces
- 性器・顔・combinedの詳細 / 簡易表示

速度と精度のバランスが最も良い標準候補は
`高速Co-DINO MH0 + Face DINO v2 + 楕円 + fast final`である。10分動画を
181.39秒、動画実時間の3.31倍速で処理できた。

速度優先なら同じモデルのポリゴン版が163.60秒、3.67倍速である。ただし
追跡後マスクに対する最終形状IoUは楕円0.9299に対しポリゴン0.8893だった。

## 測定条件

- 入力:
  `orchestration/output/benchmark_10min_20260728/input/heyzo_first_10min.mp4`
- 元動画: H.264、1280×720、24 fps
- 測定範囲: frame 0–14,399、正確に600.0秒
- GPU: NVIDIA GeForce RTX 5090
- 推論backend: TensorRT fast
- スコア閾値: 0.35
- 短命track削除: 10フレーム以下
- カット検出: `high_precision`
- ポリゴン: keyframe interval 3
- 楕円: keyframe interval 2、K2 FP32/CUDA、TF32 off
- Overlay: H.264、8 Mbit/s、NVENC p1、音声なし

ストリームコピーした入力ファイル自体にはGOP終端の都合で14,402フレームが
含まれる。すべての測定と出力は先頭14,400フレームに固定している。

WSLの壁時計は長いGPU測定中にホスト時刻補正を受けたため、正式な比較値には
各実装が`time.perf_counter()`で記録した単調時計を使用した。

## 推論

| Co-DINO | 新顔 | 推論stage | stage FPS | 動画比 | Co-DINO compute | Face compute | segmentation | faces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 高速MH0 | なし | 89.339 s | 161.18 | 6.72× | 176.962 FPS | — | 9,119 | — |
| 高速MH0 | あり | 147.892 s | 97.37 | 4.06× | 115.691 FPS | 102.824 FPS | 9,119 | 16,260 |
| 巨大 | なし | 585.117 s | 24.61 | 1.03× | 24.866 FPS | — | 7,707 | — |
| 巨大 | あり | 654.244 s | 22.01 | 0.92× | 22.303 FPS | 61.564 FPS | 7,706 | 16,260 |

顔ありでは81,300 keypointと15,717 face probability-maskを保存した。

### GPUテレメトリ

| 構成 | GPU平均 | GPU最大 | 平均電力 | 最大電力 | 最大VRAM |
|---|---:|---:|---:|---:|---:|
| MH0・顔なし | 72.7% | 95% | 479.5 W | 577.6 W | 22,174 MiB |
| MH0・顔あり | 84.7% | 100% | 520.3 W | 583.5 W | 29,199 MiB |
| 巨大・顔なし | 98.4% | 100% | 567.2 W | 584.9 W | 12,706 MiB |
| 巨大・顔あり | 98.3% | 100% | 568.0 W | 578.6 W | 20,143 MiB |

巨大Co-DINOは単体ですでにGPUをほぼ使い切る。Face DINOとの同時区間では
巨大側が約18 FPS、Face側が61.56 FPSまで低下した。Face完了後は巨大側が
約25 FPSへ戻るため、顔追加のstage増加は69.13秒に留まる。直列実行よりは
速いが、MH0との並行実行ほど効率は良くない。

この巨大＋顔の並列結果は採用判断用の履歴である。その後の同条件A/Bで短縮が
3.0%に留まり、わずかな数値揺らぎも確認されたため、現在のproduction validator
は並列オプションを`dinov3_codino_mh0 + face_dino_v2`だけに制限している。
`giant_with_face.json`は測定provenanceとして残しているが、現行設定としては
意図的に検証エラーになる。

10分連続負荷でも巨大Co-DINOは約80–81℃、約2.59–2.60 GHzを維持し、明確な
サーマルスロットリングや線形メモリ増加はなかった。

## 顔あり・なしの推論再現性

MH0は顔あり / なしで次が完全一致した。

- segmentation 9,119行
- polygon 9,237個
- segmentation point 1,183,638点
- box、score、全polygon座標

巨大Co-DINOはTensorRTの並行kernel schedulingによる数値揺らぎがあった。
意味的に対応付けた7,705マスクの比較は以下。

- raster mask IoU平均: 0.99883
- raster mask IoU 1%点: 0.98897
- raster mask IoU最小: 0.92174
- IoU 0.9未満: 0
- bbox IoU平均: 0.99941
- bbox座標最大差: 11 px
- 対応不能: 顔なし側2件、顔あり側1件

3フレームでNMS候補数が変わり、総数が7,707から7,706になった。精度の
大幅な変化ではないが、巨大モデルを顔モデルと並行実行した場合はbitwise
deterministicではない。

## カット検出

14,400フレームを3.096秒で処理し、次の2カットを検出した。

```text
[5133, 5482]
```

CPU使用量は短時間に約11コア相当である。Orchestrationでは推論と完全に重ねられ、
下記E2E合計のクリティカルパスには加算されない。

## 後処理

表の時間は`pipeline_manifest.json`に記録された全stageの単調時計合計。
顔ありでは目楕円プライバシーマスク生成、merge、combined validation、
legacy SQLite出力を含む。

| Co-DINO | 新顔 | 形状 | 後処理 | instance final | combined final | tracks |
|---|---:|---|---:|---:|---:|---:|
| 高速MH0 | なし | polygon | 5.520 s | 8,157 | — | 29 |
| 高速MH0 | なし | ellipse | 22.819 s | 8,554 | — | 29 |
| 高速MH0 | あり | polygon | 8.282 s | 8,157 | 23,874 | 15,746 |
| 高速MH0 | あり | ellipse | 25.949 s | 8,554 | 24,271 | 15,746 |
| 巨大 | なし | polygon | 5.218 s | 7,556 | — | 9 |
| 巨大 | なし | ellipse | 22.034 s | 7,595 | — | 9 |
| 巨大 | あり | polygon | 7.711 s | 7,556 | 23,273 | 15,726 |
| 巨大 | あり | ellipse | 24.886 s | 7,595 | 23,312 | 15,726 |

すべてのSQLiteで14,400フレーム契約、2カット、`high_precision` metadata、
現行schemaとlegacy schemaの出力を検証した。

### 形状精度

追跡後の生マスクをreferenceとした最終出力の全体IoU:

| Co-DINO | polygon | ellipse | ellipseの差 |
|---|---:|---:|---:|
| 高速MH0 | 0.88929 | 0.92985 | +0.04056 |
| 巨大 | 0.89149 | 0.93745 | +0.04597 |

今回の設定では楕円のほうが高精度だが、約16.8–17.3秒余分にかかる。
ポリゴンと楕円はkeyframe intervalもそれぞれ3 / 2なので、純粋な
representationだけでなく採用予定の運用設定全体の比較である。

MH0楕円の代表的な内訳:

- total tracked rows: 8,157
- K1 rows: 2,061
- K2 CUDA rows: 6,096
- 楕円approximation: 15.318秒
- K2 solve: 12.582秒
- K2 forward: 9.995秒
- approximation global IoU: 0.94418

## 高速Overlayの種類別速度

基準入力はMH0＋Face DINOのSQLite。C++/libav NVDEC + CUDA描画 +
NVENC 6 workerのレンダラー時間。

| Overlay | 描画行 | 時間 | FPS |
|---|---:|---:|---:|
| raw | 9,119 masks | 7.403 s | 1,945.13 |
| tracked | 8,157 masks | 7.232 s | 1,991.25 |
| final polygon | 8,157 masks | 7.584 s | 1,898.73 |
| final ellipse | 8,554 masks | 7.096 s | 2,029.31 |
| faces | 31,977 face rows | 7.857 s | 1,832.86 |

形状・段階による差は小さく、10分動画を約7–8秒で出力できる。

顔プライバシーマスクをmergeしたfinalも実測した。

| 構成 | 形状 | masks | 時間 | FPS |
|---|---|---:|---:|---:|
| MH0＋顔 | polygon | 23,874 | 7.430 s | 1,938.11 |
| MH0＋顔 | ellipse | 24,271 | 7.548 s | 1,907.74 |
| 巨大＋顔 | polygon | 23,273 | 7.733 s | 1,862.07 |
| 巨大＋顔 | ellipse | 23,312 | 7.943 s | 1,812.91 |

## 通常NVENC表示プリセット

確定済みのOpenCV描画を維持する通常NVENC経路。レンダラー内部の単調時計値。

| 表示 | source / shape | 時間 | FPS |
|---|---|---:|---:|
| 性器詳細生出力 | raw | 36.800 s | 391.31 |
| 性器簡易生出力 | raw | 35.547 s | 405.10 |
| 性器詳細 | final ellipse | 35.916 s | 400.93 |
| 性器簡易 | final ellipse | 33.974 s | 423.85 |
| 性器簡易 | final polygon | 33.574 s | 428.91 |
| 顔詳細 | Face DINO | 48.676 s | 295.83 |
| 顔簡易 | Face DINO | 32.465 s | 443.55 |
| combined詳細 | final ellipse + Face DINO | 54.027 s | 266.53 |
| combined簡易 | final ellipse + Face DINO | 36.374 s | 395.88 |

顔詳細はprobability mask展開・合成、head box、ラベル、全keypoint表示が最も
重い。顔簡易は楕円とkeypointを維持しながら顔詳細より33.3%短い。
combined簡易はcombined詳細より32.7%短い。

ポリゴン / 楕円の性器簡易表示差は約0.40秒、1.2%であり、Overlay速度には
ほとんど影響しない。形状選択の主な速度差は後処理側で発生する。

## E2E比較

推論stage + 後処理stage合計 + 対応するfast final rendererの実測合計。
カット検出は推論と重なるため加算していない。

| Co-DINO | 新顔 | 形状 | 合計 | 10分実時間比 |
|---|---:|---|---:|---:|
| 高速MH0 | なし | polygon | **102.44 s** | **5.86×** |
| 高速MH0 | なし | ellipse | **119.25 s** | **5.03×** |
| 高速MH0 | あり | polygon | **163.60 s** | **3.67×** |
| 高速MH0 | あり | ellipse | **181.39 s** | **3.31×** |
| 巨大 | なし | polygon | **597.49 s** | **1.00×** |
| 巨大 | なし | ellipse | **614.40 s** | **0.98×** |
| 巨大 | あり | polygon | **669.69 s** | **0.90×** |
| 巨大 | あり | ellipse | **687.07 s** | **0.87×** |

Python起動、config読込、artifact検査などの短いlauncher overheadはこの合計に
含まれない。モデル・後処理・rendererそのものを同じ単調時計基準で比較する
ための値である。

## 動画と時刻の検証

20本の10分Overlayに対して全ストリーム検証を実行し、すべてPASSした。

- 14,400 encoded packets
- 14,400 decoded frames
- 24.0 fps
- duration 600.0秒
- PTS / DTS strictly monotonic
- PTS / DTS uniform
- 最大絶対timestamp誤差: 0.000000333秒
- full-stream decode errorなし
- 高速版の5分割境界はすべてkeyframe
- worker frame rangeは欠損・重複なし

通常NVENC manifestを検査した際、validatorが分割高速版専用の
`start_frame`を要求する問題を発見した。通常版では境界検査を空集合として扱い、
frame / packet / PTS / DTS / full decodeをそのまま検査するよう修正した。

同一frame 9,600を抽出して目視確認した。

- rawとtrackedでscore表示からtrack ID表示へ正しく変化
- polygonとellipseがそれぞれ期待する形状
- 簡易性器maskはpink、alpha 0.45
- 顔詳細はhead box、face ellipse、probability-mask境界、keypoint、confidence
- 顔簡易はellipseとkeypointのみ
- combinedで性器・顔の座標ずれなし

## 推奨

1. 標準品質:
   `MH0 + Face DINO v2 + ellipse(KF=2) + fast final`
2. 速度優先:
   `MH0 + Face DINO v2 + polygon(KF=3) + fast final`
3. 人間の目視確認で通常フォントが必要:
   simple presetを通常NVENCで使用
4. 顔probability-maskの診断が必要なときだけface/combined detailedを使用
5. 巨大Co-DINOは10分でほぼ実時間であり、MH0に対する精度優位が対象データで
   明確に確認できるケースに限定する
6. 巨大Co-DINOでbitwise再現性を優先する場合はFace DINOと直列実行する

## 成果物

- 推論:
  `orchestration/output/benchmark_10min_20260728/inference`
- 後処理:
  `orchestration/output/benchmark_10min_20260728/postprocess`
- 高速Overlay:
  `orchestration/output/benchmark_10min_20260728/overlay/fast`
- 通常NVENC preset:
  `orchestration/output/benchmark_10min_20260728/overlay/presets`
- 全検証report:
  `orchestration/output/benchmark_10min_20260728/overlay/*/*.validation.json`
- 同一frame目視画像:
  `orchestration/output/benchmark_10min_20260728/qa_frames`
- GPU / process measurements:
  `orchestration/output/benchmark_10min_20260728/measurements`

## 回帰テスト

最終ソースで以下を再実行した。

- inference: 61 pass、2 skip
- postprocess: 51 pass
- overlay: 31 pass
- overlay native: 1 pass
- orchestration: 20 pass

合計164 pass、2 skip。加えて20本すべての10分動画を全フレームdecode検証した。
