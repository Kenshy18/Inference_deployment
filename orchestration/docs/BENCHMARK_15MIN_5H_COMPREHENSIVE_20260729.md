# 15分・5時間 総合動作／性能／SQLite監査（2026-07-29）

## 結論

推論・後処理・通常オーバーレイ・固定SQLite schema rev3は、試した主要構成で
正常に動作した。巨大Co-DINO／高速DINO、新旧顔検出、polygon／GPU ellipse、
クラス別shape/keyframe/gap、顔privacy mask、raw/tracked/final/facesの各出力を
実動画で確認した。顔または性器を無効にした場合も表定義は変わらず、欠ける
componentは`not_requested`または`unsupported`として空表で表現される。

初回監査では次の3点を「問題なし」と判定しなかった。うち高速overlayの欠損は
同日追試で修正・5時間再検証まで完了した。

1. 高速オーバーレイは初回実装で15分全長1枚、反復5時間67枚を欠落させた。
   packet PTS keyframe index＋decode ordinal方式へ変更後、15分21,602/21,602、
   5時間431,935/431,935となり、独立validatorの全項目に合格した。
2. polygon gap fillとexact evaluationは全行をPythonの辞書／listへ保持する。
   15分polygon時の約122MiBに対し、5時間では最大約1.09GiBであり、推論・
   production overlayと違って入力データ量依存のメモリ増加が残る。
3. 高速DINO＋新顔のGPU並行実行は15分では直列推論時間より約9.7%短いが、
   5時間では直列より41.4%遅い。さらに反復した同一画素に対する顔出力が
   周回ごとに微変動する。長尺・再現性重視の既定値は直列が適切。

## 試験条件

- GPU: NVIDIA GeForce RTX 5090, VRAM 32,607MiB
- CPU: Intel Core Ultra 9 285K, 24 logical CPUs
- RAM: 約30GiB、swap 8GiB
- 入力: `HEYZO-3545 ... .mp4`から作成した1280x720、H.264、24fpsの先頭15分
- 基準範囲: 21,600 frames（`0..21599`）
- 長時間入力: 上記15分clipをストリームコピーで反復し、約5時間へ切ったもの
- 長時間フレーム: 431,935 frames
- 長時間overlay: C++/libav + NVDEC + CUDA draw + NVENC p1、6 workers、8Mbps
- 時間はmanifest内の`time.perf_counter()`計測を優先する。外側の
  `/usr/bin/time`は契約検査を含む工程全体の確認に用いる。

## 15分推論

`推論stage`はrole subprocess、merge、stage制御を含む。`工程全体`は結果包装と
契約検査も含む。

| 性器モデル | 顔モデル | 実行 | 推論stage | stage FPS | 工程全体 | peak RAM | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| 巨大Co-DINO | 新顔 | 直列 | 996.962s | 21.67 | 18:09.82 | 3,540MiB | 12,822MiB |
| 巨大Co-DINO | 旧顔 | 直列 | 1,024.721s | 21.08 | 18:34.31 | 5,203MiB | 12,816MiB |
| 高速DINO(MH0) | 新顔 | GPU並行 | 223.455s | 96.66 | 4:08.41 | 6,133MiB | 30,984MiB |
| 高速DINO(MH0) | 旧顔 | 直列 | 265.877s | 81.24 | 4:51.96 | 5,197MiB | 23,958MiB |
| 高速DINO(MH0)のみ | なし | 単独 | 133.412s | 161.90 | 2:30.52 | 3,168MiB | 23,801MiB |
| なし | 新顔のみ | 単独 | 114.082s | 189.34 | 2:08.06 | 2,987MiB | 8,395MiB |

role本体の実測は、巨大Co-DINOが24.47–24.58fps、高速DINO単独が173.62fps、
新顔単独が200.59fps、旧顔が165.77–166.29fpsだった。巨大Co-DINOは顔モデル
より大幅に遅いため並行化の対象にする価値が低い。

高速DINO＋新顔の15分比較では、単独stage合計247.494秒に対して並行stage
223.455秒で、並行が24.039秒（9.7%）短い。ただしpeak VRAMは約31.0GiBで
余裕が小さい。

## 15分後処理

全構成でhigh-precision cut detectionは6 cutsを返し、約4.56–4.62秒だった。
カットを跨ぐtrack segmentは0件。

| 構成 | pipeline | orchestration stage | peak RAM | 主な条件 |
|---|---:|---:|---:|---|
| MH0 new polygon K1 | 13.297s | 14.743s | 121MiB | gap 0、顔maskなし |
| MH0 new polygon K3 | 19.665s | 21.152s | 122MiB | gap 30、eyes ellipse |
| MH0 new ellipse K2 | 37.866s | 39.632s | 2,185MiB | GPU ellipse、gap 30、eyes rectangle |
| MH0 new ellipse K3 | 43.373s | 45.136s | 2,182MiB | GPU ellipse、gap 30、full-face ellipse |
| MH0 new classwise | 18.987s | 20.460s | 121MiB | 入力class不一致のためdefault polygon |
| MH0 old classwise | 14.546s | 15.530s | 122MiB | 入力class不一致のためdefault polygon |
| Giant old polygon K3 | 13.138s | 14.017s | 122MiB | gap 30 |
| Giant old classwise | 35.904s | 37.141s | 2,161MiB | 実クラス別polygon/ellipse混在 |
| Giant new ellipse K2 | 38.328s | 39.985s | 2,167MiB | GPU ellipse、eyes ellipse |

実際に適用されたクラス別policyは次のとおり。

| class | shape | keyframe interval | max gap |
|---|---|---:|---:|
| 女性器 | polygon | 1 | 12 |
| 男性器 | ellipse | 2 | 30 |
| 結合部分 | ellipse | 3 | 8 |
| default | polygon | 3 | 0 |

GPU ellipseの15分内訳はellipse approximation約18.3–18.7秒で、polygon RDPの
約0.46–0.51秒より重い。GPU化自体は動作し、GPU contextのためpeak RAMは
約2.2GiBになる。

## 15分オーバーレイ

### 高速表示

全ケースを21,600 framesに明示制限した試験では、frame/packet数、開始PTS、
PTS/DTS単調性、24fps均一性、worker境界、境界keyframe、全stream decodeの
11検査に合格した。

| overlay | render time | FPS | validator |
|---|---:|---:|---|
| raw | 10.959s | 1,970.91 | pass |
| tracked | 10.293s | 2,098.47 | pass |
| final | 11.350s | 1,903.09 | pass |
| faces | 10.813s | 1,997.65 | pass |
| final + faces | 12.204s | 1,769.87 | pass |

### reader-facing preset（通常NVENC）

| preset | render time | FPS | validator |
|---|---:|---:|---|
| genital detailed / final | 59.142s | 365.22 | pass |
| genital simple / final | 55.746s | 387.47 | pass |
| genital detailed / raw | 56.762s | 380.53 | pass |
| face detailed | 69.834s | 309.30 | pass |
| face simple | 48.919s | 441.55 | pass |
| combined detailed | 84.639s | 255.20 | pass |
| combined simple | 59.035s | 365.88 | pass |

旧顔SQLiteも通常`face-detailed`で21,602/21,602 frames、完全decodeを含む全検査に
合格した。旧モデルはrich face geometryが`unsupported`なのでbox表示へ
フォールバックする。renderは43.971秒、約491.27fps。

### 高速分割のフレーム欠落

- 15分全長: 入力実測21,602 frames、出力21,601 frames。最後のworkerが1枚不足。
- 5時間: 入力431,935 frames、出力431,868 frames。67枚不足。
- 5時間出力自体は完全decode可能で、出力PTS/DTSは厳密単調・24fps均一。
- したがって破損やconcat後の時刻不連続ではなく、入力frameから高速workerへ
  割り当てる段階の欠落。

初回native rendererは`best_effort_timestamp`をfps格子へ丸めて
`source_frame`へ戻していた。
ストリームコピー反復入力には格子外のgapがあるためframe indexが飛ぶ。また
full-range終端でも丸め境界により1枚落ちる。推論SQLiteの`frame_index`はdecode
順の連番なので、overlay側もPTS丸めではなくdecode frame ordinalと実際の境界
PTS indexを対応させる必要がある。

### 同日修正・再検証

runnerがvideo packetをdecodeなしでPTS順に索引化し、各workerへ直前keyframeの
frame ordinalとPTSを渡すよう変更した。rendererはseek後のdecode ordinalを
連番として使い、PTSからframe番号を再計算しない。また指定範囲の出力枚数が
不足した場合は成功扱いにせずfailする。

- 15分全長: 21,602/21,602、validator全項目pass
- 5時間全長: 431,935/431,935、6 workerすべて要求数と一致
- 5時間: packet index 2.383秒、描画＋concat 205.966秒、
  索引込み208.349秒／2,073.14fps
- 修正前: 222.955秒／1,937.32fps
- raw／tracked／final／faces／final＋faces:
  PTS gapを跨ぐ末尾4 framesを2 workerで処理し、全件4/4、validator pass
- 音声copy付きfinal＋faces: gapを跨ぐ12/12 frames、AAC開始時刻・preroll・
  packet／decoded frame連続性を含めvalidator pass
- 5時間検査: frame/packet数、PTS/DTS、24fps均一性、境界keyframe、
  full stream decodeを含む全項目pass

## 5時間推論・メモリ

### 並行と直列の直接比較

同じ431,935 framesを`parallel_models=true/false`以外は同一設定で処理した。

| 項目 | GPU並行 | 直列 | 直列の差 |
|---|---:|---:|---:|
| inference stage | 8,038.565s | 4,710.805s | -3,327.760s / -41.4% |
| 高速DINO role wall | 54.638fps | 175.876fps | 3.22倍 |
| 新顔 role wall | 59.015fps | 203.382fps | 3.45倍 |
| peak aggregate RAM | 約6.20GiB | 約3.35GiB | -46.0% |
| peak VRAM | 31,901MiB | 27,311MiB | -14.4% |
| face detections | 879,294 | 879,337 | 直列が43件多い |
| face observations | 459,330 | 459,330 | 同じ |
| face keypoints | 2,296,650 | 2,296,650 | 同じ |
| genital segmentations | 276,972 | 276,972 | 同じ |

並行実行:

- inference stage: 8,038.565秒
- role平均: 高速DINO 54.638fps、新顔59.015fps
- 工程全体: 2:27:41（結果包装・契約検査を含む）
- result packaging subprocess: 45.932秒
- peak aggregate RAM: 約6.20GiB
- peak VRAM: 31,901MiB
- GPU使用率: median 82%、peak 100%

両role稼働中のRSSは、最初の四分位中央値約6,123MiBから最後の四分位中央値
約5,960MiBへ減少した。顔終了後の高速DINO単独区間も約3.10GiBで横ばい。
推論SQLiteは動画長に応じて増えるが、推論process RAMは線形増加しない。

顔role終了時、高速DINOは328,480 framesまで進んでいた。その後残り103,455
framesを約588秒、約176fpsで処理したと推定でき、両model同時稼働時の平均
約44.9fpsから大きく回復した。直列実測175.876fpsとも一致し、GPU共有の
干渉は明確。

直列実行では高速DINO processが約2,450秒、新顔processが約2,118秒稼働した。
高速DINO RSSは初期化後の最初の四分位中央値3,161MiB、最後3,258MiB、peak
3,293MiBで、約100MiBのallocator高水位変動がある。一方、並行2時間試験では
同process RSSが最初3,107MiB、最後3,099MiBへ減少しており、反復回数に比例する
リークとは判断しない。新顔RSSは最初／最後とも2,959MiBで完全に横ばい。

### 顔出力の再現性

15分clipの完全な反復19周を比較した。

- 直列: 全周で`Face=21,004`、`Head=22,976`、valid keypoints=86,712。
- 並行: `Head=22,976`は一定だが、`Face=20,994..21,005`、
  valid keypoints=86,691..86,720。
- 性器側は件数、score集計、34,934,954 segmentation pointsの座標総和まで一致。

並行時も顔観測の構造と件数は保たれるが、Face有無、楕円、keypoint座標／validが
閾値近傍で微変動する。速度・資源・再現性の3点から、`parallel_models`は短尺の
明示opt-inに限定し、長尺既定は`false`とするのが妥当。

## 5時間後処理

polygon K3、max gap 30、eyes ellipse、high-precision cutsの結果:

- orchestration stage: 417.708秒
- pipeline各stage合計: 387.215秒
- 工程全体: 9:51.97
- cuts: 139（元clip内6×20回 + 反復境界19）
- tracked masks: 241,295
- final masks: 676,035（うちface privacy 419,964）
- editable keyframes: 505,540
- output SQLite: 約6.50GiB

| stage | time |
|---|---:|
| normalization | 44.299s |
| score policy | 7.839s |
| NMS | 10.912s |
| cut detection | 90.295s |
| tracking | 24.417s |
| polygon approximation | 9.949s |
| keyframe selection | 2.285s |
| polygon gap fill | 48.067s |
| exact IoU evaluation | 15.333s |
| face privacy masks | 45.100s |
| face merge | 21.647s |
| combined validation | 6.597s |
| integrated result SQLite | 約59.7s |

peak RAMは約1.09GiB。`gap_fill/polygon/interpolate.py`はkeyframes、targets、
outputを全件辞書／listで保持し、`evaluation/mask_iou.py`もreferenceとprediction
を全件decodeして辞書化する。これはデータ量依存であり、track単位streamingと
sorted merge evaluationへ変更できる。

## 5時間高速オーバーレイ

- production render: 222.955秒
- aggregate: 1,937.32fps
- parallel portion: 212.918秒
- concat: 10.036秒
- 6 NVENC workers、各約338–341fps
- peak aggregate RAM: 約2.33GiB（各worker約378MiB）
- VRAM: 4,874MiB
- GPU使用率: median 60%、peak 66%
- output: 約16.76GiB

production workerのRAMは処理時間に対して横ばい。6 worker数に比例するが動画長
には比例しない。独立validatorは全frame timestampをPython listへ保持するため
約351MiBまで増え、productionとは別に検証tool自体がO(frames)である。

## SQLite rev3監査

巨大／高速×新旧顔、性器のみ、顔のみを含む6 inference-result SQLiteと、
9 postprocess SQLiteで`sqlite_master`定義が完全一致した。

- schema: `video-mask-integrated-result`
- schema version: 1
- contract revision: 3
- compatibility profile: `stable-all-modes-v2`
- `PRAGMA integrity_check`: `ok`
- `PRAGMA foreign_key_check`: 0件

5時間最終SQLiteの主な件数:

| component | rows/status |
|---|---:|
| frames / raw inference | 431,935 / complete |
| detections | 1,156,266 |
| segmentations | 276,972 / complete |
| face observations | 459,330 / complete |
| face keypoints | 2,296,650 |
| tracked masks | 241,295 / complete |
| final masks | 676,035 / complete |
| cuts | 139 / complete |
| editable mask keyframes | 505,540 / complete |
| native polygon points | 981,257 / complete |
| native ellipses | 419,964 / complete |
| native rectangles | 0 / not_requested |

意味検査:

- frame index `0..431934`、重複なし、timestamp単調
- detection score範囲外0、bbox不正0
- segmentation point範囲外0
- 顔1観測あたりkeypointは全て5点、confidence不正0
- 顔ellipse半径不正0
- cutを跨ぐmask track segment 0
- final `(frame, track_id)`重複0
- keyframeに対応しないnative形状0
- segment外keyframe 0
- polygon point範囲外0

軽微なschema/meta品質課題:

1. 顔keypointのうち57,486点が画面外だが、57,385点は`valid=0`。`valid=1`の
   画面外は101点（有効点の約0.006%）で、上端を最大1.82px越える。raw model
   座標として保存する方針ならconsumer側でclipする。
2. genital keyframeの`source_detection_id`は全てNULL。顔geometryは
   `source_face_observation_id`へ接続されるが、性器のraw detectionまでの
   直接lineageは不足。
3. `video_streams`のwidth/height/fps/frame_countはあるが、codec、time base、
   pixel format、color情報はNULL。
4. `processing_runs.git_commit`はNULLで、モデルpathはあるがweight/content hashが
   ない。再現性metadataとして追加余地がある。

## 契約検査の速度

大容量DBでは完全性検査が主な非モデルコストになる。

- 3.1GB raw inference SQLiteの契約検査: 46.49秒（warm cache）
- result SQLite契約検査: 54.04秒（warm cache）
- 6.5GB final SQLite単独`integrity_check`: 68.27秒（cold寄り）

`validate_result_sqlite`は内部で`validate_inference_sqlite`、
`validate_mask_sqlite`、結果自身の`integrity_check`を呼ぶ。さらにrunnerは
同じ結果をpostprocess publish時とresult packaging判定時に再検査する。
同一inode/mtime/sizeに対する検査結果のrun内cache、cheap contract checkと
full integrity checkの分離、full check 1回化で、出力を変えず短縮できる。

## 自動テスト

- InstanceSegmentation inference: 60 passed、2 skipped、2 subtests passed
- orchestration: 29 passed、6 subtests passed
- postprocess: 57 passed、39 subtests passed
- overlay Python: 32 passed、2 subtests passed
- overlay native C++ mode tests: 4 passed

## Git状態

試験開始時点の`HEAD`は`1b9ed74`で`main == origin/main`。ただしschema rev3、
クラス別後処理、editable geometry、runner、overlay等には未コミットの変更と
新規ファイルが多数ある。この試験ではコミット／pushしていない。

## 推奨修正の優先順位

1. **長尺の`parallel_models`既定をfalseにする。** 短尺opt-in時も
   非決定性を明記する。
2. **polygon gap fillとexact evaluationをtrack単位streamingへ変更する。**
3. **同一SQLiteへの重複full integrity checkをrun内で1回にまとめる。**
4. **genital keyframe lineage、video codec/color/timebase、git commit、
   model content hashをmetadataへ追加する。**
