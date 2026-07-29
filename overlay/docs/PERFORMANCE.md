# Low-level C++/CUDA overlay experiment

実施日: 2026-07-26

## 結論

現行OpenCV実装を変更せず、独立したC++/libav/CUDA実装で動画frameをCPUへ
取り出さない経路を実装した。

```text
MP4 -> libavformat -> NVDEC -> CUDA NV12 span blend -> NVENC -> MP4
                                    ^
postprocess SQLite -----------------+
```

同程度の容量、ラベルなしmask overlayで比較した長尺中央値:

| 解像度 | 現行OpenCV | C++ CPU合成 | C++ 完全GPU | OpenCV比 | 動画1分の処理時間 |
|---|---:|---:|---:|---:|---:|
| 1080p | 462.51 fps | 1211.12 fps | 1376.02 fps | 2.98倍 | 約1.31秒 |
| 4K | 131.72 fps | 434.92 fps | 558.86 fps | 4.24倍 | 約3.22秒 |

完全GPU版はCPU合成版からさらに1080pで1.14倍、4Kで1.28倍になった。入力動画の
実時間に対して1080pは約45.9倍速、4Kは約18.6倍速である。

## 実装

### CPU経路

```text
MP4
  -> libavformat demux
  -> libavcodec software decode
  -> YUV420Pへ整数scanline/span blend
  -> libx264 / NVENC
  -> libavformat MP4 mux
```

### GPU経路

```text
MP4
  -> libavformat demux
  -> NVDECがCUDA NV12 hardware frameを生成
  -> custom CUDA kernelが同じframeへ直接blend
  -> 同じAVHWFramesContextをNVENCへzero-copy入力
  -> libavformat MP4 mux
```

Python frame loop、OpenCV、YUV-BGR-YUV往復、BGR24 rawvideo pipe、
GPU-CPU-GPU転送を使用しない。CUDA合成は専用non-blocking streamで実行し、
NVENC全体を待たず合成streamだけを同期する。

SQLiteは`immutable=1`のread-only接続で開き、
`masks(frame, track_id, polygons)`を`frame, track_id`順に読む。track色はproduction
実装と同じ`SHA-256("track:" + track_id)`から作り、BT.709 limited-range YUVへ
変換する。

複数maskが重なる場合も結果が実行ごとに変わらないよう、fillはpolygon順、
outlineはtrack順のCUDA batchとして投入する。同一batch内は同じ色・alphaかつ
非重複scanlineである。

複数workerではpacket PTSを一度だけpresentation順へ索引化し、担当開始frame
直前のkeyframe PTSへseekする。seek後はdecoded frame ordinalをSQLiteの
`frame_index`へ対応させる。各segmentは独立encode後、FFmpeg concat demuxerで
再encodeなしに結合する。

## 最適化の段階

| 段階 | 1080p | 4K | 主な変更 |
|---|---:|---:|---|
| 現行OpenCV | 462.51 | 131.72 | BGR/OpenCV/rawvideo |
| 初期C++ YUV直結 | 1100.29 | 404.11 | Python/OpenCV/BGR pipe除去 |
| 整数span + decode 12 threads | 1211.12 | 434.92 | pixel浮動小数点loop除去 |
| NVDEC + CUDA + NVENC | 1376.02 | 558.86 | GPU-CPU-GPU転送除去 |

単純な`--hw-decode`、つまりNVDEC後にCPUへdownloadして合成し再uploadする経路は、
約1分の測定で1080p 681.64 fps、4K 358.86 fpsだった。転送が増えるため採用しない。

CUDA版の途中では`cudaDeviceSynchronize()`がNVENC処理まで毎frame待っていた。
専用streamの`cudaStreamSynchronize()`へ変更後、1080p 300 framesの単体処理は
257.86 fpsから321.87 fpsへ改善した。

## 条件

- CPU: Intel Core Ultra 9 285K
- GPU: NVIDIA GeForce RTX 5090
- CUDA: 12.9、`sm_120`
- FFmpeg SDK: 8.1 GPL shared build
- 入力: 5290 frames、29.970 fps、176.509667秒
- mask rows: 4025
- 1080p: 8 Mbps、NVENC p1、6 worker
- 4K: 26.2 Mbps、NVENC p1、4 worker
- mask alpha: 0.32
- outline thickness: 2
- H.264 High、yuv420p、BT.709 limited-range
- ラベルなし、音声なし

4K入力は負荷再現用に4K変換し、SQLite座標をscaleしたもの。4Kモデル推論の
mask精度評価ではない。

## 完全GPU長尺測定

順序保証を含む最終実装の3試行:

| 解像度 | trial 1 | trial 2 | trial 3 | 中央値 |
|---|---:|---:|---:|---:|
| 1080p NVENC6 | 1376.02 | 1363.71 | 1410.32 | 1376.02 fps |
| 4K NVENC4 | 554.35 | 560.02 | 558.86 | 558.86 fps |

最終concatを含む5290 framesの中央値は1080p 3.844秒、4K 9.466秒。短い動画では
encoder初期化、keyframe seek、MP4 finalize、concatの固定費が相対的に増える。

`faststart`なしのconcatは1080p約0.14秒、4K約0.39秒、`faststart`ありは
それぞれ約0.16秒、約0.45秒だった。画質と容量に影響しないため、速度優先の
既定は`faststart`なしとし、必要時だけ`--faststart`を使用する。

## worker探索

約1分相当、完全GPU経路:

| worker | 1080p | 4K |
|---:|---:|---:|
| 1 | 471.6 | 208.5 |
| 2 | 725.9 | 351.7 |
| 3 | 852.7 | 440.7 |
| 4 | 900.5 | **459.7** |
| 5 | **919.1** | 446.9 |
| 6 | 892.0 | 455.2 |

長尺では固定費が薄まるため、1080pは5 worker中央値1403.98 fpsに対し、
6 worker中央値1412.92 fpsだった。ただし差は1%未満で試行変動もある。
順序保証版の最終方針は1080p 6、4K 4とする。

少なくとも8 NVENC sessionの起動自体は成功したが、1080p長尺は7 worker
1393.22 fps、8 worker 1338.38 fpsへ低下した。sessionを起動できる数と最速の
並列数は同じではない。

CPU libx264と完全GPU workerの混成も測定した。最良候補でも約1分相当で
1080p 875.6 fps、4K 448.7 fpsに留まり、GPU専用構成を超えなかった。特に4Kは
CPU encodeがmemory bandwidthを奪うため、今回の実装ではGPU専用が適切である。

## production主要モード対応

速度上限確認後、同じzero-copy経路へ次を追加した。

- unified inference SQLiteのraw segmentation読取
- classificationがある場合のclass/score優先
- postprocess SQLiteのtracked/finalと任意label
- `role=face_detection`の顔box
- finalへの顔追加
- track/class/scoreのCUDA ASCII label
- 入力音声の再encodeなしstream copy
- start/end frameに合わせた音声timestampの0始まり補正
- segmented処理後の音声再mux
- atomic動画出力とatomic JSON manifest
- SQLite schema version、動画width/height/FPS、最大frameの検証
- GPU index指定

1080p、5290 frames、NVENC6、8 Mbpsの各モード実測:

| モード | masks | faces | label | audio | fps | 動画1分の処理時間 |
|---|---:|---:|---|---|---:|---:|
| raw | 4061 | 0 | あり | なし | 1343.26 | 1.34秒 |
| tracked | 4025 | 0 | あり | なし | 1406.85 | 1.28秒 |
| final | 4025 | 0 | あり | なし | 1411.41 | 1.28秒 |
| faces | 0 | 17031 | あり | なし | 1290.47 | 1.39秒 |
| final + faces | 4025 | 17031 | なし | なし | 1368.12 | 1.32秒 |
| final + faces | 4025 | 17031 | あり | copy | 1329.40 | 1.35秒 |

最重量構成でも現行OpenCVのmask-only測定462.51 fpsを大きく上回る。顔4件前後/
frameのboxとlabel生成が主な追加負荷で、音声stream copyと再muxの固定費は
全長測定で約0.14秒だった。

最重量構成では、顔boxとASCII glyphの水平・垂直線を線上の全pixelごとのdiscへ
展開していた重複spanを、同じ画素集合を表すscanline 1本へ圧縮した。単体300
framesのCUDA描画時間は0.1439秒から0.1144秒へ20.5%短縮し、全体は279.40から
298.46 fpsへ6.8%向上した。5290 frames、NVENC6の3試行中央値は1257.29から
1329.40 fpsへ5.7%向上した。

この変更後も同じCPU lossless基準に対するVMAF 97.945708、
float SSIM 0.998685、PSNR-Y 46.733943で、変更前の測定値と小数6桁まで一致した。

音声付き分割処理では、当初FFmpegの`-shortest`が音声の約20 ms短い入力で映像を
1 frame削ることを完全decodeで発見した。音声範囲は入力側`-t`で制限し、
`-shortest`を除去した最終版では映像5290 framesを維持している。

## 容量

| 解像度 | 現行OpenCV | 完全GPU | 差 |
|---|---:|---:|---:|
| 1080p | 7.960 Mbps | 8.013 Mbps | +0.7% |
| 4K | 26.194 Mbps | 26.262 Mbps | +0.3% |

容量差は1%未満で、速度差を容量増加で作った結果ではない。

## 画質

C++ CPU合成のCRF 0出力をlossless基準とし、完全GPU版の先頭300 framesを同じ
bitrateで評価した。

| 解像度 | VMAF | float SSIM | PSNR-Y |
|---|---:|---:|---:|
| 1080p 完全GPU | 96.3792 | 0.998669 | 47.007 |
| 4K 完全GPU | 96.4843 | 0.999642 | 49.435 |

比較対象のCPU合成版は1080p VMAF 96.2838、4K 96.3453であり、完全GPU化による
画質低下は観測されなかった。NVDECとsoftware decode、NVENCの実行順による
微差があるため、encoded bitstream同士のpixel完全一致を要件にはしていない。

OpenCVとのmask境界差は以前のlossless比較で、変更画素領域IoU平均0.9749、
最低0.9302、共通変更領域のoverlay delta平均絶対差4.78/255だった。差の中心は
OpenCV `LINE_AA`と整数scanline/Bresenham境界である。

顔boxとCUDA labelを含むfinalの先頭300 framesも、C++ CPU CRF 0描画を基準に
8 Mbps GPU出力を測定した。VMAF 97.9457、float SSIM 0.998685、
PSNR-Y 46.734で、追加primitiveを含めても画質劣化は観測されなかった。

## 完全性

- 1080p代表出力: 1920x1080、5290 frames、176.509667秒
- 4K代表出力: 3840x2160、5290 frames、176.509667秒
- segment合計frame数: 5290
- segment合計mask rows: 4025
- 両出力のFFmpeg全frame decode error: 0
- final + faces + labels + audio代表出力:
  映像5290 frames/176.509667秒、AAC 176.490秒、両stream 0秒開始
- final + faces + labels + audioのFFmpeg全stream decode error: 0
- 出力bitstream: H.264 High、yuv420p、BT.709 limited-range、B-frame 0
- C++/CUDA build成功
- Python benchmark runner compile成功
- native integration test成功
- repository orchestrationの`fast` execution modeとして4モード実行成功
- 実データ3600-3899の各出力は映像300 frames、AAC 0秒開始、decode error 0
- 同区間の描画数はraw 68 masks、tracked/final 55 masks、
  final/faces 974 faces

### PTS gap入力のフレーム完全性修正（2026-07-29）

旧実装は`best_effort_timestamp`を平均FPSの格子へ丸めて元frame番号を復元した。
この方法では、名目上CFRでもstream copyやconcatによりPTS gapを含む入力で
frame番号が飛ぶ。実データ由来の15分入力には通常3,750 tickに対して
15,000 tickのgapが1か所、20回反復した5時間入力には15,000／5,640 tickの
非均一間隔が計38か所あり、旧出力はそれぞれ1／67 frames不足した。

修正後はFFprobe packet indexをPTS順に並べ、workerごとに直前keyframeの
frame ordinalとPTSを渡す。native rendererはPTSをframe番号へ換算せず、
decodeごとにordinalを1増やす。要求範囲と出力数が一致しない場合はfailする。

| 条件 | 修正前 | 修正後 |
|---|---:|---:|
| 15分全長 | 21,601 / 21,602 frames | 21,602 / 21,602 |
| 5時間全長 | 431,868 / 431,935 frames | 431,935 / 431,935 |
| 5時間 renderer合計 | 222.955秒、1,937.32fps | 208.349秒、2,073.14fps |
| 5時間packet索引 | なし | 2.383秒 |

修正後の5時間出力は、frame/packet数、0秒開始、PTS/DTS単調性と24fps均一性、
worker範囲、5境界のkeyframe、FFmpeg全stream decodeの全項目に合格した。
raw、tracked、final、faces、final＋facesも実入力のPTS gapを跨ぐ4 framesで
各4/4、2 worker各2/2となり、同じvalidatorを全件通過した。
音声copy付きfinal＋facesも同じgapを跨ぐ12 framesで12/12となり、映像検査に
加えてAACの0秒開始、preroll、packet／decoded frame連続性をすべて通過した。

## 現在の制限

- H.264 yuv420p/NV12入力のみ
- OpenCVのアンチエイリアスとpixel完全一致ではない
- CUDA labelは組み込みASCII fontで、日本語labelはproduction版同様に省略
- MP4以外の出力containerは未対応
- 音声は再encodeせずMP4へcopy可能なcodecが前提
- SQLiteは処理完了後の不変artifactであることが前提
- GPU経路はNVIDIA CUDA/NVDEC/NVENC専用
- build scriptのCUDA runtime pathは現在この環境のproduction runtime固定

標準Python/OpenCV経路の完全置換ではなく、統合CLIから明示選択する高速modeとして
本番化している。任意pixel format/container、OpenCV fontとの表示方針、
GPU非搭載時fallbackは今後の対応範囲である。
