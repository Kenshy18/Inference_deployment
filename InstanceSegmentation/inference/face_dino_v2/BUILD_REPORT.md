# Face DINO v2 実装・ビルド・検証レポート（2026-07-28）

## 結論

`inference/`の第2顔モデルとして`face_dino_v2`を追加した。既存の
`rtdetr_head_face`と同じ`ObjectDetectionAdapter`、隔離プロセス、SQLite schema v2、
atomic orchestrationへ接続しつつ、モデル内部では楕円・キーポイント・occlusionを
含むrich出力を維持する。

PyTorch checkpointから全TensorRT資産を新規生成するビルダーを実データで実行し、
171.42秒で完了した。生成後は元の学習workspaceを参照せず、モデルフォルダ内の
checkpoint、runtime source、DINOv3重み、engine、pluginだけで動画推論できた。

## リポジトリ契約への接続

- model ID: `face_dino_v2`
- task: `object_detection`
- backend: `tensorrt-fast`
- fixed batch: 8
- model tensor: `[8, 3, 736, 1280]`
- source image coordinates: half-open XYXY
- subprocess isolation: 既存`orchestration/model_process.py`をそのまま利用
- persistence: 既存`DetectionFrame`→SQLite

SQLiteには次を保存する。

- `Head`: Co-DINOのHead boxとscore
- `Face`: Face存在判定がtrueの場合だけ、予測楕円のaxis-aligned envelopeと
  Face score

このため後頭部はHeadだけとなり、HeadとFaceの1対1強制には戻らない。

共有schemaに専用tableがない楕円、5キーポイント、eye/nose/mouthクラス、
visible/occluded、各確率は`FaceDinoRuntime.predict()`のGPU tensorとして利用できる。

## 高速engine

| component | precision | artifact |
|---|---|---|
| DINOv3 ViT-S+/16 + SFP | mixed FP16/FP32 | `engines/backbone_neck.engine` |
| compact Co-DINO query encoder | mixed FP16/FP32 + SM120 MSDA | `engines/query_encoder.engine` |
| compact Co-DINO decoder | FP32 | `engines/decoder.engine` |
| Face/ellipse/keypoint/occlusion | FP16 | `engines/attribute.engine` |
| BGR/resize/letterbox/normalize | SM120 CUDA | `plugins/face_preprocess_fused_sm120.so` |

query encoderのSoftmax/LayerNorm 3層、backboneの精度敏感37層はFP32を維持する。
engine bundleは全engine/pluginのsizeとSHA-256を検証し、checkpoint SHA-256
`0f1021887b99019fd0de12eacb2a474a7c2110f87ea6876b0f61ec7f2e385c52`
が一致しない場合も起動を拒否する。

## ビルダー

入口:

```text
face_dino_v2/trt/build_engines.py
```

実行工程:

1. ViT-S+とSFPを融合export/build
2. query encoderとdecoderをexport/build
3. attribute branchをexport/build
4. SM120融合前処理pluginをcompile
5. engine/pluginをatomic bundleへ集約
6. checkpointと推論sourceをモデルフォルダへsnapshot
7. manifestを生成

全工程は直列で、`MAX_JOBS=1`を設定する。途中失敗時は完成bundleを公開しない。

生成bundle:

```text
artifacts/trt/fast-sm120-fixed-b8-v1/manifest.json
```

- engine/plugin合計: 約181 MiB
- packaged checkpoint: 約163 MiB
- runtime source + ViT-S+ weight: 約157 MiB

## Test精度

Test 528画像、720p letterbox、Val固定thresholdで評価した。

| 指標 | 結果 |
|---|---:|
| bbox AP | 0.749593 |
| bbox AP50 | 0.963339 |
| bbox AP75 | 0.870455 |
| bbox AR100 | 0.816998 |
| Face macro-F1 | 0.833104 |
| ellipse IoU | 0.742672 |
| point macro-F1 @ NME .05 | 0.874072 |
| point macro-F1 @ NME .10 | 0.911252 |
| matched point NME | 0.014799 |
| occlusion macro-F1 | 0.821496 |
| e2e occluded-point F1 | 0.513919 |

直前の検証済みengineとbbox AP/AP50/AP75/AR100は完全一致した。attribute側の差は
楕円IoU `+0.000041`、point F1@.10 `+0.000329`、occlusion macro-F1
`-0.001626`で、再ビルド差として許容範囲である。

詳細:

```text
reports/test_20260728/metrics.json
```

## 動画速度

RTX 5090、1920×1080動画、800 frames、B8、warmup 3で測定した。

- compute: 133.018 images/s
- wall: 127.009 FPS
- Head rows: 950
- Face rows: 949
- SQLite integrity: `ok`
- 元画像座標範囲違反: 0

この測定は動画decode、融合GPU前処理、全モデル、GPU→CPU契約変換、
非同期SQLite保存を含む。描画・動画encodeは含まない。

詳細:

```text
reports/benchmark_800_20260728.json
```

## 動作確認

- 新規モデル単体CLI: 800 frames成功
- 統一`run_inference.py --face-model face_dino_v2`: 32 frames成功
- unified SQLite: `integrity_check=ok`
- unified `model_executions`:
  `face_dino_v2 / tensorrt-fast / face_detection / object_detection`
- rich runtime:
  `ellipses [N,5]`、`keypoints [N,5,2]`、
  eye/nose/mouth、visible/occludedを実GPU出力で確認
- inference repository全テスト: `38 passed`

## 制約

現在のbundleはTensorRT 10.13、CUDA 12.9、RTX 5090（SM120）向けである。
異なるGPU architectureでは同じcheckpointから、そのGPU向けengineを再ビルドする。
