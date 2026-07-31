# Best Pseudo MH0 backbone ROI classifier（2026-08-01）

## 結論

正しいBest Pseudo MH0のDINOv3 ViT-S+最終backbone特徴を使う単体Spatial GAP分類器で、
運用閾値score 0.65以上のheld-out test Macro-F1 **0.954748**を達成した。
目標0.95を超えたため、held-out testを見た再調整や大型ensemble化は行っていない。

## 固定した検出器

- run: `03_seed43_pseudo_rotation`
- checkpoint: `best_segm_mAP_epoch_7.pth`
- SHA-256: `31f652eece3bfe91e72fa1bc3edd00216e265c4026dcbf817a39842474d5752d`
- weight source: 通常の`state_dict`（評価用EMA weight）
- `ema_*` bufferは使用しない
- bbox mAP 0.464、mask mAP 0.436

## 入力とアーキテクチャ

```text
DINOv3 ViT-S+ final backbone [384,H/16,W/16]
  -> ROIAlign 4x4
  -> 1x1 Conv 384->320
  -> Depthwise 5x5 Conv
  -> 1x1 Conv 320->320
  -> Global Average Pooling
  -> geo_v2 metadata 5 dims
  -> Linear 3 classes
```

- class order: `male / female / junction`
- parameters: 236,178
- checkpoint: 960,231 bytes
- checkpoint SHA-256: `0696f6e92572ba48d534caebb49f1a10282efc247ab470d49dabd8eace429f29`

## データ生成

既存Best Pseudo MH0 packのbbox・label・scoreをテンプレートとして保持し、画像から
backbone特徴だけを再生成した。identity fieldsはtrainで完全一致、valではimage idで
stable sort後に完全一致した。

| split | images | matched ROI | 6-shard wall time | throughput |
|---|---:|---:|---:|---:|
| train | 67,304 | 86,169 | 362.71 s | 185.56 image/s |
| val | 4,387 | 5,588 | 23.69 s | 185.20 image/s |
| held-out test | 4,594 | 5,681 | 99.26 s | 46.28 image/s |

testは選定後に正しいMH0検出器を含むfull detection/matching経路で生成した。

## モデル選定

同じSpatial GAP familyだけで6候補を学習した。

| run | validation Macro-F1 |
|---|---:|
| gap192 k3 seed42 | 0.976115 |
| gap256 k3 seed43 | 0.977584 |
| gap320 k3 seed44 | 0.978328 |
| **gap320 k5 seed46** | **0.981008** |
| gap384 k3 seed47 | 0.977481 |
| gap384 k5 seed45 | 0.977784 |

## 最終評価（score >= 0.65）

| split | ROI | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| validation | 3,845 | 0.981008 | 0.982835 |
| held-out test | 3,642 | **0.954748** | **0.961834** |

held-out test confusion matrix:

```text
actual       male  female  junction
male         1623       6        24
female          3    1095        74
junction        6      26       785
```

閾値なし監査ではvalidation 0.933467、test 0.917760である。
したがって0.95は検出器score 0.65以上という運用条件を含む値である。

## 推論速度

RTX PRO 6000 Blackwell、batch 2048、分類器単体:

- 1,504,916 ROI/s
- 1.3609 ms / batch
- 0.000664 ms / ROI

DINOv3 backboneはMH0検出器で計算済みの最終特徴マップを再利用すること。
分類器のためにbackboneを再実行しない。

## 成果物

- `configs/student_pseudo_mh0/backbone_recommended_model.json`
- `outputs/student_pseudo_mh0_backbone/mh0_backbone_gap320_k5_seed46/checkpoints/best.pt`
- `outputs/student_pseudo_mh0_backbone/mh0_backbone_gap320_k5_seed46/val_score065.json`
- `outputs/student_pseudo_mh0_backbone/mh0_backbone_gap320_k5_seed46/test_score065.json`
