# DINOv3 ViT-L backbone ROI分類器 0.97到達レポート

## 結論

固定R6 group-disjoint splitと、実運用の検出score `>= 0.65`で目標を達成した。

| 構成 | val Macro-F1 / accuracy | held-out test Macro-F1 / accuracy | params | 分類器速度 |
|---|---:|---:|---:|---:|
| 高精度6-model ensemble | 0.98618 / 0.98757 | **0.97356 / 0.97762** | 10.30M | 294k ROI/s |
| 高速Gap320単体 | 0.98487 / 0.98618 | **0.97093 / 0.97549** | 0.436M | 1.46M ROI/s |

速度はRTX PRO 6000 Blackwell上の分類器forwardのみ。検出器、backbone、ROIAlign、転送は含まない。
選定後にheld-out testを一度だけ生成・評価し、test結果によるモデルや重みの変更は行っていない。

## 評価範囲

- train: 86,391 ROI (`34,229 / 29,889 / 22,273`)
- val全体: 5,597 ROI、score `>=0.65`: 4,343 ROI
- test全体: 5,705 ROI、score `>=0.65`: 4,244 ROI
- class: `male / female / junction`
- detector: ViT-L Co-DINO epoch-6 EMA
- detector SHA-256: `0337522008ea37f5abb10e5caa1071bc9e1c75f776516276df2a377a0c02d635`
- classifier input: final DINOv3 ViT-L mapから`RoIAlign(1024,4,4)`、stride 16、`aligned=True`
- metadata: detector score、bbox/mask面積比、log aspect ratio、mask/bbox比

元Mask ROI packとbackbone packは、train/valのimage ID、label、score、bbox、各メタ値を
canonical sortで完全照合した。比較時に変化したのは分類器入力特徴だけである。
held-out testもshard統合順だけが異なり、canonical sort後の全識別項目は完全一致した。

## 重要な解釈

score閾値なしの全ROIでは、最良4x4 ensembleはval Macro-F1 `0.96511`、
test `0.96006`であり、0.97には届かない。低確信度検出を含めた値と、実運用閾値の値を
混同してはならない。score `>=0.65`ではval/testの双方で0.97を超えた。

test高精度ensembleの混同行列:

```text
               predicted
actual       male  female  junction
male         1883       7        18
female          4    1381        41
junction        9      16       885
```

## 探索結果

| 入力・手法 | val Macro-F1（全ROI） | 判断 |
|---|---:|---|
| 旧Mask ROI ensemble | 0.92575 | backboneに置換 |
| DINOv3 backbone 4x4 Conv単体 | 0.96325 | 最強単体（全ROI） |
| DINOv3 backbone 4x4 ensemble | 0.96511 | 全ROI最良の単一scale |
| backbone+Co-DINO Mask融合 | 0.96076 | 不採用 |
| 4x4 token Transformer | 0.95530 | 過学習、不採用 |
| backbone 7x7 | 0.96119 | 容量増に対して悪化、不採用 |
| 1.25x context ROI | 0.96130 | 単体では悪化 |
| 4x4/7x7/context cross-scale | 0.96579 | 改善小、運用複雑化のため不採用 |

## 推奨アセット

- manifest: `configs/teacher_vitl_codino/backbone_recommended_models.json`
- train cache: `data/teacher_vitl_codino_backbone/semantic3_train.pt`
- val cache: `data/teacher_vitl_codino_backbone/semantic3_val.pt`
- test cache: `data/teacher_vitl_codino_backbone/semantic3_test.pt`
- 高速checkpoint: `outputs/teacher_vitl_codino_backbone/backbone_gap320_k3_seed44/checkpoints/best.pt`
- 高精度評価: `outputs/teacher_vitl_codino_backbone/final_ensemble_test_score065.json`
- 高速評価: `outputs/teacher_vitl_codino_backbone/final_fast_test_score065.json`

通常は高速Gap320を推奨する。test Macro-F1が0.97093で目標を満たし、ensembleより約5倍高速、
checkpointは約1.76MBである。追加の約0.0026 Macro-F1が重要な場合のみensembleを使う。

## 再現コマンド

```bash
cd /home/kenshin/native_linux_transfer/dinov3_vitl_head_selection_v1_full/roi_classifier_training

# 元の検出・matching済みpackからbackboneだけを再実行する高速生成
./scripts/12_prepare_codino_backbone_cache.sh

# 2 GPUの主探索
./scripts/42_train_backbone_search.sh

# held-out test（モデル選定後のみ）
./scripts/15_prepare_codino_backbone_test_fast.sh

# 固定済み高精度・高速モデルを再評価
./scripts/49_evaluate_backbone_recommended.sh test
```

`12_prepare`は検出結果も再生成する厳密経路、`15_prepare...fast`は既存bbox/labelを固定して
backboneだけを再実行する高速経路である。高速再生成ツールは
`tools/regenerate_backbone_roi_from_pack.py`。4x4プローブでは厳密経路との差はFP16丸め程度
（平均絶対差0.000188、最大0.00684）だった。
