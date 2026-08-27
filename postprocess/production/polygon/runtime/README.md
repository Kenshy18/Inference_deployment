# Polygon optimizer runtime

ここはProductionポリゴンの実行エンジンです。公開stageは一つ上の
`stage.py`、入力準備は`preparation.py`、最終SQLite化は`materialize.py`が
担当します。このディレクトリは、準備済みのトラックをキーフレーム列へ変換する
処理だけを所有します。

## 実行順序

```text
run.py
  -> coordinator.py
    -> optimizer_process.py
      -> optimizer_factory.py + optimizer_adapters/
        -> optimizer_kernel.py
      -> candidate_generation.py
      -> hard_recall_dp.py
      -> pair_vote.py
      -> topology.py
      -> reporting.py
```

## モジュールの責務

| 場所 | 責務 |
|---|---|
| `run.py` | Production stageから呼ばれるCLI入口 |
| `coordinator.py` | クラス別worker、環境、成果物mergeの調整 |
| `optimizer_process.py` | 1クラス分の候補・DP・pair-vote・監査を実行 |
| `runtime_config.py` | 凍結profile、環境キー、頂点数policy |
| `candidate_config.py`, `candidate_palette.py` | 候補形状の契約と役割 |
| `candidate_generation.py` | 各フレームの候補形状を生成しDP graphへ接続 |
| `hard_recall_dp.py` | 最小Recallをhard制約にした多状態DP |
| `native_runtime.py` | OpenCV互換の厳密区間評価を数値kernelへ接続 |
| `pair_vote.py` | 固定キーに対する厳密IoU改善 |
| `topology.py` | 反転・自己交差を拒否するhard gate |
| `reporting.py`, `diagnostics.py` | manifest、品質指標、違反分類 |
| `spatial_builder.py`, `spatial_support/` | トラック全体で対応する頂点を配置・修復 |
| `optimizer_kernel.py` | 数値計算本体。I/Oや配布設定を持たない |
| `optimizer_adapters/` | kernelへProductionのI/O・native DPを注入 |
| `optimizer_adapters/native_dp_kernel.cpp` | penalty DPとRecall修復scoreのC ABI kernel |
| `native_interval/` | C++/OpenCV厳密raster evaluatorのsource/build定義 |

`geometry.py`と`optimizer_adapters/geometry.py`は用途が異なります。前者は候補生成で
使う幾何演算、後者は凍結kernelの型とProductionの型を接続するadapterです。

## 歴史的な識別子

環境変数やmanifest互換のため、内部に`PHASE1`、`PHASE2`という文字列が残ります。
現在の処理が二つの別製品へ分かれている意味ではありません。

- 旧`PHASE1`: raw形状と厳密Recall evaluatorをkernelへ接続する基底adapter
- 旧`PHASE2`: 複数候補、hard-Recall DP、pair-voteを適用するProduction optimizer

新しいファイル名・公開API・ドキュメントにはphase番号を増やさず、処理の責務を
表す名前を使います。既存の環境キーやmanifest IDを変更する場合は、配布済みGUIと
再現用manifestの移行を伴うため、単独の互換性変更として扱ってください。

## 依存境界

- `postprocess/experiments/`と削除済みvendorコードをimportしない
- 動画をdecodeしない。入力は準備済みSQLiteとpolygon geometryのみ
- CUDAは候補screeningに使えるが、採用辺と最終出力はnative exactで監査する
- Recall、topology、頂点数policyを暗黙fallbackで緩めない
