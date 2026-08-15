# polygon14_keyframe_v1

14/16/18/20頂点のトラック整合ポリゴン近似と、凍結した既存のキーフレーム最適化を組み合わせたProduction実装です。内部profile IDは既存成果物との互換性のため維持しています。

## 処理契約

1. 入力SQLiteから、トラッキング後の毎フレーム・各連結成分の輪郭を読む。
2. 各トラック区間を14、16、18、20頂点の順に近似し、CPU native-exactで全フレームを検査する。
3. 同一区間では全フレーム・全成分の頂点数を固定し、最小の合格点数をDPへ渡す。Recallの参照である `run.gt_polygons` は変更しない。20点でもRecall 0.97を満たせなければ停止する。
4. `new_production_v1` と同一の状態候補・目的関数で、目標キーフレーム間隔を努力目標としてDPを実行する。
5. 固定されたキー位置に対し、per-key pair-voteでIoUを改善する。各変更は厳密な毎フレームRecall 0.97を条件とする。
6. 自己交差は品質ペナルティではなくハード制約とする。DPで選択された区間だけを遅延検査し、交差区間だけRecallを再検証して局所キーを追加する。pair-voteはIoU順で最良の正常候補を使い、正常候補がなければDP形状へ戻す。
7. キーフレームだけでなく、全整数出力フレームの線形補間も自己交差がないことを検証する。
8. CUDAは区間候補のスクリーニングに使用するが、採用辺と最終密フレーム列は厳密評価する。
9. 既存exporterへ渡すため、最終SQLiteのテーブル・列・shape表現は変更しない。

固定値は [config.py](config.py) の `Polygon14CandidateConfig` が唯一の定義です。監査JSONにも同じ契約を埋め込み、異なる設定の結果を同じ候補名で扱えないようにしています。

## 実行

```bash
cd /home/kenshin/inference_backend2
PYTHONPATH=/home/kenshin/inference_backend2/postprocess \
python -m experimental.production_candidate_polygon14.run \
  --intervals 1,3,6 \
  --output-root output/polygon14_keyframe_v1
```

短いスモークテストでは `--max-tracks 1 --labels 男性器` を追加できます。各intervalの `phase2_audit.json` と、ルートの `production_candidate_manifest.json` に契約・厳密Recall・処理時間を保存します。

## 品質基準と既知の範囲

- 空間近似: 各フレームで tracked source mask に対するRecall 0.97、IoU 0.95を品質ガードに使う。
- 時間最適化: 同じ tracked source mask に対する毎フレームRecall 0.97をハード制約にする。
- トポロジー: 各成分は単純ポリゴンであり、キーフレームと補間後の全整数フレームで自己交差を許容しない。
- 頂点数: 1連結成分につき14/16/18/20。トラック区間内では選択数、成分スロット、頂点IDを固定する。
- 既存の5トラック・3,389共通フレーム比較では、旧方式に対して平均IoUが `0.948912 -> 0.959923`、最低Recallが `0.754942 -> 0.970008`、キー数が `1113 -> 1116` だった。
- 複数成分スロットの実装と単体テストは含むが、上記実データ比較は単一成分の5トラックが対象。昇格前には全クラス・複数成分・画面端を含むSQLite検証を追加する。

この候補でいう「元マスク」は生のニューラルネット出力そのものではなく、ポリゴン段へ入力されたトラッキング済みsource maskです。空間段と時間段のRecallは、どちらもこの同じ参照を使います。
