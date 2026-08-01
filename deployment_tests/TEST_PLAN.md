# テスト計画

## 1. 8時間の配分

| 区分 | 予定 | 内容 |
| --- | ---: | --- |
| fixture・preflight | 15分 | Windows release hash、install、runtime、空き容量、fixture probe、GUI起動 |
| GUI・Inspector操作 | 20分 | 簡単／詳細、全基本control、Dry Run、black-screen回帰 |
| V3 2時間動画 | 175分 | V3単独＋分類＋後処理＋性器overlay、全時間resource監視 |
| 一気通貫モデル行列 | 77分 | V3-lite中心、旧モデル短時間、顔mask全形状 |
| 動画・overlay互換 | 44分 | 4K、縦、VFR、container/codec/audio、3 encoder方式 |
| 中断・復旧 | 45分 | 4 phase停止、終了、再起動、resume、再処理、衝突 |
| バッチ・Live | 40分 | V3-lite 10本batch、Live A/B |
| 全成果物検証・目視 | 30分 | SQLite、MP4、品質、cleanup、性能、resource slope |
| 報告 | 10分 | gate集計、未実施項目、最終判定 |
| reserve | 34分 | V3実測ぶれと失敗の1回だけの再現 |
| 合計 | **480分** | reserveを使い切っても8時間以内 |

通常ケースの合計予算は391分、固定作業は55分で、34分をreserveとして残します。
V3の2時間caseはG00直後に開始し、予算超過が見えた時点でP1を先に削ります。

## 2. V3とV3-liteの役割分担

両モデルはregistry上で`INSTANCE_SEGMENTATION`として登録され、どちらのadapterも
共有`SegmentationFrame`へ変換されます。永続化は同じschema-v3 writerを使い、
後処理以降はモデルID以外の公開契約が同じです。したがって以下をV3-liteへ委譲
できます。ただし今回はV3自身の長時間安定性もdeployment gateに含めます。

- batch scheduler、出力suffix、次ジョブへのresource解放
- 中断、GUI終了、resume、再処理
- 後処理、顔処理、overlayの組み合わせ
- container、resolution、FPS、audioの入力行列

次はV3固有なので、2時間動画1本でまとめて検査します。

- GUIからV3とTensorRT高速engineを選択できる
- engine、classifier、backbone特徴接続がロードできる
- canonical 3クラス分類を含む生出力が保存される
- V3出力が同じ後処理・固定schema・詳細overlayへ到達する
- 2時間を通した速度、メモリ、VRAM、FD、SQLite増加とcleanup

V3のstress/soakは`S00_v3_120m_single_load`だけを許可し、入力はちょうど120分と
します。それ以外のV3負荷試験を追加しません。V3のcaseではFace V2を同時実行せず、
顔経路はG02/G04で検査して、2時間caseの時間と原因切り分けを守ります。

## 3. ケース戦略

### P0: deployment blocker

- `G00`: GUI全基本操作とInspector coverage。短いDry Runのみ。
- `S00`: V3で2時間動画を1本だけ処理。分類、mixed後処理、性器簡易／詳細overlay。
- `G02`: V3-lite + Face V2の推奨構成。簡易・詳細、Liveあり。
- `G03`: V3-lite、後処理なし、生出力詳細。
- `G04`: Face V2のみでmaskなし、目元長方形、目元楕円、顔全体。
- `M01`: MP4/MOV/MKV、H.264/H.265、24/29.97/30/60、縦横、音声有無を
  同じ入力queueへ入れる短尺batch。
- `M02`: 4K短尺で推論、後処理、詳細overlayまで通す。
- `R01`: 性器推論、顔推論、後処理、overlayの各phaseでGUI停止。
- `R02`: 実行中にGUIを閉じ、再起動後の状態と安全な再処理／resumeを確認。
- `S02`: 10本batch。失敗しないこと、queue順序、suffix、job間resource解放。

### P1: 一般的だがP0より発生頻度が低い

- V1/V2/Face V1の短尺起動と固定schema
- CPU、NVENC、高速overlayの一致
- VFR、非ゼロPTS、長GOP、B-frame、音声なし
- 出力名衝突、再処理、エクスプローラー、GUI終了後再起動
- Live ON/OFFの速度・応答性A/B

### P2: reserveが残った場合

- 未対応codec、破損containerの追加パターン
- PyTorch低速engineの追加ケース
- 60 FPSと回転metadataの追加目視

P2を省略した場合はpassに含めず、最終レポートへ`not_run_due_to_budget`として記録
します。P0を省略してdeployment passにはできません。

## 4. fixture設計

実映像から15～120秒の内容が異なる短尺を作り、次の合成変種を作ります。

| fixture | 主目的 |
| --- | --- |
| `golden_1080p2997_h264_aac.mp4` | 顔・性器・cut・音声を含む短時間基準 |
| `landscape_720p24_h265.mkv` | H.265/MKV |
| `portrait_720x1280_30_h264.mp4` | 縦動画・座標scale |
| `uhd_2160p24_h265_noaudio.mp4` | 4K、音声なし、負荷 |
| `vfr_pts_gap_h264.mp4` | VFR、PTS差、分割seek |
| `long_gop_bframes.mov` | MOV、長GOP、B-frame、AAC |
| `short_60fps.mp4` | 60 FPS、進捗・frame count |
| `unicode_日本語 space.mp4` | 日本語・空白path |
| `invalid_truncated.mp4` | 期待される入力エラー |
| `codino_120m_mixed.mp4` | V3専用。複数sceneと密度を混ぜた120分 |

4Kや長時間は空間／時間方向へ単純複製するだけにしません。顔数、mask数、cut密度が
区間で変化する素材を連結し、track数とSQLite密度の増減を含めます。長時間fixtureは
stream copyを優先し、fixture作成時間を20分のpreflight予算に含めます。

## 5. GUI操作coverage

全直積ではなく、次の条件を満たします。

1. 全ての表示controlを少なくとも1回、pointerまたはkeyboardで操作する。
2. 各model/backendの許可された組み合わせを少なくともDry Runする。
3. P0構成はDry Runだけでなく実ジョブを完了させる。
4. 各overlay presetを最低1本生成する。
5. 各顔mask形状、各性器shape、異なるKF/max-gapを成果物SQLiteで確認する。
6. 無効な組み合わせが非表示、disabled、またはGUI検証エラーになることを確認する。
7. Source、Inspector、Monitor STATUS/LIVE、Console、input/output queueを操作する。
8. 深いInspector項目を操作してもdocument rootがscrollせず、black screenにならない。

## 6. 適応的スケジューリング

executorは各ケース終了時に残り時間を再計算します。

- S00はG00直後、開始30分以内に実行を開始する。
- 8時間終了の60分前までに残りの全P0実行を開始済みにする。
- 実測が予定の1.5倍を超えたら、未実行P2を即座に省略する。
- 残り時間が「未実行P0予算 + 45分の最終検証」を下回ったらP1を省略する。
- 失敗ケースの再実行は、証跡を保存したうえで同一条件1回だけ。
- 2回目も失敗したP0はfail確定。修正作業をテスト時間へ混ぜない。
- 7時間35分で新しいpipeline jobを開始しない。
- 7時間45分で実行中jobをGUIから停止し、15分で成果物とレポートを確定する。

## 7. 並行できるもの

GPU pipeline同士は並行しません。次はjobと同時に行います。

- 0.5～1秒間隔のCPU/GPU/RAM/VRAM/温度sampling
- GUI event-loop heartbeatとpreview受信記録
- 出力ディレクトリ容量、FD、process treeのsampling
- 前ジョブの読み取り専用SQLite/MP4検査（I/O競合が性能測定を汚すため、速度を
  gateするgolden run中は停止）

## 8. 品質の目視

全動画を通しで見る代わりに、各代表出力から以下を自動抽出してcontact sheetを作り、
人間が署名します。

- 最初／最後、worker境界前後
- detection最多、顔最多、maskなし
- cut前後、キーフレーム、補完フレーム
- 低確信度の顔／頭部が存在するframe
- portrait、4K、VFRの代表frame

位置、ちらつき、track跨ぎ、class/ID、日本語、CUT、顔thresholdを確認します。
