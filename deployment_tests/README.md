# 8時間 GUIデプロイ判定テスト

このディレクトリは、Mask Pipeline Studioを実際のElectron GUIから操作し、
推論・後処理・SQLite統合・overlay・GUI自体を8時間以内でデプロイ判定するための
テスト仕様です。製品pipelineをCLIから直接起動するケースは含めません。

Playwrightはボタン、タブ、入力キュー、Inspectorを人間と同じGUI経路で操作します。
終了後の読み取り専用QAだけがSQLite、manifest、MP4、ログ、一時ファイルを直接検査
します。fixture作成にFFmpegを使うことはありますが、製品処理の迂回には使いません。

## 基本方針

- 予算は480分。通常計画446分、不可避な再試験用reserve 34分です。
- V3（巨大Co-DINO）の負荷試験は2時間動画1本だけに集約します。
- それ以外のバッチ、中断、LIVE A/BはV3-lite（MH0）で検査します。
- V3とV3-liteは同じ`SegmentationFrame`契約、schema-v3 writer、後処理、統合、
  overlay経路を使います。ただし実装固有のロード・TensorRT・分類器接続は共有
  されないため、V3自身を2時間連続で検査します。
- 全直積は行いません。P0の一般的構成を完全に通し、残りは各GUI項目を最低1回、
  重要な相互作用をpairwiseで通します。
- 途中の予期しない例外、非0終了、GUI black screen、成果物破損は即P0 failureです。
- 「完了」と表示されても成果物検査に失敗したジョブは不合格です。

## ファイル

- [`TEST_PLAN.md`](TEST_PLAN.md): 時間配分、ケース構成、適応的な打ち切り規則
- [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md): deployment gateの合否判定
- [`RUNBOOK.md`](RUNBOOK.md): GUI操作、監視、証跡、終了時の手順
- [`FIXTURES.md`](FIXTURES.md): 短時間で媒体差を網羅する動画セット
- [`cases.json`](cases.json): 機械可読なケース、予算、coverage tag
- [`scripts/check_plan.py`](scripts/check_plan.py): 8時間予算と必須coverageの静的検査

既存のGUI実行基盤は`gui/scripts/gui-real-matrix.mjs`、成果物検査は
`gui/scripts/validate-gui-matrix.py`を再利用します。8時間版executorはこの
`cases.json`を読み、ケースごとに独立したElectron user-data directoryを使います。

## 実行結果の扱い

生成fixture、スクリーンショット、時系列resource sample、動画、SQLite、レポートは
`deployment_tests/work/`へ置きます。このディレクトリはGit追跡しません。最終的に
`deployment-report.json`と`deployment-report.md`へ、pass/fail、実測時間、速度、
ピークと傾向、成果物検査、未実施項目を残します。

計画自体の検査:

```bash
python3 deployment_tests/scripts/check_plan.py
```

このコマンドは製品pipelineを実行せず、ケース予算、ID重複、P0/P1/P2、必須coverage、
V3長時間実行禁止を検査します。
