# Mask Pipeline

動画に対するインスタンスセグメンテーション推論と、そのマスク後処理を行う
ワークスペースです。

## 処理フロー

```text
Windows GUI
  -> orchestration（実行計画・進捗・成果物管理）
    -> InstanceSegmentation（性器・顔の推論）
    -> postprocess（NMS・追跡・形状近似・キーフレーム最適化）
    -> overlay（確認動画の描画・エンコード）
  -> result.sqlite / overlay videos
```

各パッケージはSQLiteとmanifestだけで接続します。別パッケージの内部モジュールを
直接importしないため、推論・後処理・overlayを独立して検証できます。

## Repository layout

```text
InstanceSegmentation/
  inference/
    <model>/
      artifacts/
        detector/       検出モデルの重み
        backbone/       バックボーンの重み
        classifier/     分類器の重み
        trt/
          <profile>/
            engines/    TensorRTエンジン
            plugins/    ビルド済みランタイムプラグイン

postprocess/             マスク後処理パイプラインとProduction実装

overlay/                 SQLiteと元動画から確認用overlayを生成

orchestration/           推論、後処理、overlayの一気通貫runner

gui/                     Windows/Electron GUI、設定入力、Liveプレビュー

deployment/              WSLイメージ、Windows EXE、配布フォルダの構築

deployment_tests/        配布物と新旧バージョン共存の受け入れテスト

data/                    動画、推論SQLite、後処理SQLite

output/                  ローカル実行結果（Git管理外）
```

## 変更内容から探す

| 変更したい内容 | 最初に見る場所 |
|---|---|
| 性器・顔モデルの推論 | `InstanceSegmentation/inference/<model>/` |
| NMS、穴埋め、島処理 | `postprocess/nms/` |
| トラッキング | `postprocess/tracking/` |
| ポリゴン近似・DP・pair-vote | `postprocess/production/polygon/` |
| Catmull--Rom曲線・DP | `postprocess/production/curve/` |
| 公開SQLite schema・書き込み | `postprocess/contracts/`, `postprocess/artifacts/` |
| 顔・目の最終マスク | `postprocess/face_privacy/` |
| 全工程の順序・再利用・進捗 | `orchestration/` |
| overlay描画・動画encode | `overlay/` |
| GUIの入力項目・実行状態 | `gui/src/`, `gui/electron/` |
| 配布物の作成・検証 | `deployment/`, `deployment_tests/` |

Productionへ影響しない調査コードは`postprocess/experiments/`、出力を変えない
監査ツールは`postprocess/diagnostics/`へ置きます。本番コードから
`experiments`をimportしてはいけません。

## Artifact policy

Gitではソースコード、設定、manifest、テスト、ドキュメントだけを管理します。
次の実行時成果物は`.gitignore`の対象です。

- モデル重みとcheckpoint
- ONNX、TensorRTエンジン、ビルド済みプラグイン
- 動画、SQLite、JSONL
- モデル別の`.runtime`環境
- `input`、`output`、ルート`data`配下の実行データ
- `InstanceSegmentation/tentative_folder`のローカル受け渡し一式

モデルの設定やTensorRT bundleのmanifestは、再現性のためGit管理に含めます。
重みとエンジンは上記の既定ディレクトリへ配置してください。

詳細な実行方法は
[`InstanceSegmentation/inference/README.md`](InstanceSegmentation/inference/README.md)
、[`postprocess/README.md`](postprocess/README.md)、
[`overlay/README.md`](overlay/README.md)、
[`orchestration/README.md`](orchestration/README.md)、
[`gui/README.md`](gui/README.md)を参照してください。

クリーンCloneへの外部資産配置、production runtime検証、Windows GUI buildは
[`deployment/README.md`](deployment/README.md)を正本とします。

## 共通の品質確認

ルートから全コンポーネントの静的コンパイル、単体テスト、GUI型検査を同じ入口で
実行できます。GPUやモデル資産を使う長時間検証は、この高速な品質確認とは分離
しています。

```bash
/path/to/production/bin/python scripts/check_repository.py
```

例えば後処理とrunnerだけなら、末尾へ `postprocess orchestration` を指定します。
`make check PYTHON=/path/to/python` も利用できますが、`make` を含まないProduction
WSLでも上記Pythonコマンドは動作します。
