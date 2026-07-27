# Inference deployment

動画に対するインスタンスセグメンテーション推論と、そのマスク後処理を行う
ワークスペースです。

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

postprocess/
  models/               後処理モデルの設定と実装

overlay/                 SQLiteと元動画から確認用overlayを生成

orchestration/           推論、後処理、overlayの一気通貫runner

data/                    動画、推論SQLite、後処理SQLite
```

## Artifact policy

Gitではソースコード、設定、manifest、テスト、ドキュメントだけを管理します。
次の実行時成果物は`.gitignore`の対象です。

- モデル重みとcheckpoint
- ONNX、TensorRTエンジン、ビルド済みプラグイン
- 動画、SQLite、JSONL
- モデル別の`.runtime`環境
- `input`、`output`、ルート`data`配下の実行データ

モデルの設定やTensorRT bundleのmanifestは、再現性のためGit管理に含めます。
重みとエンジンは上記の既定ディレクトリへ配置してください。

詳細な実行方法は
[`InstanceSegmentation/inference/README.md`](InstanceSegmentation/inference/README.md)
、[`postprocess/README.md`](postprocess/README.md)、
[`overlay/README.md`](overlay/README.md)、
[`orchestration/README.md`](orchestration/README.md)を参照してください。
