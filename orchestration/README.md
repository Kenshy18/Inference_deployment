# Repository orchestration

`InstanceSegmentation`、`postprocess`、`overlay`を、公開CLIとSQLite/manifest契約で
一気通貫に接続します。各リポジトリの内部実装はimportしません。

## 実行

```bash
cd /home/kenshin/inference_backend2

/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python \
  -m orchestration \
  --config orchestration/configs/production.json
```

既存stageを再利用する場合:

```bash
python3 -m orchestration --config config.json --resume
```

設定と再利用入力だけを検証する場合:

```bash
python3 -m orchestration --config config.json --dry-run
```

設定の土台は
[`configs/production.example.json`](configs/production.example.json)です。既存SQLiteから
高速4種類を作る例は
[`configs/overlay_fast.example.json`](configs/overlay_fast.example.json)にあります。

## 既存SQLiteから開始

推論を行わず、既存のunified inference SQLiteから開始できます。

```json
{
  "inference": {
    "enabled": false,
    "input_sqlite": "/path/to/inference.sqlite",
    "mode": "segmentation-face"
  }
}
```

`mode`はSQLiteに期待するroleです。`segmentation-face`では
`instance_segmentation`と`face_detection`の両roleが必要です。
`face_model: face_dino_v2`ではschema v3、Head/Face対応、楕円、顔確率マスク、
5点キーポイントと確率テーブルまで検証します。従来の
`face_model: rtdetr_head_face`および既存schema v2も利用できます。

postprocessも再利用する場合は、tracked/finalを明示します。

```json
{
  "postprocess": {
    "enabled": false,
    "tracked_sqlite": "/path/to/tracked.sqlite",
    "final_sqlite": "/path/to/predictions.sqlite"
  }
}
```

Postprocess実行時に旧`Dinov3_postprocess`互換SQLiteも作る場合は、
`postprocess.export_legacy_sqlite: true`を設定します。現行成果物はそのまま
生成され、互換版が追加されます。

## Overlay設定

実行方式は3種類です。

| `execution_mode` | 描画・encode経路 | 用途 |
|---|---|---|
| `cpu` | OpenCV CPU描画＋libx264 | GPU非依存の通常モード |
| `nvenc` | OpenCV CPU描画＋NVENC | 通常表示を保ったGPU encode |
| `fast` | C++/libav＋NVDEC＋CUDA＋分割NVENC | 対応入力の最大速度 |

`backend`と`codec`は`execution_mode`から自動決定します。旧設定の
`backend: experimental_cpp`、`execution_mode: fast_parallel`、codecだけの指定は
読み込み互換として残していますが、新規設定では使用しません。

作成するオーバーレイは個別に選択できます。

```json
{
  "overlay": {
    "enabled": true,
    "execution_mode": "cpu",
    "raw": true,
    "tracked": true,
    "final": true,
    "faces": true,
    "final_include_faces": true,
    "mask_alpha": 0.32,
    "outline_thickness": 2,
    "box_thickness": 2,
    "show_labels": true
  }
}
```

- `raw`: AI生出力instance mask
- `tracked`: NMS、カット分割、tracking、短命track削除後
- `final`: 最終後処理後
- `faces`: 顔・頭部。Face DINO v2の通常rendererではbox、楕円、mask、keypoint
- `final_include_faces`: `final`へ上記の顔情報も追加

通常NVENC:

```json
{
  "overlay": {
    "execution_mode": "nvenc",
    "nvenc_cq": 18,
    "nvenc_preset": "p5",
    "nvenc_gpu": 0
  }
}
```

通常CPU:

```json
{
  "overlay": {
    "execution_mode": "cpu",
    "h264_crf": 18,
    "h264_preset": "veryfast"
  }
}
```

`h264_crf`と`nvenc_cq`は0～51で、小さいほど高品質です。
`target_bitrate_mbps`を指定した場合はCRF/CQよりbitrate制約が優先されます。
新規の3実行方式はすべてMP4コンテナ、H.264、yuv420pで出力します。

高速:

```json
{
  "overlay": {
    "execution_mode": "fast",
    "workers": 6,
    "cpu_workers": 0,
    "target_bitrate_mbps": 8.0,
    "nvenc_preset": "p1",
    "nvenc_gpu": 0,
    "copy_audio": true,
    "faststart": true
  }
}
```

`workers`は区間数、`cpu_workers`はそのうちlibx264へ割り当てる数です。
既定の`workers: 6, cpu_workers: 0`は完全GPU、CPU 3＋NVENC 3は
`workers: 6, cpu_workers: 3`です。今回の1080p実測では完全GPU 6 workerが
最速でした。入力全体を処理するときは`end_frame`を省略できます。

高速版は事前にbuildします。

```bash
overlay/native/build.sh
```

品質・フレーム・時刻・分割境界の検証結果は
[`../overlay/docs/ADOPTION_VALIDATION.md`](../overlay/docs/ADOPTION_VALIDATION.md)
を参照してください。

## GPU policy

- inferenceは`inference.device`をモデルCLIへ渡します。
- postprocessはrepository orchestrationではCPUだけを許可します。
- overlayは`nvenc`または`fast`だけにGPUを公開します。
- `cpu` subprocessには`CUDA_VISIBLE_DEVICES=""`と
  `NVIDIA_VISIBLE_DEVICES=none`を設定します。

## 成果物

```text
output_root/
├── run_manifest.json
├── resolved_config.json
├── logs/
├── 01_inference/
│   └── inference.sqlite
├── 02_postprocess/
│   ├── pipeline_manifest.json
│   └── ...
└── 03_overlay/
    ├── raw.mp4
    ├── raw.json
    ├── tracked.mp4
    ├── final.mp4
    └── faces.mp4
```

最終SQLiteの場所はstage番号から推測せず、postprocessの
`pipeline_manifest.json`にある`tracked_sqlite`と`predictions_sqlite`を
使用します。各overlay JSONには選択した実行方式、overlay種別、入力role、
encode設定と処理結果が記録されます。

`postprocess.export_legacy_sqlite: true`では、現行`final_sqlite`とは別に旧
`Dinov3_postprocess`互換の`legacy_final_sqlite`もrun manifestへ公開します。
互換版は旧契約の`masks`、`tracks`、`cuts`のみを持ち、元マスクおよび詳細な
カット検出メタデータは現行SQLiteにだけ保持されます。
