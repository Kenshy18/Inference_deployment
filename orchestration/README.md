# Repository orchestration

`InstanceSegmentation`、`postprocess`、`overlay`を、それぞれの公開CLIと
SQLite/manifest契約だけで接続する上位workflowです。モデル実装や後処理実装を
直接importしません。

## 実行

```bash
cd /home/kenshin/inference_backend2

/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  -m orchestration \
  --config orchestration/configs/production.json
```

既存の完了stageを再利用する場合:

```bash
python3 -m orchestration --config config.json --resume
```

設定と既存入力だけを検証する場合:

```bash
python3 -m orchestration --config config.json --dry-run
```

## 既存inference SQLiteから開始

GPU推論を行わず、既存のunified inference SQLiteから開始できます。

```json
{
  "inference": {
    "enabled": false,
    "input_sqlite": "/path/to/inference.sqlite",
    "mode": "segmentation"
  }
}
```

`mode`はSQLiteに期待するroleを表します。顔overlayも作る場合は
`segmentation-face`を指定し、SQLiteに`instance_segmentation`と
`face_detection`の両roleが存在する必要があります。

## Overlay execution modes

`overlay.execution_mode`で作成方式を3種類から選択します。

| execution_mode | 描画・encode経路 | 用途 |
|---|---|---|
| `cpu` | OpenCV CPU描画＋libx264 | GPU非依存の通常モード |
| `nvenc` | OpenCV CPU描画＋NVENC | 通常描画を保ったGPU encode |
| `fast_parallel` | NVDEC＋CUDA描画＋NVENC分割並列 | 採用した高速モード |

```json
{
  "overlay": {
    "execution_mode": "cpu"
  }
}
```

```json
{
  "overlay": {
    "execution_mode": "nvenc",
    "nvenc_preset": "p5",
    "nvenc_gpu": 0
  }
}
```

`execution_mode`を指定した場合、`backend`と`codec`は自動決定されます。移行用に
従来の`backend`と`codec`だけの設定も読み込めますが、新規設定では
`execution_mode`を使用してください。

## GPU policy

- inference: `inference.device`をそのままモデルCLIへ渡します。
- postprocess: repository orchestrationではCPUだけを許可します。
- overlay: `execution_mode`が`nvenc`または`fast_parallel`のときだけGPUを公開し、
  `cpu`はCPU stageとして実行します。

CPU stageのsubprocessには`CUDA_VISIBLE_DEVICES=""`と
`NVIDIA_VISIBLE_DEVICES=none`を設定します。

標準のPython/OpenCV overlayでNVENCを使う場合は次のように選択できます。

```json
{
  "overlay": {
    "execution_mode": "nvenc",
    "nvenc_preset": "p5",
    "nvenc_gpu": 0,
    "extra_args": ["--nvenc-cq", "18"]
  }
}
```

## 低レイヤー高速overlay

`overlay.execution_mode`を`fast_parallel`にすると、C++/libavで直接デコードし、
NVDEC → CUDA上の描画 → NVENCをフレーム単位のCPU往復なしで実行します。
複数区間を並列処理し、最後に再エンコードなしで結合します。通常モードは
OpenCV経路のままで、明示的に切り替えない限り高速版へ変わりません。

```json
{
  "overlay": {
    "execution_mode": "fast_parallel",
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

`workers`は全区間数、`cpu_workers`はそのうちlibx264へ割り当てる数です。
たとえばCPU 3＋NVENC 3は`workers: 6, cpu_workers: 3`、純NVENC 6並列は
`workers: 6, cpu_workers: 0`です。現実測では1080pの最大スループットは
純NVENC 6並列だったため、上の例を推奨値にしています。入力全体を処理する
場合は`end_frame`を省略でき、MP4に記録されたフレーム数を自動取得します。

このバックエンドには事前ビルドが必要です。

```bash
cd overlay/experimental
./build.sh
```

高速版の採用検証結果と既知の表示差は
[`../overlay/experimental/ADOPTION_VALIDATION.md`](../overlay/experimental/ADOPTION_VALIDATION.md)
を参照してください。生成物ごとのフレーム数、PTS/DTS、分割境界、音声連続性、
全stream decodeを再確認する場合は、同文書の`validate_fast_output.py`を使用します。

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
    ├── tracked.mp4
    ├── final.mp4
    └── faces.mp4
```

最終SQLiteの場所はstage番号から推測せず、postprocessの
`pipeline_manifest.json`にある`tracked_sqlite`と`predictions_sqlite`を
使用します。
