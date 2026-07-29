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

## 性能設定

品質・SQLite schema・atomic公開を維持したまま使える主な設定です。

```json
{
  "inference": {
    "mode": "segmentation-face",
    "segmentation_model": "dinov3_codino_mh0",
    "face_model": "face_dino_v2",
    "parallel_models": true,
    "parallel_model_stagger_seconds": 0.0,
    "fast_sqlite": true
  },
  "postprocess": {
    "precompute_cuts_during_inference": true
  }
}
```

- `parallel_models`: segmentationと顔検出を隔離プロセスのまま同時実行。
  `mode=segmentation-face`、高速`dinov3_codino_mh0`、新顔検出
  `face_dino_v2`の3条件を満たす場合だけ`true`を選択できる。RTX 5090・
  3分の同条件比較では推論を74.69秒から60.55秒へ18.9%短縮した。
  巨大`dinov3_codino`、旧顔検出、片方だけの推論では設定エラーになる
- `parallel_model_stagger_seconds`: モデル起動間隔。`0.0`は完全同時。
  高速`dinov3_codino_mh0`とFace DINO v2のRTX 5090実測では`0.0`が最速
- `fast_sqlite`: SQLiteの異常終了耐性を速度優先に変更。最終公開はatomicのまま
- `precompute_cuts_during_inference`: CPUカット検出を推論と重ね、同じ
  `cuts.json`を後処理へ渡す。現在は`high_precision`方式に対応。3分の同条件比較
  では並列推論を0.88秒遅くした一方、約4.03秒のカット検出を全て隠し、全体を
  77.31秒から71.70秒へ短縮

開始ずらしの最適値はGPUと実行環境に依存します。
採用判断の同条件A/Bは
[`docs/PARALLEL_VALUE_BENCHMARK_20260728.md`](docs/PARALLEL_VALUE_BENCHMARK_20260728.md)
にあります。
3分の再現用設定と計測結果は
[`configs/profile_3min_optimized_20260728.json`](configs/profile_3min_optimized_20260728.json)と
[`docs/OPTIMIZATION_3MIN_20260728.md`](docs/OPTIMIZATION_3MIN_20260728.md)にあります。

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

### クラス別後処理

性器クラスごとに形状、キーフレーム密度、欠損補完上限を変える場合は、
postprocess policyを指定します。

```json
{
  "postprocess": {
    "enabled": true,
    "shape_mode": "polygon",
    "keyframe_interval": 3,
    "max_gap": 0,
    "class_postprocess_policy_json":
      "../../postprocess/configs/class_postprocess_policy.example.json",
    "device": "cuda:0"
  }
}
```

`shape_mode`、`keyframe_interval`、`max_gap`はpolicyに値がない場合の共通
fallbackです。policy内はクラス指定、`default`、このfallbackの順に解決します。
`max_gap=0`は観測のないフレームを追加せず、正数は両側に同一trackが存在する
欠損をそのフレーム数まで補完します。

オーケストレーターはpolicyに楕円クラスが1つでもあればpostprocessをGPU stage
として計画し、CLIへpolicyをそのまま渡します。設定例は
[`../postprocess/configs/class_postprocess_policy.example.json`](../postprocess/configs/class_postprocess_policy.example.json)
です。任意グラフの`postprocess.pipeline_config`とは併用できません。10分入力の
実測値は
[`docs/CLASSWISE_POSTPROCESS_BENCHMARK_20260728.md`](docs/CLASSWISE_POSTPROCESS_BENCHMARK_20260728.md)
にあります。

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
- postprocessは`shape_mode: ellipse`かつ`device: auto/cuda/cuda:N`の場合、
  K2楕円近似へGPUを公開します。`device: cpu`でCPU実行を強制できます。
- ポリゴン後処理は`postprocess.device`にかかわらずCPU処理です。
- `postprocess.face_mask_target: face/eyes`はFace DINO v2の楕円・keypointから
  CPUでprivacy maskを作り、性器の最終SQLiteへ統合します。
- overlayは`nvenc`または`fast`だけにGPUを公開します。
- `cpu` subprocessには`CUDA_VISIBLE_DEVICES=""`と
  `NVIDIA_VISIBLE_DEVICES=none`を設定します。

楕円近似の標準設定は、下流で使わないsoft maskを生成しない
`states_only`です。調整が必要な場合は`postprocess.extra_args`へ
`--k2-batch-size 128`、`--k2-prep-workers 4`、
`--k2-cudnn-benchmark on`などを指定できます。CPU版との数値的一致を
優先する本番設定は`--k2-tf32 off`、最大速度優先は`--k2-tf32 on`です。

## 成果物

```text
output_root/
├── run_manifest.json
├── resolved_config.json
├── logs/
├── 00_preflight/
│   └── cuts.json
├── 01_inference/
│   └── inference.sqlite             # 内部中間成果物
├── 02_postprocess/
│   ├── pipeline_manifest.json
│   ├── ...                          # 内部中間成果物
│   └── NN_integrated_result_sqlite/
│       └── result.sqlite            # 後処理ありの公開SQLite
├── 02_result/
│   └── result.sqlite                # 後処理なしの場合の公開SQLite
└── 03_overlay/
    ├── raw.mp4
    ├── raw.json
    ├── tracked.mp4
    ├── final.mp4
    └── faces.mp4
```

後処理の有無、性器検出の有無、旧/新顔検出の組み合わせにかかわらず、公開
SQLiteは`result_sqlite`の1つです。推論の全生出力を保持したまま、
`tracking_assignments`、最終編集キーフレーム、`tracks`、`cuts`、監査・
provenanceテーブルを常に同じ列契約で持ちます。実行していない機能は
テーブル欠落ではなく空テーブルで表現し、`result_capabilities`で利用可否と
件数を確認できます。
stage番号から場所を推測せず、
`run_manifest.json`の`artifacts.result_sqlite`を使用してください。
raw/tracked/final/facesのoverlayもこの1ファイルだけを読みます。

安定契約は`result_schema_info`の
`schema_version=3`、`compatibility_profile=keyframe-primary-v3`、
`contract_revision=4`で識別します。
`result_capabilities`には`instance_segmentation`、`face_detection`、
`rich_face_geometry`、`tracking_assignments`、`final_annotations`、
`cut_detection`、
`classwise_postprocess`、`face_privacy_masks`などが入り、各行は
`available`、`row_count`、`source_table`、`details_json`を持ちます。

ソフトウェア編集用には`mask_track_segments`、`mask_keyframes`、
`keyframe_components`を起点に、`keyframe_ellipses`、
`keyframe_rectangles`、`keyframe_polygon_rings/points`を読みます。後処理楕円は
96点polygonではなく中心・半径・radian角度、ポリゴンは選択された実
keyframe頂点として保存されます。`editable_keyframe_components`と
`editable_polygon_vertices`は安定reader viewです。
`annotation_state`はこのキーフレーム層が唯一の編集正本であり、毎フレーム
形状を永続cacheとして持たないことを明示します。overlayはV3から必要範囲の
一時cacheを生成して既存の高速rendererへ渡します。

`result_components`は未実行、非対応、実行済み0件を区別します。
`processing_runs`にはpostprocess CLIとオーケストレーターの解決済み設定、
`processing_stage_runs`にはstage実装、options、device、所要時間を保存します。

顔だけの`inference.mode=face`では、未指定時のpostprocessは自動的に無効となり、
overlayは`faces=true`、`raw/tracked/final=false`になります。性器推論で
postprocessを明示的に無効にした場合も、rawだけが既定で有効です。

新顔検出だけからソフトウェア用の顔／目マスクを作る場合は、性器pipelineを
起動せず、result packaging内で直接生成できます。

```json
{
  "inference": {
    "mode": "face",
    "face_model": "face_dino_v2"
  },
  "postprocess": {
    "face_mask_target": "eyes",
    "eye_mask_shape": "ellipse",
    "minimum_eye_confidence": 0.35
  }
}
```

この場合も顔／目マスクは通常の`mask_keyframes`、`tracks`、
`mask_provenance`に入り、`final_annotations`と`face_privacy_masks`
capabilityが有効になります。

`01_inference`と各postprocess stageのSQLiteは、失敗時の安全性、stage契約、
resumeのための内部中間成果物です。`run_manifest.json`の公開artifactには出さず、
下流ソフトウェアへ渡しません。各overlay JSONには選択した実行方式、overlay
種別、入力role、encode設定と処理結果が記録されます。

`postprocess.export_legacy_sqlite: true`では、現行`result_sqlite`とは別に旧
`Dinov3_postprocess`互換の`legacy_final_sqlite`もrun manifestへ公開します。
互換版は旧契約の`masks`、`tracks`、`cuts`のみを持ち、元マスクおよび詳細な
カット検出メタデータは現行SQLiteにだけ保持されます。
