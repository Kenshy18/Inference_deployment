# Face DINO v2

顔推論の第2モデルです。DINOv3 ViT-S+/16、SFP、compact Co-DINOのHead検出と、
Head ROI上のFace有無、楕円、5点の意味クラス付きキーポイント、
visible/occluded分類を実行します。

## 高速backend

固定入力はB8、`[8, 3, 736, 1280]`です。16:9動画は1280×720へ縮小し、上下8 pxを
paddingします。

- ViT-S+とSFP: mixed-FP16/FP32 TensorRT
- query encoder: SM120 MSDA pluginを用いたmixed-FP16/FP32 TensorRT
- decoder: FP32 TensorRT
- attribute branch: FP16 TensorRT
- 前処理: BGR、resize、letterbox、ImageNet正規化を融合したSM120 CUDA kernel
- bbox復号: batch-wide GPU top-k/filter
- CPU prefetch: 1 batchに制限

engine bundleは全ファイルのsizeとSHA-256をmanifestで検証します。checkpointの
SHA-256もbundleと一致しなければ起動しません。

## Checkpointからengineを生成

production環境のPythonで、4工程を直列実行します。`MAX_JOBS=1`を強制するため、
CPUメモリを並列ビルドで圧迫しません。

```bash
cd /home/kenshin/inference_backend2/InstanceSegmentation/inference
PY=/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python

$PY -m face_dino_v2.trt.build_engines \
  --source-root /home/kenshin/face_detection \
  --checkpoint \
    /home/kenshin/face_detection/runs/face_dino_overnight_best_20260727/model_residual_v2.pth
```

生成先:

```text
face_dino_v2/
├── artifacts/detector/model_residual_v2.pth
├── artifacts/trt/fast-sm120-fixed-b8-v1/
│   ├── manifest.json
│   ├── engines/
│   │   ├── backbone_neck.engine
│   │   ├── query_encoder.engine
│   │   ├── decoder.engine
│   │   └── attribute.engine
│   └── plugins/
│       ├── codino_msda_direct_mh0_sm120.so
│       └── face_preprocess_fused_sm120.so
└── .runtime/src/face_detection/
```

ビルド時に必要なPython source、Co-DINO/DINOv3 source、ViT-S+公式重み、
PyTorch checkpointもスナップショットします。生成後の推論は元の
`/home/kenshin/face_detection`に依存しません。

## 検証

```bash
$PY face_dino_v2/setup_environment.py
```

## 単独動画推論

```bash
$PY face_dino_v2/infer.py \
  --input input.mp4 \
  --output face_dino_v2.sqlite \
  --batch-size 8 \
  --classes Head Face \
  --overwrite
```

統一SQLite契約には、Head検出boxと、Face楕円のaxis-aligned envelopeを保存します。
HeadとFaceは独立行であり、後頭部ではHeadだけが保存されます。

## 統一CLI

```bash
$PY run_inference.py \
  --mode face \
  --face-model face_dino_v2 \
  --input input.mp4 \
  --output result.sqlite \
  --runtime-python $PY \
  --overwrite
```

## Rich runtime API

`FaceDinoRuntime.predict()`はSQLiteへ縮約する前の次のGPU tensorを返します。

- `boxes`, `scores`
- `face_scores`, `face_present`
- `ellipses`: `cx, cy, major_radius, minor_radius, theta`
- `keypoints`
- `point_classes`: eye/nose/mouth
- `keypoint_states`: visible/occluded
- 各クラス・状態確率

共有SQLite schemaには現時点で楕円・点専用tableがないため、Adapterだけが
Head/Face boxへ縮約します。rich出力は失われず、モデルruntime APIから利用できます。

## 検証済み性能

RTX 5090、1920×1080動画、800 frames、B8、SQLite保存込み:

- compute: 133.018 images/s
- wall: 127.009 FPS
- SQLite integrity: `ok`
- 座標範囲違反: 0

Test 528画像ではbbox AP 0.749593、Face macro-F1 0.833104、
ellipse IoU 0.742672、point F1@NME .10 0.911252、
occlusion macro-F1 0.821496です。詳細は`BUILD_REPORT.md`を参照してください。
