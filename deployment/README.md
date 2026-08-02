# Deployment

このディレクトリは、Gitに含めないモデル資産と、Git Clone後の実行環境との境界を
固定します。現在完成させる対象はphase 2です。phase 3のWSL distribution移送は、
この手順を基準に別途固定します。

## 配置単位

- Git: ソース、設定、テスト、SQLite schema契約、TensorRT bundle manifest
- production asset pack: 重み、分類器、検証済みengine/plugin、Face V2 runtime source、
  後処理重み
- Windows: Node.jsと、WSL正本から生成するElectron exe
- WSL外部runtime: `/home/kenshin/.local/share/video-mask-runtime/envs/production`
- 入出力: `data/`と`output/`。どちらもGit管理外

TensorRT engineはGPU、TensorRT、CUDA、driverとの組合せに依存します。同じRTX 5090
であっても、現地で無条件に再生成したものを即本番登録しません。通常のデプロイでは
検証済みengine bundleをasset packで配置します。再生成スクリプトはモデルごとに
残していますが、生成後は速度・品質gateを通してからbundleを更新します。

## Phase 2: このPC上のクリーンClone

正本でasset packを一度作ります。`--link`は同一ディスク上での再現試験専用で、
Google Driveへ渡す実体コピーでは付けません。

```bash
python3 deployment/export_assets.py \
  /home/kenshin/inference_backend2_exports/production_assets_20260802 \
  --profile all
```

新しいCloneでは次だけを行います。

```bash
git clone git@github.com:Kenshy18/Inference_deployment.git inference_backend2
cd inference_backend2
python3 deployment/install_assets.py /path/to/production_assets_20260802
INFERENCE_RUNTIME_PYTHON=/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10 \
  ./deployment/setup_phase2.sh --profile all --full-hash
./gui/scripts/dev-windows.sh
```

正式exeは、clean commitから次で作成します。

```bash
./gui/scripts/build-windows.sh
```

`core`はV3、V3-lite、Face V2、後処理、高速overlayだけです。GUIに残るV1、V2、
Face V1まで全て実行可能にする配布物は`all`を使います。

## Engine再生成

正式入口は各READMEの次のスクリプトです。

- V3: `dinov3_codino/trt/build_fast_engines.py`
- V3-lite: `dinov3_codino_mh0/trt/build_fast_engines.py`
- Face V2: `face_dino_v2/trt/build_engines.py`
- V2: DINOv3 backbone engine保守スクリプト
- V1: `eva02_cascade/trt/build_engine.py`

build先は一時ディレクトリにし、既存production bundleを上書きしません。新bundleを
明示指定した推論、SQLite schema比較、マスク品質比較、長尺速度試験を通した後にだけ
production manifestとasset packを更新します。

## Phase 3の境界

WSL export/importでは、Gitだけでなくproduction Python、モデルruntime、engine、
native overlay runtimeを含むdistribution全体を移します。Windows側で別途合わせるのは
WSL2、NVIDIA driver、distribution登録名、Windows Node.js/exeです。phase 3では現在の
環境情報を固定し、import後preflightと短尺GUI E2Eを実行するスクリプトを追加します。
