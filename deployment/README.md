# Deployment

このディレクトリは、Gitに含めないモデル資産と実行環境との境界を固定します。
Phase 2のclean clone構築と、Phase 3のWSL distribution＋Windows GUI配布の両方を
再現可能な入口として管理します。

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

`install_assets.py`はasset packを作成したGit commitとCloneの`HEAD`が異なる場合に
停止します。重み・engine・ソースの世代を、似たファイル名だけで混在させません。

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

## Phase 3: 配布用WSLとWindowsデプロイヤー

正式な候補リリースはGit正本のクリーンなコミットから、次の1コマンドで作成します。

```bash
./deployment/build_release.sh
```

この入口は、アセットとproduction runtimeのステージング、隔離されたUbuntu 24.04
build distributionの作成、portable Windows GUIのビルド、WSL archiveの構築・検証、
`MaskPipelineDeployer.exe`の作成、成果物hash検証までを順番に実行します。build
distributionは一意な一時名を使い、成功・失敗にかかわらず既存distributionを変更しません。
成功した成果物は`D:\MaskPipelineDeployment\release\mask-pipeline-*`へ候補として保存され、
実デプロイ試験に合格するまでは`LATEST`へ昇格させません。

配布用distributionと配布フォルダは開発対象ではありません。修正は必ずGit正本で行い、
配布物はこの入口から再生成します。調査目的で配布済みdistributionへ一時変更を入れても、
それを次のリリースへ手動コピーしません。

配布イメージは、空のUbuntu 24.04へ次だけを配置して作ります。

- 固定Git commitのclean clone
- commit検証済みproduction asset pack
- production Python環境
- production環境の`.pth`が固定参照する21MBのruntime source tree
- 構築・検証済みnative overlay runtime
- asset packに同梱したSHA-256固定のFFmpeg、FFprobe、Zig、SQLite build archive
- 非センシティブな8秒のsynthetic E2E fixture

overlay runtimeの構築は`latest` URLの可用性に依存しません。asset packに含まれる
固定archiveを検証して展開するため、GitHubのautobuildが削除・更新されても、同じcommit
から同じ配布イメージを再構築できます。

`.codex`、SSH鍵、shell履歴、入力動画、過去出力、開発用cacheは含めません。
`phase3/prepare_image.sh`は配布用distribution内でrootとして実行し、assetのfull hash、
全モデルruntime、GPU、native overlay、単体テストを検証します。現在作業中のdistributionを
停止せず、配布用distributionだけを停止して標準WSL tar archiveへexportします。
`phase3/wsl.conf`はinteropを有効にしますがsystemdは有効にしません。複数distributionが
同時起動するPCでsystemd-binfmtが共有`WSLInterop`登録を外す事象を避けるためです。

Windows配布物は`windows/Build-Deployer.ps1`で作成します。成果物は次です。

- `MaskPipelineDeployer.exe`: ユーザー単位の導入を開始する入口
- `Deploy-MaskPipeline.ps1`: hash検証、WSL import、GUI配置、rollback
- `payload/backend.tar`: 検証済みLinux backendの標準WSL archive
- `payload/Mask Pipeline Studio.exe`: Node.js不要のportable GUI
- `payload/deployment-smoke.mp4`: 非センシティブな短尺fixture
- `payload/deployment-manifest.json`: commit、asset世代、GPU/driver、全SHA256

デプロイヤーは同名の既存distributionを上書きしません。標準`wsl --import`で専用の
ext4.vhdxを作成し、full-hash backend preflightとWindows GUI経由の120-frame E2Eを
通過した後にだけショートカットと完了reportを作ります。失敗時は新規distribution、GUI
設定、作成途中のbackendをrollbackします。Windows Node.jsは配布先には不要です。
WSL distribution、GUI設定、ショートカットはWindowsユーザー単位です。通常入口のexeは
UAC昇格せず、`%LOCALAPPDATA%\MaskPipeline`へ同一ユーザーとして導入します。
`-AllowNonAdministrator`は旧QAコマンドとの互換性のため受理しますが、現在は不要です。

初回のWSL導入、再起動を伴うGPU driver交換、Secure Boot/組織ポリシーは自動化の境界外です。
デプロイヤーは検証済みRTX 5090/driverとの不一致を、環境を変更せず明示的に停止します。

配布用WSL archive、portable GUI、非センシティブfixtureを用意した後の作成例です。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows\Build-Deployer.ps1 `
  -BackendArchive D:\release-input\backend.tar `
  -GuiPortable D:\release-input\MaskPipelineStudio.exe `
  -Fixture D:\release-input\deployment-smoke.mp4 `
  -OutputRoot D:\MaskPipelineDeployment\release `
  -ReleaseCommit <backend-commit> `
  -GuiCommit <gui-commit> `
  -DeployerCommit <deployer-commit> `
  -AssetCommit <asset-commit>
```

通常の現地導入は生成された`MaskPipelineDeployer.exe`をダブルクリックします。
既に同名distributionがある場合は上書きせず停止します。配布前の異常系試験は、隔離された
書込み可能ディレクトリに対して次で実行できます。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\deployment\windows\Test-DeployerNegative.ps1 `
  -ReleaseRoot D:\MaskPipelineDeployment\release\mask-pipeline-YYYYMMDD-HHMMSS `
  -ExistingDistribution MaskPipelineQA
```

この試験は、hash改ざん、GPU不一致、既存distribution、壊れたarchiveをすべて拒否し、
新規distribution・部分backend・既存GUI設定を残さないことを検証します。
