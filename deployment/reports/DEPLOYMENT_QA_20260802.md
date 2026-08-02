# Deployment QA — 2026-08-02

## 判定

Windows portable GUI＋移送用WSL VHDX＋ワンクリックデプロイヤーの配布経路は合格。
初回WSL導入、GPU driver交換、組織ポリシー、コード署名は配布物外の作業として残る。

## 最終配布物

- release: `D:\MaskPipelineDeployment\LATEST`
- release id: `mask-pipeline-20260802-104355`
- backend/GUI commit: `6fde9e834b2f6e85a13ee8b6ea0ab4fbd8ae7972`
- deployer commit: `2ea885212450ff757a4155ef179a7c97c99fccee`
- asset commit: `6f6823927eefc178a55a53c2615c011fc1ce0076`
- backend VHDX: 36,519,804,928 bytes
- payload＋deployer SHA256: 6/6 files passed
- compatibility: Windows 10.0.26200.8655, WSL 2.7.10.0, Ubuntu 24.04,
  RTX 5090, NVIDIA driver 596.21, compute capability 12.0

配布VHDXには入力動画、過去出力、`.codex`、SSH鍵、shell履歴を含めない。
V1/V2/V3/V3-lite、Face V1/V2を含む21 asset groups、production Python、
TensorRT/CUDA、native overlay、高速FFmpeg/FFprobeを含む。

## クリーン導入

最終配布物を`MaskPipelineQAFinal`へ、Windows PowerShellのデプロイヤーだけで導入した。

- 8/8 stages passed
- elapsed: 64 seconds（11:00:44–11:01:48 JST）
- install report: `D:\MaskPipelineDeployment\qa-final\deployment-report.json`
- backend full-hash preflight: passed
- GUI E2E: passed, exit 0, renderer error 0
- 120/120 frames; inference, face, postprocess, overlay, GUI sync completed
- LIVE preview: 24 updates
- UI heartbeat max: 246.2 ms; over 500 ms: 0
- WSL staging, pipeline process, SQLite WAL/SHM, partial artifact after completion: 0

既存の`Ubuntu-24.04`は停止せず、配布用・QA用distributionだけを停止／起動した。
QA終了後、Windows GUIのbackend設定は`Ubuntu-24.04`へ復元済み。

## 実データGUI試験

Windows D:上に置いた実データ20秒fixtureの先頭300フレームを、配布GUIから処理した。
CLIでパイプラインを直接起動していない。

- elapsed: 17.525 s（GUI harness）
- V3-lite: 69.55 FPS
- Face V2: 62.00 FPS
- fast overlay: 240.44 FPS
- detections: 1,625
- segmentations: 367
- rich face observations: 629
- face keypoints: 3,145
- face tracking assignments: 615
- face interpolation: 7
- final mask keyframes: 622
- renderer error: 0
- LIVE preview: 87 updates
- UI heartbeat max: 293.2 ms; over 500 ms: 0

開発distributionのUNCパスをQA distributionへ直接渡す試行は、意図した隔離により
dry-runで「入力なし」として拒否された。本番想定どおりWindowsドライブ上へ入力を置くと合格した。

## SQLite固定契約

配布GUI成果物2件、実データ成果物1件、既存software handoff 4パターンを
`validate_result_sqlite`で再検証した。

- schema name: `video-mask-integrated-result`
- schema version: 3
- contract revision: 5
- compatibility profile: `keyframe-primary-v3`
- schema signature: `a7fbe8262ec2115af32cf49129534a58ff5190f4dd537c5811f4a1fefdecfa11`
- validated variants: 顔なし、目元長方形、目元楕円、顔全体、空検出、非空検出
- all signatures identical

この作業ではSQLite schema実装を変更していない。モデルや顔マスク設定による欠損項目は
テーブル削除ではなく、固定テーブルとcapability/component rowsで表現される。

## 異常系・回復性

`Test-DeployerNegative.ps1`で次を実行した。

1. payload SHA256改ざん: rejected
2. GPU不一致: rejected
3. 既存WSL distribution名: rejected without overwrite
4. 壊れたVHDX: rejected and rolled back

全ケースでGUI設定は不変、新規distributionなし、`ext4.vhdx`/`.partial`残留なし。
初回試験で壊れたVHDX後のrollbackがエラー表示に阻害される問題を検出し、修正後に
同じ4ケースを再実行して合格した。

その後、Windows PowerShell 5.1で引数なし起動時の`PSScriptRoot`がparameter default
評価中に空になる問題を実地起動で検出した。payload解決をscript bodyへ移し、配布物を更新。
隣接payloadを引数なしで解決するregression testも追加した。

UAC昇格したprocessが`C:\ProgramData`へ作成したVHDXを、ユーザー単位のWSL serviceが
attachできず`E_ACCESSDENIED`になる問題も実地起動で検出した。WSL distribution、GUI設定、
shortcutを同一Windowsユーザーへ揃え、既定導入先を`%LOCALAPPDATA%\MaskPipeline`へ変更した。
更新版EXEそのものから`MaskPipelinePerUserQA`をC:へ導入し、8/8 stages、full-hash preflight、
Windows GUI 120-frame E2E、renderer error 0で合格した。

GUI処理中キャンセルも合格。job statusは`cancelled`、stageは`inference`、
終了後のWSL pipeline processとstaging directoryはいずれも0件だった。
QA distribution停止後の再起動E2Eも合格した。

## 成果物品質

synthetic outputは入力と同じH.264、1280x720、30 FPSを維持した。

- 120-frame run: exactly 120 frames, 4.000 s
- 60-frame restart run: exactly 60 frames, 2.000 s
- missing/duplicate frame count: 0
- output audio: intentionally omitted by current overlay QA preset

Python contract testsは37 passed＋12 subtests、native overlay testsは10 passed。

## 配布前に残る事項

- `MaskPipelineDeployer.exe`とGUI portable exeはAuthenticode未署名。配布先で
  Unknown Publisher/SmartScreen警告が出る可能性がある。正式外部配布前はコード署名を推奨。
- WSL未導入PC、再起動を伴うdriver変更、Secure Boot、組織ポリシーは手動runbook対象。
- 移送用SSDのファイルシステムは、36.5GBのVHDXを扱えるNTFSまたはexFATを使う。
