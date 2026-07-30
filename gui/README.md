# Mask Pipeline Studio

`inference_backend2`の推論、後処理、確認用overlayを操作するデスクトップGUIです。
Electron、React、TypeScriptで実装し、既存のPythonコードを直接importせず、
公開されている`python -m orchestration` CLIを子プロセスとして実行します。

## 画面構成

DaVinci Resolve / Premiere Proに倣った固定4ペインのNLE型レイアウトです。
角丸パネルをキャンバス上に浮かべたモダンなダークテーマで、30pxの操作行、
トグルスイッチ、ピル型セグメントコントロールで構成しています。

```text
┌──────────────────────────────────────────────────────────┐
│ topbar   入力/出力 + Dry Run / 実行 / 停止 + 実行環境      │
├────────────┬─────────────────────────────┬───────────────┤
│ SOURCE     │ MONITOR                     │ INSPECTOR     │
│ 入出力パス │  pipeline flow              │ Inference     │
│ 既存SQLite │  throughput scope           │ Postprocess   │
│ 成果物     │  進捗 / 経過 / stage timeline│ Overlay      │
│            │  metrics 8列                │ Runtime       │
├────────────┴─────────────────────────────┴───────────────┤
│ CONSOLE   orchestration の stdout/stderr、tag色分け       │
├──────────────────────────────────────────────────────────┤
│ status    状態 / stage / 経過 / exit / backend / python   │
└──────────────────────────────────────────────────────────┘
```

- 3本のペイン境界はドラッグで変更でき、幅は`localStorage`に保存します。
- Inspectorの各セクションは折りたたみ状態を保存します。
- Inspectorは日常操作向けの「簡単」と、公開設定を網羅する「詳細」を切り替えられます。
- Runtime設定は変更の500ms後に自動保存し、実行時にも保存します。

全設定とGUI上の配置、意図的に内部管理するCLI引数は
[ARGUMENT_COVERAGE.md](ARGUMENT_COVERAGE.md)に一覧化しています。

### ショートカット

| キー | 動作 |
| --- | --- |
| `Ctrl` + `Enter` | 実行 |
| `Ctrl` + `D` | Dry run |
| `Esc` | 実行中のジョブを停止 |

## セグメンテーションモデル

表示名と`InstanceSegmentation/inference/registry.py`の対応です。バックエンドは
各モデルが登録しているものだけを表示し、既定は必ず高速側です。

| 表示名 | model_id | 高速・推奨 | 互換 |
| --- | --- | --- | --- |
| EVA-02 + Cascade | `eva02_cascade` | `tensorrt-backbone` | `pytorch` |
| DINOv3 + Cascade | `dinov3_cascade` | `tensorrt-backbone` | なし |
| Co-DINO（巨大） | `dinov3_codino` | `tensorrt-fast` | `pytorch` |
| Co-DINO（高速） | `dinov3_codino_mh0` | `tensorrt-fast` | `pytorch` |

顔モデルは旧`rtdetr_head_face`と、頭部box・顔楕円/マスク・キーポイントを持つ
`face_dino_v2`を選択できます。推論デバイスを含む性能設定は「詳細」にあります。
`registry.py`が変わった場合は`src/lib/models.ts`と`src/lib/models.test.ts`を更新します。

## 現在の機能

- 入力キュー: ドラッグ&ドロップまたは追加ボタンで複数動画を登録し、順次処理
  - サムネイル・タイトル・長さ・状態（未処理/処理中/処理済み/失敗）を表示
  - 右クリックで削除、処理中の停止して削除、失敗/処理済みの再処理、出力フォルダを開く
  - 処理済みアイテムは実行時の推論設定サマリを表示
- 出力リポジトリ: 全ジョブ共通の出力先。各動画は「リポジトリ/動画名」へ出力し、
  処理中はリポジトリを変更不可
- 新規推論または既存unified inference SQLiteの再利用
- 推論モード、全モデル/engine、顔モデル、device、warmup、限定並行推論
- クラス別polygon/ellipse・keyframe・補完のGUI編集、K2 GPU設定、カット検出
- 顔privacy maskと軽量tracking、短命track除去、補完
- 6表示preset、CPU/NVENC/分割高速overlay、品質・worker・描画設定
- orchestration設定のdry-run
- ジョブ実行、キャンセル、stage timeline、throughput scope、リアルタイムログ
- `run_manifest.json`から成果物を読み、出力フォルダを開く
- Linux native PythonとWindows WSL2の切り替え

## 開発

WSL内で実行します。

```bash
cd /home/kenshin/inference_backend2/gui
npm install
npm run dev
```

rendererだけをブラウザで確認する場合:

```bash
npm run dev:renderer
```

ブラウザ表示ではファイル選択やPython実行はモックになります。
`?mock=running` / `?mock=done` / `?mock=failed` を付けると、実行中・完了・失敗の
表示をGPUなしで確認できます。`running`は擬似的な進捗を流し続けます。

WSLgでElectronを直接起動する場合、Chromiumが要求する共有ライブラリを
`.runtime-libs/`から解決します。

```bash
LD_LIBRARY_PATH=$PWD/.runtime-libs/usr/lib/x86_64-linux-gnu npx electron .
```

## 検証

```bash
npm run typecheck
npm run test
npm run build
```

Linux用の展開済みアプリを作る場合:

```bash
npm run package
```

Windowsのinstallerは、Windows上で次を実行して生成します。

```powershell
npm ci
npm run dist
```

`release/`へNSIS形式の`setup.exe`が生成されます。本番配布時はWindowsコード署名を
追加してください。

## Windows + WSL2

Inspectorの`Runtime`セクションで次を指定します。

```text
実行方式:
  WSL2

WSL distribution:
  Ubuntu-24.04

backend root:
  /home/kenshin/inference_backend2

runtime python:
  /home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10
```

Windowsで選択した`C:\...`は`/mnt/c/...`へ変換します。
`\\wsl.localhost\Ubuntu-24.04\home\...`は`/home/...`へ変換します。

GUIのinstallerにPython、モデル、TensorRT engineは含めません。対象PCのWSL2内に
現在の推論環境がセットアップされていることを前提とします。

## プロセス境界

```text
React renderer
  │ 限定されたIPC
  ▼
Electron main process
  │ native: <python> -m orchestration
  │ WSL2:   wsl.exe ... <python> -m orchestration
  ▼
inference_backend2
  ├── unified inference SQLite
  ├── postprocess mask SQLite
  ├── overlay proxy
  └── run_manifest.json
```

rendererではNode.jsを有効にしていません。`contextIsolation`とsandboxを有効にし、
preloadからファイル選択、設定保存、ジョブ操作だけを公開しています。

## ディレクトリ

```text
electron/
  main.ts             ウィンドウとIPC
  preload.ts          rendererへ公開する限定API
  job-manager.ts      Python/WSLプロセスとログ
  orchestration.ts    JSON設定と起動コマンド
  settings.ts         GUI実行環境の保存
  telemetry.ts        ログ行から進捗と速度を抽出

shared/
  types.ts            main/preload/renderer共通契約

src/
  App.tsx             レイアウトと状態
  styles.css          デザイントークンと全スタイル
  components/
    TopBar.tsx        入出力表示とtransport
    SourcePanel.tsx   入出力パス、既存SQLite、成果物
    MonitorPanel.tsx  pipeline flow、scope、timeline、metrics
    InspectorPanel.tsx 推論/後処理/overlay/runtimeの詳細設定
    ConsolePanel.tsx  実行ログ
    StatusBar.tsx     最下段のステータス
    Scope.tsx         throughputトレース
    ui.tsx            Panel/Row/入力部品
    Icons.tsx         16pxアイコン
  hooks/
    useSplit.ts       ペイン境界のドラッグと保存
  lib/
    api.ts            preload API とブラウザプレビュー
    defaults.ts       初期draft
    format.ts         時間・数値の整形
    stages.ts         stage計画と状態判定
```
