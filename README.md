# Mask Pipeline Studio

`inference_backend2`の推論、後処理、確認用overlayを操作するデスクトップGUIです。
Electron、React、TypeScriptで実装し、既存のPythonコードを直接importせず、
公開されている`python -m orchestration` CLIを子プロセスとして実行します。

## 画面構成

DaVinci Resolve / Premiere Proに倣った固定4ペインのNLE型レイアウトです。
装飾的な余白を持たず、1pxのhairlineと24pxの操作行で構成しています。

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
- Runtime設定は変更の500ms後に自動保存し、実行時にも保存します。

### ショートカット

| キー | 動作 |
| --- | --- |
| `Ctrl` + `Enter` | 実行 |
| `Ctrl` + `D` | Dry run |
| `Esc` | 実行中のジョブを停止 |

## 現在の機能

- 入力動画と出力フォルダの選択
- 新規推論または既存unified inference SQLiteの再利用
- 推論モード、モデル、backend、GPU、最大フレーム数、warmup、顔クラスの設定
- polygon/ellipse後処理、スコア、カット検出、短命track除去、キーフレーム間隔
- raw、tracked、final、faces確認用overlayとcodec、透明度、範囲、ラベルの設定
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
