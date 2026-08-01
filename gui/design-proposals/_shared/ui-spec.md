# Mask Pipeline Studio — UI Inventory Spec (single source of truth)

Audience: the four redesign designers. Every user-visible string below is quoted **exactly** as it appears in the current app (Japanese as-is). Sources: `gui/src/App.tsx`, `gui/src/components/*.tsx`, `gui/src/lib/*.ts`, `gui/shared/types.ts`, `gui/electron/job-manager.ts`, `gui/electron/telemetry.ts`, `orchestration/runner.py`. Where a behavior exists in the data model but is **not currently rendered**, it is flagged — you may choose to surface it.

The app is an Electron + React desktop GUI that drives a video mask pipeline (segmentation + face inference → postprocess → overlay rendering). One sequential job queue; one job runs at a time.

---

## 1. LAYOUT MAP

NLE-style fixed shell (no page scrolling; `body { overflow: hidden; min-width: 1080px }`):

```
┌───────────────────────────────────────────────────────────────┐
│ TopBar                                            46 px       │
├───────────┬─┬─────────────────────────────┬─┬─────────────────┤
│ SOURCE    │║│ MONITOR                     │║│ INSPECTOR       │
│ (left)    │║│ (center, flexible 1fr)      │║│ (right)         │
│ 250 px    │ │                             │ │ 296 px          │
├───────────┴─┴─────────────────────────────┴─┴─────────────────┤
│ ═ horizontal drag handle (6 px)                               │
│ CONSOLE                                          168 px       │
├───────────────────────────────────────────────────────────────┤
│ StatusBar                                         26 px       │
└───────────────────────────────────────────────────────────────┘
```

- Grid rows: `46px / minmax(0,1fr) / 6px / <console height> / 26px`. Main row columns: `var(--w-left, 250px) 6px minmax(0,1fr) 6px var(--w-right, 292px)`, `padding: 0 8px`.
- **Resizable**: left pane 250 px default (min 200, max 420); right pane 296 px default (min 240, max 460); console 168 px default (min 46 — collapses to just its header, max 560). Widths persist in localStorage (`mask-studio-split-left` / `-right` / `-console`). 6 px invisible drag handles between panes.
- Each pane is a `Panel`: a ~24 px header strip (uppercase-ish title on the left: `Source` / `Monitor` / `Inspector` / `Console`, optional meta text, actions on the right) above a scrollable body. Panels are rounded cards floating on a darker canvas.
- A transient toast can overlay the bottom of the shell (normal + `is-error` variants, auto-dismisses after 4 s). Example error toast texts: `動画ファイルではありません。`, `既にキューへ追加済みです。`(non-error), `実行できませんでした。`, `検証できませんでした。`.

---

## 2. TOPBAR

Left → right:

1. **Logo mark** — a film-strip icon (`FilmIcon`) in a small square chip. No app name text is rendered in the TopBar.
2. **File chips** (`fchip`, label-in-italics + bold value):
   - `QUEUE` chip — value `キューは空` (dimmed "empty" style when queue empty) or `` `${total}本 · 残り${pending}` `` e.g. `4本 · 残り2`. Tooltip: `入力キューの状態`.
   - `OUT` chip — value = last path segment of the output repository (e.g. `mask_out`), or `保存先未選択` (dimmed) when unset. Tooltip: full path.
3. **Transport** (right-aligned):
   - **`Dry Run`** button — check icon + text `Dry Run` + kbd hint `^D`. Tooltip: `設定と入力だけを検証 (Ctrl+D)`. Enabled only when `canRun`.
   - **`実行`** button (primary accent) — play icon + text `実行` + kbd hint `^↵`. Tooltip: `キューを順番に処理 (Ctrl+Enter)`. Shown when **not** busy; enabled only when `canRun`.
   - **`停止`** button (danger red) — stop icon + text `停止` + kbd hint `Esc`. Tooltip: `キューの処理を停止 (Esc)`. **Replaces** `実行` while busy; disabled while status is `cancelling`.

States (verified in `App.tsx`):
- `busy` = job status ∈ { `validating`, `running`, `cancelling` }.
- `canRun` = at least 1 pending queue item **and** output repository set **and** not busy.
- Keyboard shortcuts (global `keydown` listener): **Esc** → cancel (only while busy); **Ctrl+Enter / Cmd+Enter** → run queue (only when `canRun`); **Ctrl+D / Cmd+D** → dry run (only when `canRun`).

Job status labels (used by TopBar context, Monitor HUD and StatusBar):
`idle`=`待機中`, `validating`=`検証中`, `validated`=`検証済み`, `running`=`実行中`, `cancelling`=`停止中`, `cancelled`=`キャンセル`, `completed`=`完了`, `failed`=`失敗`.

---

## 3. SOURCE (left pane)

Panel header: title `Source`, meta `` `${queue.length}本` `` (e.g. `4本`).

### 3.1 Output repository row (top of body)
- Row label `出力リポジトリ` (stacked layout). Hint below: `各動画は「リポジトリ/動画名」へ出力します`, or `処理中は変更できません` while busy (input disabled).
- `PathInput` = text field (placeholder `全ジョブ共通の出力先`) + folder-icon browse button (tooltip `参照…`).

### 3.2 Input queue
- Sub-header bar: `入力キュー` + note `` `残り${pending}` `` (only when pending > 0) + small quiet button `追加` with plus icon (tooltip: `動画ファイルを選択して追加`) → opens multi-file video picker.
- **Add affordances**: (a) the `追加` picker button; (b) drag & drop video files anywhere onto the queue list (list gets a highlighted `is-dragover` state). Accepted extensions: mp4, mov, mkv, avi, webm. Non-video drops toast `動画ファイルではありません。`; duplicates toast `既にキューへ追加済みです。`.
- **Empty state** (icon + two lines): `ここに動画をドラッグ` / `または「追加」で選択します`.

**Queue item anatomy** (`qitem`, one row ~grid: thumb | main | badge):
- **Thumbnail** (left): probed JPEG video thumbnail (~192 px source width) or film icon fallback; for `pending` items a small duration overlay `em` in the corner, e.g. `05:21`.
- **Main**: bold **title** = filename without directory & extension (e.g. `studio_b_4k`); below it a one-line **meta** string:
  - pending: `` `MM:SS · W×H · FF.FF fps` `` e.g. `05:21 · 1280×720 · 30.00 fps` (parts omitted until probe completes)
  - processing / done: the run's settings summary, e.g. `V3 + Face V2 · ポリゴン · overlay fast` (format: models joined with ` + `, then `ポリゴン`/`楕円`, then `overlay ${executionMode}` or `overlayなし`; `既存SQLite` replaces models when inference is reused)
  - failed: the error text, fallback `処理に失敗しました`
  - done fallback when no summary: `完了`
- **Status badge** (right, `qbadge is-<status>`): `未処理` (pending) / `処理中` (processing) / `処理済み` (done) / `失敗` (failed).
- **Progress bar**: bottom edge of the processing item only — width = overall pipeline estimate %, or an indeterminate animation when no estimate.
- Tooltip on the whole row: full file path + newline + `右クリックで操作`.
- **Batch position** is not shown per-item; it appears in the Monitor HUD as `BATCH 2 / 4` (position = settled items + 1 while running).

**Context menu** (right-click on an input-queue item; closes on click/Esc/blur):
- `出力フォルダを開く` (folder icon) — only for `done` items
- `再処理（未処理に戻す）` — for `done` or `failed` items
- `停止して削除` (danger) — for the `processing` item
- `キューから削除` (danger) — for any non-processing item

### 3.3 Output queue (history)
- Sub-header bar: `出力キュー` + note `` `${n}本完了` `` (e.g. `1本完了`) + right-aligned hint `クリックでフォルダを開く` (only when non-empty).
- Empty state (compact, folder icon): `処理済み動画がここに表示されます`.
- **Output entry** (whole row is a button; opens the output folder): source thumbnail; bold name = output folder name (e.g. `interview_a_1080p`); meta = `` `MM/DD HH:MM · ${artifactCount}成果物 · ${summary}` `` e.g. `08/01 10:52 · 9成果物 · V3 + Face V2 · ポリゴン · overlay fast`; trailing folder icon (`aria-label` `出力フォルダを開く`). Tooltip: full output dir + `クリックして出力フォルダを開く`. One entry per completed run (re-runs of the same video append entries).

---

## 4. MONITOR (center pane)

Panel header: title `Monitor`; meta = inference-mode label: `性器のみ` / `性器 + 顔` / `顔のみ`. Header actions: tab strip + job chip.

- **Tabs** (exact labels): `STATUS` and `LIVE` (role=tablist, aria-label `Monitor view`).
- **Job chip** (right of tabs): `DRY RUN` during validation, else `` `JOB ${id.slice(0,19)}` `` e.g. `JOB 2026-08-01T10-58-24`, empty when idle.

### 4.1 STATUS tab

**(a) Viewer HUD** (one line above the flow): left `` `PIPELINE · ${n} STEPS` ``; center `` `BATCH ${position} / ${total}` ``; right `` `${STATUS_LABEL} · ${activeNodeLabel}` `` e.g. `実行中 · 性器推論` (fallback active label: `待機`).

**(b) Pipeline stage flow** — a horizontal chain of nodes with connector links (`is-done` / `is-active` states on links). Each **FlowNode**: zero-padded step number (`01`…), optional badge (`並列` when parallel inference; `再利用` on the reuse node), state dot, icon, bold label, value line, and a mini progress bar (determinate %, or indeterminate when active without progress). Node states: `waiting / ready / active / done / failed`.

Node sequence (built dynamically from the draft):
1. `入力` (video icon) — value: active item title, or `` `${total}本を待機` ``, or `動画を選択`
2. `性器推論` (cpu icon, when mode ≠ 顔) — value: model label (`V1`/`V2`/`V3`/`v3-lite`)
3. `顔推論` (eye icon, when mode ≠ 性器) — value: face model label (`Face V2`/`Face V1`)
   - When inference is disabled, nodes 2–3 are replaced by `推論結果` (database icon), value `既存SQLiteを再利用`, badge `再利用`
4. `後処理` (layers icon, when enabled) — value: `` `${policy}${cut}` `` where policy = `` `${n}クラス個別` `` (class editor) / `楕円` / `ポリゴン`, cut suffix = ` · カット先行` (precompute during inference) or ` · カット` or empty → e.g. `3クラス個別 · カット先行`
5. `オーバーレイ` (film icon, when enabled and ≥1 output) — value `` `${count}本 · ${mode}` `` with mode `CPU`/`NVENC`/`高速`, e.g. `2本 · 高速`
6. `出力` (database icon) — value `SQLite + 動画` or `SQLite`

**(c) Throughput / hardware scopes** — a 3×2 grid of mini line-chart scopes (SVG area+line, 3 gridlines, up to 180 samples; hardware polled every 1 s):

| Label | Unit | Line color | Scale |
|---|---|---|---|
| `FPS` | `fps` | `#5e8bff` | auto (`peak N.N` in axis) |
| `GPU` | `%` | `#a879ff` | fixed 100 |
| `CPU` | `%` | `#43c6ac` | fixed 100 |
| `VRAM` | `%` | `#ff9f5a` | fixed 100 |
| `MEMORY` | `%` | `#5ac8fa` | fixed 100 |
| `GPU TEMP` | `°C` | `#ff6577` | fixed 100 |

Scope chrome: header `LABEL` + current value with unit (or `no signal`); axis line `scale 100%` (or `peak 23.9`) and `` `${n} pt` `` (or `待機中`). Note: absolute VRAM/RAM byte values exist in `HardwareMetrics` (e.g. `vramUsedMiB`/`vramTotalMiB`) but are **not currently rendered** — only percentages.

**(d) Viewer footer** — left readout: big overall percentage (e.g. `45.8%`, `—` when idle) + summary sentence, e.g. `studio_b_4k — inference を処理中` (other summaries: `起動しています`, `停止を要求しました`, `設定と入力は妥当です`, `ジョブを中断しました`, `設定を検証しています`, `完了 — 残り2本`, `` `${n} 件の成果物` ``, `キューに動画を追加してください`, `出力リポジトリを選択してください`, `` `${n}本を実行できます` ``, `キューは全て処理済みです`). Right ETA block, two cells: value `06:12` label `経過時間`; value `13:32` label `予測所要`.
- The wall-clock predictor also computes `remainingSeconds`, `completionAt` and a **confidence** level (`live` / `pc-baseline` / `rough`) plus per-phase planned/remaining seconds (labels `推論` / `後処理` / `オーバーレイ` / `SQLite検証・出力`) — **computed but not currently rendered**; fair game for redesigns.

**(e) Stage timeline** — a single row of equal-width cells, one per planned orchestration stage, in run order. Cell label = stage label; states waiting/active/done/failed; active cell has a fill bar at the current frame fraction (indeterminate when unknown). Stage labels are English/mono style: `inference`, `postprocess`, `ovl combined_simple`, `ovl combined_detailed` (pattern `ovl <preset with _>`; legacy overlays: `ovl raw`, `ovl tracked`, `ovl final`, `ovl faces`). Empty fallback: `実行するステージがありません`.

**(f) Phase progress grid** — 4 fixed rows (2×2 grid), one per telemetry phase:

| Row label | Phase key | Count unit |
|---|---|---|
| `性器推論` | `segmentation_inference` | `フレーム` |
| `顔推論` | `face_inference` | `フレーム` |
| `後処理` | `postprocess` | `ステージ` |
| `オーバーレイ` | `overlay` | `フレーム` |

Row anatomy: bold label + right-aligned percent (`45.0%`; `約 ` prefix when estimated; `—` when unknown); a progress track (indeterminate while running w/o %); meta line with detail label left and counts right `` `6,480 / 14,400 フレーム · 21.4 fps` `` (or `総量を確認中` while running without total; `—` otherwise). Disabled phases render at 100% with detail `対象外`. Detail strings mapped for display: `モデル準備中` (model-loading), `フレーム処理` (frames), `描画・エンコード` (rendering), `セグメント結合` (concatenating), `ワーカー準備中` (preparing-workers), `完了` (complete), `待機中` (empty), plus suffix forms ` · 入力検証` / ` · 処理中` / ` · 出力検証` / ` · 完了`.

### 4.2 LIVE tab

- **Stage**: latest preview JPEG (max 5 fps, 960×540, pushed by the pipeline) with a moving scan-line effect and a `LIVE` badge (pulsing dot + word `LIVE`). Empty state, three lines: `LIVE PREVIEW` / `処理結果を待っています` / `最大5fps / 960 × 540 / 非同期プレビュー`. Alt text on the image: `` `処理フレーム ${frameIndex}` ``.
- **Meta strip**, six cells (label over value): `PHASE` (`性器推論` / `顔推論` / `後処理`, else raw phase string; `待機中` when no frame), `FRAME` (e.g. `6,480`), `TIMECODE` (`04:30`), `MODEL` — label switches to `STAGE` during postprocess — (e.g. `dinov3_codino`), `DETAIL` (preview detail/status, fallback `running`), `COALESCED` (dropped/coalesced frame count, e.g. `3`). All `—` when idle.
- Preview streaming is enabled only while the LIVE tab is selected.

### 4.3 Metrics strip (bottom of the panel, both tabs)

Seven cells, label + mono value (+ small unit), `—` when null: `wall fps` (2 decimals), `frames` (`` `6,480 / 14,400` ``), `detections`, `masks`, `faces`, `device` (e.g. `cuda:0`), `codec` (e.g. `h264_nvenc`, null when overlay disabled).

---

## 5. INSPECTOR (right pane)

Panel header: title `Inspector`; header action: a 2-button segment toggle **`簡単` / `詳細`** (SettingsView `simple` / `advanced`, persisted in localStorage `mask-studio-settings-view`, disabled while busy). `簡単` hides every advanced row/subsection below; `詳細` shows the full set. Four collapsible **Sections**, each with a chevron, name, and a small state badge; open state persists (`mask-studio-sections`; defaults: 推論/後処理/オーバーレイ open, 実行環境 closed). Nearly every control is **disabled while busy**.

Section names + badges (exact):
1. `推論` — badge `on` (green state) or `reuse` (off state)
2. `後処理` — badge `性器 on` / `性器 off`
3. `オーバーレイ` — badge = execution mode string (`cpu`/`nvenc`/`fast`) when enabled, `skip` when off
4. `実行環境` — badge `native` / `wsl2`

Row anatomy: fixed-width left label (96 px), control(s) right; optional inline hint below-right; optional tooltip via `title`. "stack" rows put the control full-width under the label. Rows can render dimmed (`is-off`) when contextually inactive.

### 5.1 `推論` — COMPLETE control list

Simple view (`簡単`):

| Label | Control | Options | Default |
|---|---|---|---|
| `推論` | checkbox `新規に実行` | — | **on** |
| `AI推論SQLite` (only when 新規に実行 off; stacked; hint `生成済みunified inference SQLiteをキューの全動画で再利用します`) | PathInput | placeholder `inference.sqlite` | `""` |
| `処理モード` | segment | `性器` / `両方` / `顔` | **`両方`** (`segmentation-face`) |
| `性器モデル` (hidden in 顔 mode) | select | `V1` (eva02_cascade) / `V2` (dinov3_cascade) / `V3` (dinov3_codino) / `v3-lite` (dinov3_codino_mh0) | **`V3`** |
| `推論エンジン` | select | `高速（デフォルト）` / `低速（安定）` (per-model registry; V2 has fast only) | **`高速（デフォルト）`** (`tensorrt-fast`) |
| `顔モデル` (hidden in 性器 mode) | select | `Face V2` (face_dino_v2) / `Face V1` (rtdetr_head_face) | **`Face V2`** |
| `顔推論エンジン` (tooltip: `選択中の顔モデルを実行するバックエンドです。現行モデルは新顔=TensorRT、旧顔=PyTorchに固定されています。`) | select | `高速（デフォルト）` (Face V2) or `低速（安定）` (Face V1) — single option each, so disabled | **`高速（デフォルト）`** |

Advanced view adds (`詳細`):

Sub-header `モデル入力・実行範囲`

| Label | Control | Options / range | Default |
|---|---|---|---|
| `顔検出の保存対象` (stacked; tooltip `顔モデルの出力からSQLiteへ保存する対象です。Headは頭部box、Faceは顔領域です。少なくとも1つを選択します。`) | 2 checkboxes | `Head（頭部box）` / `Face（顔領域）` (min 1 must stay checked) | **both on** |
| `顔TRT bundle` (Face V2 only; stacked; tooltip `新顔モデルのTensorRT bundleだけを上書きします。`) | mono text | placeholder `空欄: 自動選択` | `""` |
| `推論デバイス` | mono text | — | `cuda:0` |
| `処理上限` | number | min 1, placeholder `空欄: 全フレーム` | empty (null) |
| `速度計測除外` | number + unit `f` | min 0 | `0` |
| `顔ウォームアップ` (face modes) | number + unit `回` | min 0 | `3` |

Sub-header `性能`

| Label | Control | Notes | Default |
|---|---|---|---|
| `モデル同時推論` (hint `v3-lite + Face V2限定`, or `現在の組合せでは不可` when incompatible) | checkbox `性器・顔モデルを同時実行` | only enabled for mode 両方 + v3-lite + Face V2 | **off** |
| `顔→性器の開始差` (only when 同時推論 on; hint `0秒が実測上の推奨値`) | number + unit `秒` | min 0, step 0.1 | `0` |
| `SQLite書き込み` (hint `高速化する代わりに異常終了時の耐性が低下`) | checkbox `速度優先モード` | — | **off** |

Sub-header `専門設定`

| Label | Control | Default |
|---|---|---|
| `追加引数` (stacked; tooltip `将来の未型付け引数を1トークン1行で指定します。主要引数の上書きは禁止されます。`) | textarea, placeholder `--future-option\nvalue` | empty |

### 5.2 `後処理` — representative controls (full behavior in source)

Always / simple view:
- `性器後処理`: checkbox `追跡・整形を実行` (label becomes `顔のみでは不要` and disables in 顔 mode) — default **on**
- When off: `追跡後SQLite` (PathInput, placeholder `tracked.sqlite`), `旧最終SQLite` (PathInput, placeholder `predictions.sqlite`, hint `必要な場合だけ指定`)
- Simple only — sub-header `クラス別形状・キーフレーム`, row `クラス別設定`: a mini table, columns `クラス` / `形状` / `KF間隔`, one row per class with a `ポリゴン`/`楕円` segment and a number+`f` input. Default rows: `男性器` ポリゴン 2 / `女性器` 楕円 2 / `結合部分` 楕円 2
- `検出スコア下限`: number 0–1 step 0.01, placeholder `設定既定値` — default `0.6`
- `保存する顔マスク` (Face V2 modes; tooltip `最終result.sqliteへ保存するプライバシーマスクです。`): segment `なし` / `顔全体` / `目元` — default **`目元`**
- `目元形状` (when 目元): segment `楕円` / `長方形` — default **`長方形`**

Advanced adds (8+ representative):
1. `既定形状`: segment `ポリゴン` / `楕円` (hidden while class editor active; hint `JSONで未指定のクラスに適用` in file mode) — default ポリゴン
2. `既定KF間隔`: number, unit `f`, placeholder `設定既定値` — default `2`
3. `カット検出`: checkbox `カット位置を保存し、trackを分割` — default **on** (hint when both inference+postprocess reuse: `推論・後処理の両方を再利用する場合は実行不可`)
4. Sub-header `性器後処理 — 構成`: `パイプラインJSON` (mono text, placeholder `空欄: 標準パイプライン`), `クラス別スコアJSON` (placeholder `空欄: 共通の検出スコア下限`), `形状・KF・補完` select `共通値` / `クラス別GUI` / `クラス別JSON` — default **`クラス別GUI`**; in JSON mode `形状設定JSON` (placeholder `class_postprocess_policy.json`); in GUI mode `クラス別ルール` editor — table headers `確定クラス名` / `形状` / `KF間隔` / `補完上限`, first fixed row `その他（未指定）`, per-row remove `−`, add button `＋ クラスを追加`
5. Sub-header `カット検出`: `検出方式` select `高精度（推奨・FFmpeg）` / `フレーム差分（OpenCV）` — default 高精度; `推論と同時検出` checkbox `CPU検出をGPU推論と並行` (tooltip `別のFFmpeg縮小decodeをCPUで動かし、GPU推論時間へ重ねます。`) — default **on**
6. Sub-header `性器後処理 — 追跡・補完`: `短命track上限` number unit `f` (hint `指定フレーム以下のtrackを除去`) — default `10`; `既定補完上限` number unit `f` — default `0`
7. Sub-header `楕円近似（K2）` (rows dim unless an ellipse shape is in use): `モデルroot` (placeholder `空欄: 自動検出`), `K2 run directory` (placeholder `空欄: model root/k2_v5`), `GPUバッチ数` (default `128`), `CPU前処理worker` (default `4`), `計算精度` select `パイプライン既定`/`FP32`/`FP16` (default パイプライン既定), `計算範囲` select `パイプライン既定`/`必要値のみ（推奨）`/`全出力（診断用）` (default 必要値のみ（推奨）), `内部時間計測` select `パイプライン既定`/`計測する`/`計測しない（推奨）` (default 計測しない（推奨）), `cuDNN autotune` select `パイプライン既定`/`有効`/`無効` (default 有効), `TF32` select `パイプライン既定`/`PyTorch既定`/`有効`/`無効（再現性優先）` (default パイプライン既定), `K2デバイス` mono text (default `cuda:0`)
8. Sub-header `互換出力`: `旧形式SQLite` checkbox `旧形式も追加` — default off
9. Sub-header `顔後処理 — 追跡・プライバシーマスク` (Face V2 modes): `目キーポイント下限` slider 0–1 (default `0.35`), `追跡保持gap` number `f` (hint `未検出を許容する最大フレーム数`, default `5`), `追跡high閾値` slider (default `0.50`), `追跡low閾値` slider (default `0.05`), `短命track上限` number (hint `観測回数（hits）で判定`, default `2`), `短命保持スコア` slider (default `0.90`), `補完gap上限` number `f` (default `3`)
10. Sub-header `専門設定`: `追加CLI引数` textarea (tooltip `未型付けの将来オプション用です。上の管理済み引数は上書きできません。`)

### 5.3 `オーバーレイ` — representative controls

Simple view:
- `確認動画`: checkbox `オーバーレイを生成` — default **on**
- `表示プリセット` (stacked): 6 checkboxes — `性器・詳細` / `性器・簡易` / `顔・詳細` / `顔・簡易` / `両方・詳細` / `両方・簡易`; presets requiring an inactive mode are disabled; at least one output must remain. Default checked: **`両方・簡易` + `両方・詳細`**
- `エンコード`: segment `CPU` / `NVENC` / `高速` — default **`高速`** (fast = split-worker NVENC+CPU encode)
- `マスク濃度`: slider 0–1 step 0.01 — default `0.32`

Advanced adds (8+ representative):
1. Sub-header `旧形式の追加オーバーレイ（任意）` — `追加動画` (stacked, tooltip `通常の表示プリセットとは別に、旧ソフト互換の工程別MP4を追加生成します。SQLiteの内容は増えません。`): checkboxes `AI生出力（raw.mp4）` / `追跡後（tracked.mp4）` / `最終後処理（final.mp4）` / `顔box（faces.mp4）` — all default off; `finalへ顔を追加`: checkbox `互換finalへ顔boxを合成` — default off
2. Sub-header `描画内容` — `性器の描画データ`: segment `AI生マスク（後処理前）` / `最終マスク（後処理後）` — default 最終マスク（後処理後）
3. `描画時追加マスク` (tooltip `確認動画だけに顔/目元マスクを追加します。result.sqliteの保存内容は変更しません。`): segment `なし`/`顔全体`/`目元` — default `目元`; `追加目元マスク形状`: segment `楕円`/`長方形` — default `長方形`; `目キーポイント下限`: slider — default `0.35`
4. `顔詳細要素` (stacked): checkboxes `確率マスク` / `キーポイント` / `顔楕円` — all default **on**
5. `線幅`: two numbers with units `マスク` and `box` — defaults `2` / `2`; `ラベル`: checkbox `クラス・確信度・track ID` — default **on**
6. Sub-header `エンコード品質・速度` — mode-dependent: CPU: `x264 CRF` (0–51, default `18`, hint `小さいほど高画質` / `ビットレート指定中は不使用`), `x264 preset` select `ultrafast`…`veryslow` (default `veryfast`); NVENC: `NVENC CQ` (default `18`), `NVENC preset` `p1`–`p7`, `NVENC GPU` (default `0`); 高速: `CPU preset`, `NVENC preset` (default `p1`), `NVENC GPU`
7. `FFmpeg実行ファイル` (stacked): mono text, placeholder `空欄: 同梱FFmpeg`
8. `目標ビットレート`: number unit `Mbps` (fast: hint `高速モードでは必須（空欄時8 Mbps）`, placeholder `8.0`; else hint `空欄ならCRF/CQ品質指定`, placeholder `空欄: CRF/CQ`) — default `8` in fast mode
9. Fast only: `分割worker総数` (default `6`), `CPU割当数` (hint `残りをNVENCへ割当`, default `0`), `音声保持` checkbox `元音声をコピー` (off), `MP4 faststart` checkbox `Web再生用に最適化` (off)
10. Sub-header `処理範囲・ログ` — `フレーム範囲` (hint `開始 / 終了`): two numbers, defaults `0` / empty (placeholder `終端`); `進捗ログ間隔`: number unit `f` (default `300`; disabled in fast with hint `分割高速モードでは内部管理`, else hint `0で無効`)
11. Sub-header `専門設定` — `追加CLI引数` textarea

### 5.4 `実行環境`

| Label | Control | Options / placeholder | Default |
|---|---|---|---|
| `再開` (advanced only) | checkbox `完了済みstageを再利用` | — | off |
| `バックエンド` | segment | `Native` / `WSL2` | **Native** |
| `リポジトリroot` (stacked) | PathInput | placeholder `/home/user/inference_backend2` | backend repo path |
| `実行Python` (stacked) | PathInput | placeholder `/path/to/python3.10` | runtime python path |
| `WSL distribution` (WSL2 only; hint `wsl.exe -l -v`) | mono text | placeholder `Ubuntu-24.04` | `Ubuntu-24.04` |

---

## 6. CONSOLE (bottom pane)

Panel header: title `Console`; meta = current stage id (e.g. `inference`); actions right-to-left: follow checkbox `追従` (default on, auto-scrolls to tail), filter text input placeholder `フィルタ` (case-insensitive substring), counter `` `${n} lines` `` (e.g. `132 lines`).

Empty state: `Dry Run または実行を開始すると、orchestration の出力がここに流れます。`

**Log line format**: `gutter line-number` + monospace text. A leading `[tag]` (regex `^\[[^\]]+]`) is split out and tinted as a stage tag. Line styling classes: command lines start with `$ `; error styling when text matches `/error|traceback|exception|failed|not found/i`; warning styling on `/warn/i`. Ring buffer capped at 2,000 lines. `[phase-progress] {json}` and `[live-preview] {json}` control lines are consumed for telemetry/preview and **never shown**.

The orchestration runner prefixes every child-process line with its stage id: `[inference]`, `[postprocess]`, `[overlay_combined_simple]`, `[overlay_combined_detailed]`, `[overlay_raw]`, … Inner formats (verified in `telemetry.ts` + backend): `[progress] processed=N/M detections=D fps=F.FFF`, `processed N frames in S.SSSs (F.FFF fps)`, `measured compute throughput: F.FFF img/s`, `[orchestrator] role=… model=… backend=…`, `[orchestrator] frames=… detections=… classifications=… segmentations=… face_observations=… face_keypoints=…`, `[overlay] frames=… source_frame=… masks=… faces=…`. GUI-origin lines: `$ <launch command>`, `キャンセル要求を送信しました。`, `[gui] 既存の出力を保護するため新しい保存先を使用します: <path>`.

**Sample lines** — see §8.7.

---

## 7. STATUSBAR (26 px footer)

Left → right (each item is `label bold-value`):
1. Status dot (colored per status) + status label, e.g. `実行中`
2. `stage` + current stage id or `—` (mono), e.g. `stage inference`
3. `経過` + elapsed `H:MM:SS`/`MM:SS`, e.g. `経過 06:12`
4. `開始` + wall clock `HH:MM:SS` or `--:--:--`, e.g. `開始 10:58:24`
5. `終了` + wall clock or `--:--:--`
6. `exit` + exit code or `—`
7. (only on failure) the error text in red, path-truncated
Right-aligned group:
8. Backend mode: `NATIVE`, or `` `WSL2 ${distro}` `` e.g. `WSL2 Ubuntu-24.04`
9. `python` + python executable filename, e.g. `python python3.10` (tooltip: full path)
10. Backend root full path, e.g. `/home/user/inference_backend2` (truncating, tooltip full path)
11. (browser dev mode only) `BROWSER PREVIEW` in amber

---

## 8. SHARED SAMPLE STATE — "mid-run, item 2 of 4 at ~45%"

Every mockup renders exactly this state. All filenames are fictional; do **not** substitute real content names. Use `/home/user/...` paths verbatim.

### 8.1 Global
- Date/time now: **2026-08-01 11:04:36** local. Job started **10:58:24** → elapsed **06:12** (372 s).
- Job: id `2026-08-01T10-58-24-512Z`, status `running` (`実行中`), stage `inference`, dryRun false.
- Draft = defaults (§5 defaults; mode 両方, model V3 + Face V2, both `tensorrt-fast`, postprocess on w/ class editor, overlay fast w/ 両方・簡易+両方・詳細, mask alpha 0.32, bitrate 8 Mbps, workers 6).
- Output repository: `/home/user/videos/mask_out`.
- Settings: Native backend, backendRoot `/home/user/inference_backend2`, python `/home/user/.local/share/video-mask-runtime/envs/production/bin/python3.10`, WSL distro `Ubuntu-24.04` (unused).
- Inspector: view `簡単` selected; sections 推論/後処理/オーバーレイ open, 実行環境 closed; badges `on` / `性器 on` / `fast` / `native`; **all controls disabled** (busy).

### 8.2 TopBar
- QUEUE chip: `4本 · 残り2`. OUT chip: `mask_out` (tooltip `/home/user/videos/mask_out`).
- `Dry Run` disabled; `停止` button shown (enabled); `実行` not visible.

### 8.3 SOURCE queue (4 items, in order)

| # | Title (file) | Status badge | Meta line | Video facts |
|---|---|---|---|---|
| 1 | `interview_a_1080p` (`interview_a_1080p.mp4`) | `処理済み` | `V3 + Face V2 · ポリゴン · overlay fast` | 12:34, 1920×1080, 29.97 fps, 22,604 frames |
| 2 | `studio_b_4k` (`studio_b_4k.mp4`) | `処理中` + progress bar at **46%** | `V3 + Face V2 · ポリゴン · overlay fast` | 10:00, 3840×2160, 24.00 fps, 14,400 frames |
| 3 | `daily_scrum_720p` (`daily_scrum_720p.mp4`) | `未処理` | `05:21 · 1280×720 · 30.00 fps` (thumb overlay `05:21`) | 9,630 frames |
| 4 | `night_walk_b_1080p` (`night_walk_b_1080p.mp4`) | `未処理` | `03:47 · 1920×1080 · 59.94 fps` (thumb overlay `03:47`) | 13,606 frames |

- Sub-headers: `入力キュー` note `残り2`; `出力キュー` note `1本完了` + `クリックでフォルダを開く`.
- Output queue, 1 entry: `interview_a_1080p` — `08/01 10:52 · 9成果物 · V3 + Face V2 · ポリゴン · overlay fast` (dir `/home/user/videos/mask_out/interview_a_1080p`).

### 8.4 MONITOR — STATUS tab
- Header meta: `性器 + 顔`. Job chip: `JOB 2026-08-01T10-58-24`.
- HUD: `PIPELINE · 6 STEPS` | `BATCH 2 / 4` | `実行中 · 性器推論`.
- Flow nodes (state — label — value): 01 done `入力` `studio_b_4k` · 02 **active** `性器推論` `V3` mini-bar 45% · 03 done `顔推論` `Face V2` (Face V2 finished first) · 04 waiting `後処理` `3クラス個別 · カット先行` · 05 waiting `オーバーレイ` `2本 · 高速` · 06 waiting `出力` `SQLite + 動画`. No `並列` badges (parallel off).
- Scopes (current values): `FPS 21.4 fps` (auto scale, `peak 23.9`), `GPU 97 %`, `CPU 34 %`, `VRAM 78 %`, `MEMORY 41 %`, `GPU TEMP 68 °C`; each axis `180 pt`, fixed scopes `scale 100%`. Hardware narrative: **NVIDIA RTX 5090, VRAM 24.9 / 32 GB (78%), RAM 26.2 / 64 GB (41%), GPU 97%, CPU 34%, 68 °C** (absolute values currently appear nowhere — surface them if your design wants).
- Footer readout: **`45.8%`** + `studio_b_4k — inference を処理中`. ETA cells: `06:12` `経過時間` / `13:32` `予測所要`. (Estimator internals for this state: remaining ≈ 440 s, confidence `live` — not rendered today.)
- Timeline cells: `inference` **active, fill 45%** · `postprocess` waiting · `ovl combined_simple` waiting · `ovl combined_detailed` waiting.
- Phase grid:
  - `性器推論` — running, **`45.0%`**, detail `フレーム処理`, counts `6,480 / 14,400 フレーム · 21.4 fps`
  - `顔推論` — complete, `100.0%`, detail `完了`, counts `14,400 / 14,400 フレーム · 181.3 fps`
  - `後処理` — pending, `—`, detail `待機中`, counts `—`
  - `オーバーレイ` — pending, `—`, detail `待機中`, counts `—`
- Metrics strip: `wall fps 21.41` · `frames 6,480 / 14,400` · `detections 18,204` · `masks —` · `faces 21,347` · `device cuda:0` · `codec h264_nvenc`.

### 8.5 MONITOR — LIVE tab (if your mockup shows it)
Preview frame present (masked 4K frame downscaled to 960×540). Meta: `PHASE 性器推論` · `FRAME 6,480` · `TIMECODE 04:30` · `MODEL dinov3_codino` · `DETAIL frames` · `COALESCED 3`.

### 8.6 STATUSBAR
`● 実行中` · `stage inference` · `経過 06:12` · `開始 10:58:24` · `終了 --:--:--` · `exit —` ▏right: `NATIVE` · `python python3.10` · `/home/user/inference_backend2`.

### 8.7 CONSOLE
Header: meta `inference`, `132 lines`, filter empty, `追従` checked. Visible tail = these 12 lines (gutter numbers 121–132):

```
$ /home/user/.local/share/video-mask-runtime/envs/production/bin/python3.10 -m orchestration --config /home/user/.config/mask-pipeline-studio/jobs/2026-08-01T10-58-24-512Z/orchestration.json
[inference] [orchestrator] role=face model=face_dino_v2 backend=tensorrt-fast
[inference] processed 14400 frames in 79.412s (181.334 fps)
[inference] measured compute throughput: 205.118 img/s
[inference] [orchestrator] frames=14400 detections=14283 classifications=14283 segmentations=0 face_observations=21347 face_keypoints=106735
[inference] [orchestrator] saved unified SQLite: /home/user/videos/mask_out/studio_b_4k/inference.sqlite
[inference] [orchestrator] role=segmentation model=dinov3_codino backend=tensorrt-fast
[inference] [progress] processed=1440/14400 detections=4116 fps=21.198
[inference] [progress] processed=2880/14400 detections=8102 fps=21.264
[inference] [progress] processed=4320/14400 detections=12275 fps=21.377
[inference] [progress] processed=5760/14400 detections=16204 fps=21.398
[inference] [progress] processed=6480/14400 detections=18204 fps=21.412
```

Line 1 renders as a command line (`$ ` style); `[inference]` tags are tinted; none of these match the error/warn regexes.

---

## 9. CURRENT DESIGN SUMMARY

1. Modern **dark-only control surface**: rounded panels floating on a near-black canvas with a faint blue radial glow at the top; a single blue accent; amber for "running", green for OK, red for errors.
2. Dense, utilitarian **12 px base type**; 30 px rows / 26 px controls in the Inspector; heavy use of mono type for numbers, paths and log text; Japanese UI labels mixed with English technical terms (`Dry Run`, `STATUS`/`LIVE`, `wall fps`, stage ids).
3. Motion only where it carries state: hover/focus, indeterminate progress shimmer, live-preview scan line, toast slide-in; generous 2 px accent focus rings.
4. Custom-built controls throughout (portal dropdown, segments, checks, sliders, path inputs, context menu) — no native widget chrome; panels resize via invisible 6 px splitters.
5. Hierarchy is flat and text-driven; state is conveyed by tiny badges, colored dots and thin progress bars rather than large graphics.

**Current tokens** (from `styles.css` `:root`, for deliberate divergence):
- Canvas `--bg #0b0d11`; panel `#14171d` / alt `#171b22`; header `#181c23`; field `#0e1015` (hover `#12151b`).
- Lines: white at 7% / 5% / 14% alpha. Text `#ccd2da`, strong `#eef1f5`, muted `#8f97a3`, dim `#626a76`.
- Accent `#5e8bff` (strong `#82a5ff`, soft 14% alpha, ring 35% alpha); run/amber `#e8a33d`; ok `#34c98e`; error `#ef5f66`.
- Scope trace palette: `#5e8bff` `#a879ff` `#43c6ac` `#ff9f5a` `#5ac8fa` `#ff6577`.
- Fonts: UI `"Noto Sans JP", "Inter", system-ui, "Hiragino Kaku Gothic ProN", "Yu Gothic UI", sans-serif`; mono `ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Mono", Menlo, Consolas, "Noto Sans JP", monospace`. Base size 12 px.
- Radii: panel 10 px, control 7 px, small 5 px. Dimensions: row 30 px, control 26 px, label 96 px. Panel shadow `0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.18)`; popup shadow `0 12px 32px rgba(0,0,0,.55)`.
