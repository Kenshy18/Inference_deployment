# Horizon — cinematic blue control surface

**Direction.** Contemporary Adobe / Frame.io professionalism: one cool hue temperature from
canvas to overlay, a single disciplined accent (#4C8DFF), and depth carried by layering +
light instead of heavy strokes. Every panel is a card with a 1px top inner light
(`inset 0 1px rgba(255,255,255,.045)`) floating on a canvas that carries a faint blue
radial glow at the horizon line (top edge) — the app reads like a graded still from a
color suite, not a settings dialog. Density is medium (8–12px paddings on a strict 4px
grid), radius language is two-step (6px surfaces/controls, 4px nested elements), and
every metric is semibold `tabular-nums`.

**The Monitor is the hero.** The pipeline stage flow is restyled as a frame-accurate
timeline strip: six cells on a shared rail with a tick ruler across the top (minor ticks
every 12px, taller majors every 60px), a continuous 3px progress rail along the bottom
edge (done = dim accent fill, active = gradient fill at 45% with a soft glow), zero-padded
step numbers and state dots — done dots are green (`--ok`) so the dot grammar matches the
pill grammar app-wide, while the rails stay accent-blue as the timeline metaphor. The
orchestration stage timeline below the readout repeats the same language — a left-aligned
label with the fill and bright playhead edge sweeping behind it NLE-style — so "where is
the run" is answered twice at two zoom levels with one visual grammar. The six telemetry
scopes keep the real app's trace hues in raised cards with mono micro-headers, but their
area fills sit at .13 peak opacity and the grid is capped at 380px tall so the `45.8%`
readout, stage timeline, and phase grid read second after the flow strip. The readout
gains a `残り時間 07:20` cell — surfacing `remainingSeconds`, which the estimator already
computes but the current app never renders (explicitly allowed by the spec). All fixed
scopes use the spec's exact axis string `scale 100%`, including GPU TEMP.

**Presentation changes vs. the current app (no functional change).**
- Status is chromatic-coded once: accent blue = running (dot, pills, fills), green = done,
  amber reserved for warnings, red only on 停止 and errors. The current app's amber-running
  is replaced by blue to keep the cinematic monochrome, with green strictly for "finished".
- Status badges become tinted translucent pills (処理中 / 処理済み / 未処理 / 失敗 and the
  Inspector section badges `on` / `性器 on` / `fast` / `native`).
- Monitor tabs STATUS / LIVE use the signature 2px gradient underline indicator.
- The processing queue item gets the selection tint (rgba(76,141,255,.13) gradient), an
  accent border and a glowing gradient progress bar on its bottom edge.
- The console keeps its gutter + tinted `[inference]` stage tags, pinned to the tail
  because 追従 is on; `$` command lines get a green prompt glyph.
- Inspector rows keep the 96→106px label column, custom checks/segments/selects/sliders.
  The busy state dims controls to 55% with `cursor:not-allowed`, `aria-disabled` on every
  control stub, and `disabled` on all 参照… buttons — row labels keep full contrast.
- Control stubs are real keyboard stops: selects/checks/sliders get `tabindex="0"` plus
  combobox/checkbox/slider roles (via load-time JS), so the 2px blue `:focus-visible`
  ring demonstrates on every control class, not just native buttons and inputs.
- 簡単/詳細 and section collapse are live (class-toggling JS only); LIVE tab shows the
  masked-frame preview placeholder with scan line + pulsing LIVE badge.
- The output-queue sub-header carries the spec hint クリックでフォルダを開く right-aligned
  (`.sh-hint`, 10px `--t3`, shown only while the queue is non-empty); the row tooltip
  additionally repeats クリックして出力フォルダを開く with the full output dir.

**Token table.**

| Role | Value |
|---|---|
| Canvas / panel / raised / overlay | `#0F1218` / `#151A23` / `#1B2130` / `#232B3D` |
| Field (inputs) | `#10141D` |
| Hairline / soft | `rgba(151,169,201,.13)` / `rgba(151,169,201,.08)` |
| Text 1 / 2 / 2b / 3 | `#DCE3EF` / `#92A0B5` / `#78879D` / `#5E6B80` — 2b (≈4.6:1) carries content-bearing meta (queue meta, flow values, phase counts, hints); 3 is reserved for uppercase micro-labels and axis chrome |
| Accent / gradient / text-on-dark | `#4C8DFF` / `#4C8DFF→#6FA5FF` / `#8FB4FF` |
| Selection tint | `rgba(76,141,255,.13)` |
| Success / warn / danger | `#58BE8C` / `#E3B25B` / `#E5685F` |
| Scope traces | `#4C8DFF #A879FF #43C6AC #FF9F5A #5AC8FA #FF6577` (area fills at .13 peak opacity) |
| Type | 7-step ramp 9 / 10 / 11 / 12 / 13 / 16 / 30; UI base 13px; EN micro-headers 11/10/9px uppercase +0.06–0.09em (never Japanese); metrics semibold tabular-nums; mono for paths/logs/ids |
| Radius | 6px surfaces + controls; 4px nested (checks, kbd, thumbnails, seg buttons); pills 999px |
| Spacing | 4px grid throughout (8/12px paddings, 4/8/12px gaps); 8px gutters between panes; 10px panel body padding; 30px rows / 26px controls |
| Elevation | inner top light + `0 1px 2px` + `0 10px 28px`; floating `0 8px 24px rgba(0,0,0,.35)`; progress glow `0 0 10px rgba(76,141,255,.45)` |
| Focus | 2px blue ring `rgba(76,141,255,.55)` |

**Distinctiveness.** Not a generic dark theme: one blue-tinted gray ramp (no mixed hue
temperatures), the timeline-strip treatment of pipeline state as the signature move, a
restrained gradient used exactly twice (primary progress + active fills), pill-chip status
language shared by queue, sections, and job id, and letterspaced EN micro-labels against
untouched Japanese text — the exact mix Premiere/Frame.io use to feel "engineered".
Layout deviates deliberately: 288px source / 312px inspector (spec allows 220–420px), the
metrics strip as a full-width data footer of the Monitor card, and a hardware section that
reads like scopes in a color page rather than a task-manager table.

**Status-color grammar (final).** Green = finished everywhere a state is named: 処理済み
pill, phase-card check + 完了, 1本完了 note, and the flow-strip done dots. Accent blue =
running (dots, pills, gradient fills) and the timeline rails (done-segment fills stay dim
accent so the strip reads as one continuous timeline). Amber = warnings only; red = 停止
and errors only.

**詳細 view coverage.** The advanced Inspector renders every documented group: the full
楕円近似（K2） block (モデルroot, K2 run directory, GPUバッチ数, CPU前処理worker, 計算精度,
計算範囲, 内部時間計測, cuDNN autotune, TF32, K2デバイス), the 旧形式の追加オーバーレイ（任意）
group (raw/tracked/final/faces + finalへ顔を追加), and FFmpeg実行ファイル. Remaining
curation vs. the spec is limited to per-mode encode variants that the current 高速 mode
hides anyway (x264 CRF/NVENC CQ).
