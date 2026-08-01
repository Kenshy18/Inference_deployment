# Atelier — warm luxury minimalism

## Rationale

Atelier treats the pipeline runner as a **mastering suite**: the operator loads material,
verifies the chain, and lets it run — so the interface behaves like a quiet instrument, not a
dashboard. Everything sits on one warm, near-black ground (#141210) with a single champagne
accent that is *only* allowed to mean "active": the running queue item's frame, the active
pipeline node and its progress, the selected tab/segment, the running phase's rule, the status
dot. Completed work is ivory, waiting work is umber — the eye finds the live edge of the
pipeline instantly because gold appears nowhere else. There are **no shadows anywhere**;
depth comes exclusively from three surface values (#141210 canvas, #1B1815 panels, #232019
raised) and 1px warm hairlines, which is what gives the UI its printed, engraved character.

The second signature is typographic: hero readouts — the overall **45.8%**, the ETA pair
(経過時間 / 予測所要), scope currents (21.4 fps, 97%…), phase percentages — are set in a
Palatino-led serif display stack, while everything operational stays in a 13px sans and
a mono reserved for machine strings (paths, stage ids, logs, JOB id). The serif numerals do
the emotional work a colored gauge would normally do, so the palette can stay silent.
All serif readouts force **lining tabular figures** (`font-variant-numeric: lining-nums
tabular-nums` + `font-feature-settings:"lnum","tnum"`), and the stack puts Noto Serif before
Georgia so non-Windows renders never fall back to old-style proportional figures that would
shimmy under a live timer. Tiny English labels (QUEUE, PIPELINE · 6 STEPS, WALL FPS, PHASE…)
are 10px uppercase mono with +0.10–0.12em tracking; Japanese is never letterspaced.

## Token table

| Role | Value |
|---|---|
| Canvas / Panel / Raised | `#141210` / `#1B1815` / `#232019` |
| Input & chart wells | `#100E0B` |
| Hairline / soft / strong | `rgba(214,197,166,.10)` / `.06` / `.18` |
| Active hairline (gold) | `rgba(201,168,106,.35)` — reserved for selected/active only |
| Text 1 / 2 / 3 | `#EAE4D8` / `#A69C8C` / `#7D7466` (≈4:1 — operative meta/hints) |
| Micro-chrome dim | `#6E665A` — only scope axes, step numerals 01–06, kbd chips |
| Accent champagne / hover / ink-on-accent | `#C9A86A` / `#D8BA80` / `#1C1408` |
| Success sage / danger clay | `#93AE8C` (done dots, on-badges) / `#CE7B62` (停止, failures) |
| Scope trace | monochrome `#C2B396`, fill fade `#C9B692` 18%→0 |
| Type: UI | 13px/1.5 Segoe UI · Noto Sans JP stack |
| Type: display | "Palatino Linotype", Palatino, Noto Serif, Georgia, serif — titles + hero numerals, lining+tabular figures forced |
| Type: data | Cascadia/Consolas mono, `tabular-nums` on every metric |
| Type scale | 10 (micro EN) / 11 (meta) / 12 (secondary) / 13 (body) + display 15 (panel titles) / 17 / 19 / 23 / 52 (compact display 42 / 20 / 18) |
| Micro labels | 10px uppercase mono, +0.12em (EN only) |
| Radius | 6px cards / 4px inner controls — strictly two values |
| Spacing | 4px grid; panel padding 18px, rows ≥34px, shell gaps 12px (compact: 14/10) |
| Elevation | none — hairlines only, zero `box-shadow` |

## UX-presentation changes vs. the current app (same features, new hierarchy)

- **Progress is the hero.** The viewer-footer readout moved to the top of the Monitor: a
  52px serif 45.8% with the run summary and the 経過時間/予測所要 pair on one baseline.
  The HUD (PIPELINE · 6 STEPS / BATCH 2 / 4 / 実行中 · 性器推論) becomes a hairline-thin
  instrument line above it.
- **Pipeline flow as a rail.** The node chain is drawn as per-link hairline segments
  between state dots (hollow → champagne → ivory), step numerals 01–06, and a 45% champagne
  mini-bar under the active node — replacing boxed nodes with a quieter, more legible line.
  Links into done/active nodes tint ivory (`rgba(234,228,216,.28)`) so the traversed part
  of the chain reads, and the rail is grid-derived — it can never overshoot the last dot.
- **Queue as gallery cards.** Each input is a thin-framed card: thumbnail, title row with a
  dot-status (処理済み sage / 処理中 champagne / 未処理 hollow), meta line, and — only on
  the processing item — a gold frame plus a 2px bottom progress bar at 46%.
- **Monochrome instrumentation.** All six scopes share one warm bone trace on inset wells
  instead of six rainbow colors; identity comes from the label and the serif current value.
  Semantic color is reserved for state, not decoration.
- **Console follows the tail** (追従 on): the log is pinned to line 132 like a real tailer;
  stage tags are tinted a desaturated bronze, the launch command line is brightened.
- Inspector keeps the real app's exact sections/controls (簡単/詳細 toggle, collapsible
  推論/後処理/オーバーレイ/実行環境, all controls disabled mid-run) restyled as hairline
  cards with gold-bordered active segments; 詳細 carries the full representative sets from
  the spec (K2 楕円近似, 顔後処理 tracking sliders, pipeline/score JSON, legacy overlays,
  FFmpeg / 処理範囲・ログ, 専門設定). Checked-but-disabled boxes render as a quiet
  raised well with a gold hairline and champagne glyph — a solid champagne fill is reserved
  for enabled interactive checks (e.g. Console 追従), keeping "gold = the live edge" scarce
  even with 詳細 open. Range/select controls carry `-moz-` mirrors so nothing drops to
  stock chrome off-Chromium.

## Distinctiveness

Not a generic dark theme because: (1) a warm, single-hue surface family — no neutral greys
anywhere; (2) serif display numerals as the primary data voice; (3) a one-accent discipline
where champagne strictly encodes "this is the live thing"; (4) zero shadows — an entirely
hairline-built elevation system; (5) the lowest density of the four variants: 1.5× paddings,
a breathing dark ground, and gallery-card queues. It is the variant that should feel the
most expensive sitting next to Resolve.
