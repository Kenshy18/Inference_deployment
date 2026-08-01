# Graphite Console — design notes

## Rationale

Graphite Console treats Mask Pipeline Studio as a **broadcast-grade instrument**, in the
lineage of DaVinci Resolve and Blackmagic hardware panels. The app is rendered as one
machined block: panels are flat matte slabs (`#131315`) separated by 1&nbsp;px dark seams and
hairlines only — **zero shadows on panels** (shadows would imply floating cards; a console
is milled, not stacked). Every panel header is a slim 28&nbsp;px toolbar with a letterspaced
mono micro-label (`SOURCE / MONITOR / INSPECTOR / CONSOLE`), so chrome reads as tooling,
not decoration. Density is deliberately the highest of the four variants: 8&nbsp;px paddings,
24–26&nbsp;px control rows, 12.5&nbsp;px UI type, and every number set in the mono stack with
`tabular-nums`.

State is carried by a small, strict vocabulary repeated everywhere: **6&nbsp;px status LEDs**
(green = done/ok, pulsing amber = running, dim = waiting, blue = environment) on queue
items, flow nodes, section badges, the job chip and the status bar; **scrub-style progress
bars** — an amber core with a bright playhead edge and faint tick marks — for the master
progress, phase rows, queue item and flow minis; and **segmented precision meters** (thin
ticked bars in the trace color) atop each hardware scope. A single amber accent
(`#E8A44F`) is reserved for "attention/active/primary"; success, warning, danger and info
are separate muted channels. The TopBar centers a **transport cluster** — Dry&nbsp;Run,
停止 and an amber elapsed-timecode well — the one place the eye goes during a run,
exactly like a deck's transport row.

## Token table

| Role | Value |
|---|---|
| Canvas / seam | `#0A0A0B` / `#060607` |
| Panel / panel-2 / raised / well / field | `#131315` / `#101012` / `#1A1A1D` / `#0C0C0E` / `#0D0D0F` |
| Hairlines | `rgba(255,255,255,.07)` (plus .045 / .11 steps) |
| Text 1 / 2 / 3 | `#DEDEE1` / `#9B9BA1` / `#67676E` |
| Accent (hover / pressed / ink-on-accent) | `#E8A44F` (`#F2B563` / `#D8933D` / `#1A1206`) |
| Success / warn / danger / info | `#7DBE82` / `#E0B75F` / `#E06E5A` / `#7FA8D9` |
| Scope traces (FPS/GPU/CPU/VRAM/MEM/TEMP) | `#7FA8D9 #A88BD4 #6FBFB2 #E8A44F #7FC4DE #D97A6A` |
| Type | UI 12.5px/1.45 system+Noto Sans JP; micro-labels 10px caps +0.08em (EN only, mono — Japanese is never letterspaced); all metrics mono `tabular-nums` |
| Radii | 4px controls · 6px menus/transport · 0 on panels (hairline-separated slabs) |
| Spacing / density | 4px grid (4/8/12 padding steps, 2px fine offsets); 8px pane padding; 28px panel headers; 24–28px rows; inspector rows min 25px |
| Micro-text floor | recessive micro-labels ≥ 3:1 (console gutter `#5E5E66`, scope axes `#6E6E76`) |
| Hit areas | ≥ 24px: seg buttons 22px in 26px wells; quiet buttons 24px; mini-table segs 20px inside ≥ 24px rows |

## UX-presentation changes vs. the current app (same data, same controls)

- **Transport moved to a centered cluster** in the TopBar with an amber elapsed-timecode
  well (`経過 06:12`); job status + `stage` echo on the right. Queue/OUT chips get mono
  micro-labels instead of italics.
- **Hardware scopes fused with precision meters**: each of the six trend scopes carries a
  segmented current-value meter; VRAM/MEMORY axes surface the absolute byte values
  (`24.9 / 32.0 GB`, `26.2 / 64.0 GB`) that exist in `HardwareMetrics` but were never
  rendered.
- **Estimator surfaced**: the footer readout adds the computed-but-unrendered
  `残り 07:20` cell with a green `live`-confidence LED beside 経過時間/予測所要, plus a
  master scrub bar for the overall 45.8%.
- **Queue rows re-gridded**: title + status LED/badge on line one, the mono meta string
  spanning under the badge on line two, wrapping to at most two lines — the settings
  summary survives a 264px pane without losing its tail; the processing item gets a
  2px amber rail + bottom scrub.
- **Inspector as a rack**: section headers are flat toolbars with LED state chips
  (`on` / `性器 on` / `fast` / `native`); rows tightened to a 96px label gutter with
  strict JP line-breaking (`word-break:keep-all` + manual `<wbr>` break points); all
  controls visibly dimmed while busy, while tabs/toggles stay live.
- **Deliberate demo deviation**: spec §5 disables the `簡単`/`詳細` toggle while busy;
  this mock keeps it live so reviewers can inspect both views. Shipping behavior would
  add `disabled` to the toggle during a run, like every other Inspector control.
- **Busy accent discipline**: while running, selected-segment underlines drop to 40%
  amber and checked boxes desaturate to `#8A6A3A`, so full-chroma amber stays exclusive
  to live state (running LEDs, active node, scrub cores, elapsed timecode).
- Console keeps gutter numbers behind a hairline rule, tints stage tags info-blue and the
  launch command amber; status LEDs replace colored-text status everywhere.

## What makes it distinct from a generic dark theme

No gradients-on-surfaces, no glows, no floating cards: hierarchy comes from *milled*
separations (seams + hairlines + 3 surface steps of one hue temperature) and from the
instrument vocabulary — LEDs, ticked scrubs, segmented meters, timecode well — applied
consistently to every stateful element. Amber is used as calibration-mark, not paint: it
appears only where the machine is *doing something* (running LEDs, active flow node, the
active tab's underline, progress cores, the elapsed clock). Everything else is measured
graphite. The result reads as hardware you operate, not a website in dark mode.
