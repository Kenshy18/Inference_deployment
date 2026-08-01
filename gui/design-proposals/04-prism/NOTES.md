# Prism — Mask Pipeline Studio redesign proposal 04

## Rationale

Prism moves the app from "utilitarian dark control surface" to the modern software-tool
premium of Linear / Raycast / Vercel. The canvas is a deep cool near-black (`#0E0E13`)
with a faint violet cast, and three fixed very-low-alpha radial glows sit *behind* the
app frame. Every panel is a translucent layer (`rgba(255,255,255,.035)`) rather than an
opaque grey, so the ambient light bleeds through the surface stack and the whole shell
reads as glass sheets floating over one light source. The TopBar and StatusBar are true
glass — `backdrop-filter: blur(16px)` over the glow — which gives the frame its
signature depth without a single decorative image.

Color is spent with extreme discipline. There is exactly ONE brand gradient
(`#8B7CFF → #4CC2FF`) and it appears in exactly two places: the primary 実行 CTA
(styled in CSS; hidden in this mid-run state because the real app swaps it for 停止
while busy) and the master pipeline progress bar, which carries a softly glowing
cyan head that breathes at 2.2 s. Everything else that means "running" is flat violet
`#8B7CFF`: the active flow node ring, the per-phase bars, the processing queue item,
the console stage tag, the status dot. Green/amber/red are reserved for semantics
(done / warn / failed) and never mix with the accent. Micro-interaction polish carries
the rest: 150 ms transitions, hover lifts on the CTA, pill segmented controls with an
accent-tinted active state for form controls (neutral for chrome like the STATUS/LIVE
tabs), 10 px-mono kbd chips (`^D`, `Esc`, `^↵`) beside every transport action, and a
composite focus ring — 1 px `#8B7CFF` plus a 4 px `rgba(139,124,255,.35)` glow, drawn
with box-shadow so pills and cards glow along their own radius.

Information design follows the "developer-tool luxury" idea: the Monitor is a single
vertical narrative — pipeline flow → six telemetry scopes with gradient-filled traces →
one big 45.8% readout with the gradient master bar → stage timeline → phase grid →
metrics strip pinned to the panel floor. The scopes stretch with viewport height
(flex + max-height per breakpoint) so the layout is dense at 1366×900 and breathes at
1920×1080 without dead slack.

## Tokens

| Role | Value |
|---|---|
| Canvas | `#0E0E13` + 3 fixed radial glows (violet .13 / cyan .05 / violet .045 alpha); the primary violet glow is centered at 18% 4% so it visibly grazes the TopBar (fill `rgba(19,19,27,.45)` + blur) and the Source/Monitor panel tops |
| Panel / raised / field | `rgba(255,255,255,.035)` / `.06` / `.045` |
| Border / soft / inner highlight | `rgba(255,255,255,.08)` / `.06` / `.055` |
| Text primary / secondary / dim | `#E4E4EC` / `#9A9AA6` / `#63636E` |
| Accent / accent text / accent soft | `#8B7CFF` / `#B3A8FF` / `rgba(139,124,255,.14)` |
| Brand gradient (CTA + master bar only) | `linear-gradient(90deg,#8B7CFF,#4CC2FF)` |
| Focus ring | `0 0 0 1px #8B7CFF` + `0 0 0 4px rgba(139,124,255,.35)` (box-shadow, radius-aware) |
| Success / warn / danger | `#5BC48F` / `#E2B35C` / `#E56370` |
| Scope traces | `#4C8DFF` `#A879FF` `#43C6AC` `#FF9F5A` `#5AC8FA` `#FF6577` — area fills capped at .10 alpha (.08 for the warm VRAM/GPU-TEMP hues) fading to 0, so the 1.3 px stroke carries the value and high traces never read as alarm slabs |
| Radii | cards 10 px · controls 7 px · small 5 px · pills 999 px (token set only — no ad-hoc radii) |
| Type | UI 13 px (Segoe UI / system + Noto Sans JP), micro-caps 8.5–10 px mono with letterspacing (EN only — JP is never letterspaced), data/log mono 11–11.5 px, tabular-nums everywhere |
| Spacing | 4 px grid; rows 30 px; panel padding 12 px (10 px below 960 px height) |
| Elevation | `0 10px 30px rgba(0,0,0,.4)` + 1 px inner top highlight on raised surfaces |
| Motion | 150 ms ease; pulse/breathe on live indicators; all off under `prefers-reduced-motion` |

## UX-presentation changes vs. the current app (zero functional change)

- **Glass TopBar** over the ambient glow; QUEUE/OUT file chips become raised glass
  chips with mono micro-labels; transport buttons keep their exact labels, tooltips,
  disabled logic and kbd hints (`Dry Run ^D` disabled, danger 停止 `Esc` while busy).
- **Pipeline flow nodes** are now raised cards with step numerals, state dots
  (green done / pulsing violet active) and per-node mini progress — same six nodes,
  same values, stronger glanceability. Active node gets a violet ring + outer glow.
- **Scopes** get faint gradient area fills under the trace lines (≤ .10 alpha, fading
  to 0 — the stroke, not the fill, carries the value) and stretch vertically with
  the window; chrome (peak/scale, `180 pt`) is preserved verbatim.
- **Overall progress** is promoted from a text percentage to the single gradient
  master bar — an 8 px track with a 1 px inset edge and an 11 px glowing head — the
  one place besides the CTA the gradient may live.
- **Segmented controls everywhere are pills** (処理モード, 保存する顔マスク, エンコード,
  Monitor tabs, 簡単/詳細); form pills tint violet when active, chrome pills stay neutral.
- **Queue items** are hover-lifting cards; the processing item gets a violet tinted
  border + edge progress bar; status badges are tinted pills (green 処理済み, violet
  処理中); thumbnails are framed 16:9 chips with duration overlays.
- **Console** keeps the exact log grammar (gutter numbers, `$` command line, tinted
  `[inference]` stage tags); the stage id and the Monitor's JOB id share one mono
  chip class (`.chip-mono`), both wearing the violet `is-live` tint while running.
- Inspector sections keep the exact section names, badges (`on` / `性器 on` / `fast`
  / `native`), row set and defaults from the spec. Badge color is semantic: green
  means enabled, mode-string badges (`fast`, `native`) stay neutral, and violet is
  reserved exclusively for live/running state. The 簡単/詳細 toggle and section
  collapse remain interactive in the mockup for review purposes even though the live
  app disables them while busy; disabled dims only the affordance (checkbox glyphs,
  chevrons, browse buttons at .55), while value-bearing controls — selects, number
  fields, paths, active segments — keep full-opacity text on a flattened field so the
  operator can still read V3 / Face V2 / 0.6 / 目元 mid-run.

## What makes Prism distinct from a generic dark theme

A generic dark theme paints opaque grey boxes; Prism builds one light model: fixed
ambient glows → translucent layers → blur on the outermost chrome → 1 px inner
highlights standing in for edge lighting. Its color system is intentionally almost
monochrome — a single violet doing all interactive work, one gradient rationed to two
sanctioned locations, semantics kept pure — which is precisely the Linear/Raycast
signature. And the detailing is tool-grade rather than decorative: mono micro-caps for
telemetry chrome, tabular numerals in every figure, kbd chips on the real shortcuts,
pill segments, and a progress head that glows because it is *the* number that matters
mid-run.
