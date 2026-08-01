# Shared guidelines — Mask Pipeline Studio design proposals

Four independent, high-fidelity visual redesign proposals for the existing Electron app
"Mask Pipeline Studio" (`gui/`). Each proposal is a **standalone static mockup**
(HTML/CSS + minimal vanilla JS) in its own folder under `gui/design-proposals/`.

## Hard rules

1. **UI/UX presentation only — zero functional change.** Every control, label, panel,
   and piece of information shown MUST exist in the real app (see `_shared/ui-spec.md`).
   Do not invent features, buttons, or data the app does not have. You MAY change
   grouping, hierarchy, panel chrome, tab styling, iconography, density, and
   micro-interactions — that is the point.
2. **Same state everywhere.** All four mockups render the exact SHARED SAMPLE STATE
   defined in `ui-spec.md` (same queue items, same progress numbers, same logs,
   same metrics) so the four directions can be compared apples-to-apples.
3. **Do not touch anything outside your own variant folder.** Never modify `gui/src`,
   `gui/electron`, or another variant.
4. Use the exact Japanese labels from `ui-spec.md` (推論 / 後処理 / オーバーレイ /
   実行環境 …). Keep English only where the real app uses English.

## Layout contract

Fixed desktop-app layout (Electron, DaVinci/Premiere-class NLE):

- TopBar (app identity + run controls) — SOURCE (left: video queue) — MONITOR
  (center: status/live) — INSPECTOR (right: settings) — CONSOLE (bottom: logs) —
  StatusBar (bottom edge: hardware/job metrics).
- You may adjust proportions and panel chrome but not remove regions.
- The app fills the window: `100vh`, no page scroll. Wrap the app in
  `.viewport { height: 100vh; overflow: auto; }` and give the app frame
  `min-width: 1280px; min-height: 720px;` so smaller windows scroll *inside* the
  viewport wrapper — the body itself must never scroll horizontally.
- Design target: 1920×1080. Must also look composed at 1366×900.
- Default visible state = the richest one: Monitor on the status tab, mid-run,
  Inspector open on the 推論 section. (Screenshots capture the default state.)

## File contract (exact skeleton — downstream tooling depends on it)

```
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask Pipeline Studio — <VariantName></title>
<style>
/* ALL CSS in this single block */
</style>
</head>
<body>
<!-- all markup -->
<script>
/* ALL JS in this single block */
</script>
</body>
</html>
```

- **Fully self-contained. Zero external requests** — no webfonts, no CDNs, no image
  files. Icons are inline SVG (`stroke="currentColor"`, ~1.5px stroke, consistent
  grid). Video thumbnails / live preview frames are CSS-gradient or inline-SVG
  placeholders that read as dark video frames (vignette, faint detail) — never
  broken-image boxes, never emoji.
- Fonts: deliberate **system stacks** only. Suggested:
  - UI: `"Segoe UI Variable Text","Segoe UI",system-ui,"Helvetica Neue",Arial,"Noto Sans JP",sans-serif`
  - Data/mono: `"Cascadia Code",Consolas,"SF Mono",ui-monospace,monospace`
  - A serif display stack is allowed only where a variant brief calls for it.
- `font-variant-numeric: tabular-nums` on every metric, timer, and table of digits.

## Interactivity (small, demonstrative — JS is class-toggling only, ≤ ~120 lines)

- Monitor tabs switch (ステータス / ライブ).
- Inspector sections collapse/expand; the 簡単/詳細 toggle shows/hides advanced rows.
- Hover states on everything interactive; visible `:focus-visible` rings.
- `prefers-reduced-motion` respected (transitions off).

## Quality bar — "paid pro tool", never "default Qt"

Premium tells (do): a layered surface scale (3–4 elevations of one hue temperature);
ONE accent used with discipline; a consistent 4px spacing grid; one radius language
per variant; hairline borders (`rgba` ~6–13% white) instead of heavy strokes; styled
`select`/checkbox/slider/scrollbar; restrained motion (~150ms); realistic data;
uppercase+letterspacing for tiny EN labels only — **never letterspace Japanese text**;
semantic colors (success/warn/danger) kept separate from the accent and used sparingly.

Cheap tells (ban): default/unstyled form controls; default focus outlines; pure
`#000`/`#fff` surfaces; harsh uniform `#333` borders; bevels, glossy gradients;
inconsistent gaps; emoji as icons; centered-everything; clashing greys of mixed hue
temperature; fake lorem content; scroll bars in default chrome style.

Dark-only is a deliberate choice for all four variants (pro NLE convention) — no
light theme required.

Accessibility: body text contrast ≥ 4.5:1 against its surface; secondary text ≥ 4.5:1
where practical, micro-labels ≥ 3:1; interactive targets ≥ 24px hit area.

## Mandatory self-verification loop

1. Build `index.html` in your variant folder.
2. `cd /home/kenshin/inference_backend2/gui && node design-proposals/_shared/screenshot.mjs <abs path to index.html> <variant folder>/shot`
3. **Read** `shot-1920.png` and `shot-1366.png` as images. Critique yourself
   ruthlessly: spacing rhythm on the 4px grid, alignment, contrast, hierarchy,
   overflow warnings printed by the script, and the one question that matters —
   *"would this pass as a paid professional tool next to Resolve or Premiere?"*
4. Fix and repeat. Minimum 2 full build→screenshot→review iterations. Stop only when
   you would ship it.

## NOTES.md (per variant)

2–3 paragraphs of rationale; a token table (colors, type roles, radii, spacing);
the UX-presentation changes vs. the current app and why; what makes this variant
distinct from a generic dark theme.
