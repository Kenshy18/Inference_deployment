# Round 2 addendum — "professional-grade software" constraints

Client feedback on round 1 (folders 01–04): **rejected as too decorative.** The verdict:
"シンプルかつモダンで、あくまで業務用の本格的なソフトウェア感がいい" — simple, modern,
and above all it must feel like serious, working professional software. Round 1's
signature moves (amber timecode wells, champagne serif numerals, violet glass,
glowing gradients) read as *themes*, not tools.

Everything in `DESIGN_GUIDELINES.md` still applies (file contract, fidelity to
`ui-spec.md`, shared sample state, self-verification loop). The following OVERRIDES
tighten it.

## The one test that decides everything

**The squint test: could the 1920×1080 screenshot pass as a screenshot of real,
licensed, commercial professional software (Resolve, Premiere, Nuke, a JetBrains
IDE)?** Anything that makes a viewer think "this is a design concept" instead of
"this is a tool someone uses for work" is a defect.

## Banned outright (round 2)

- Gradient fills of any kind (buttons, progress bars, text, backgrounds).
  Progress bars are flat, solid color.
- Glow, bloom, colored box-shadows, `backdrop-filter`/glass, translucent tinted panels.
- Serif or display typefaces. One sans UI stack + one mono stack, nothing else.
- Gold/champagne/violet/neon hues. Accent chroma must stay muted/desaturated.
- Decorative animation (scanlines, sweeps, pulsing LEDs). A subtle opacity pulse on
  ONE small "running" indicator is the maximum permitted motion.
- Oversized hero numerals (> 22px). Big-number theatrics read as dashboard-demo, not tool.
- Decorative tick patterns, segmented-LED meters, "timecode wells", kbd chips as
  ornament (a plain shortcut hint in a tooltip/label is fine).
- Panel background tints in accent colors (e.g. amber-tinted running rows). State is
  shown with a 2px edge marker, a small solid badge, or text color — not a wash.

## Required stance

- Neutral surfaces carry the whole design: 3 surface steps max, one hue temperature,
  separated by hairlines and/or one-step surface changes. Flat.
- The accent appears ONLY on: primary action, selection/active state, focus, links.
  Everything else is grayscale. Semantic colors (success/warn/danger) muted and rare.
- Premium = precision: perfect 4px-grid alignment, consistent control heights,
  consistent 1px borders, disciplined type scale (2–3 sizes + micro-label),
  tabular numerals, restrained iconography (1.5px stroke, uniform grid).
- Density and spacing are the personality axis — not color, not effects.
- Controls look like controls (buttons/inputs/selects have quiet but visible bounds);
  nothing looks clickable that isn't.

## What distinguishes the four round-2 variants

Only these sober axes: neutral temperature (cool / pure / warm-dark / light-dark),
overall lightness, information density, separation technique (hairline vs surface-step),
accent hue (steel blue / industrial orange / blue-gray / IDE blue), radius language
(0–6px). Execution quality is identical; the *stance* differs.
