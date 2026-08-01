# North — modern professional IDE-grade (JetBrains Fleet / refined VS Code lineage)

## Rationale

North is the lightest dark of the four round-2 variants: an elevated-gray ground
(#1A1B1E canvas, #202124 panels) instead of near-black, with one quiet IDE blue
(#5490DE) reserved strictly for primary action, selection/active state, progress and
focus. (Round-2 review: accent nudged from #4C8AD4 to #5490DE — brighter and cooler —
so the lineup shows two clearly different blues next to 05-slate's steel #4D7FB8.) Everything else is grayscale separated by 1px hairlines (8% white) and one-step
surface changes — no shadows on surfaces, no gradients, no glow, no tinted panels.
The personality axis is legibility and calm: 13px/1.5 base type, ~28px rows, 12–14px
paddings, and inset "wells" (canvas-colored boxes inside panels) for charts, flow
nodes, phase cards and the stage timeline. That inset/flat alternation is how Fleet
and modern VS Code builds hierarchy without decoration, and it is the whole visual
system here.

State is never a wash of color: the running queue item gets a neutral one-step raise,
a 46% flat 2px blue progress edge and a small outlined 処理中 badge; the active flow
node gets a 1px blue border and a blue 3px mini-bar; done states are a 6px green dot
or green-text badge. Progress is always a flat solid blue bar (2–4px). The six
hardware scopes keep per-metric trace colors (they encode data, not decoration) but
all six hues are desaturated to sit quietly on the ground; fills are flat 9% alpha,
gridlines 5% hairlines. The one permitted motion is a 2s opacity pulse on the
status-bar running dot.

Layout follows the app's real shell (46px TopBar / 250px Source / flexible Monitor /
296px Inspector / 6px row-resize handle / Console / 26px StatusBar). The console is
rendered at a user-resized 220px (the pane is resizable 46–560px in the real app, via
the 6px drag handle above it) so the entire 12-line sample tail, including the launch
command, is visible — a deliberate "working session" reading. Keyboard hints moved
from kbd chips into native tooltips (`title`), per the round-2 ban on ornament.
The 詳細 view is fully modeled: the 簡単-only クラス別設定 table swaps for the
クラス別GUI mode (形状・KF・補完 select + 4-column クラス別ルール editor with the
fixed その他（未指定） row), plus the 楕円近似（K2）, 旧形式の追加オーバーレイ and
extended 顔後処理 groups from ui-spec §5.2/5.3.

## Tokens

| Role | Value |
|---|---|
| Canvas / field / well | `#1A1B1E` |
| Panel / top & status bars | `#202124` |
| Raised (buttons, chips, active list row) | `#26272B` (active segment cell `#2E3035`) |
| Hover / quiet raise / disabled fill (one token for all) | `--hover: #232428` |
| Hairline | `rgba(255,255,255,.08)` (soft `.05`, strong `.13`) |
| Progress groove (every track: queue edge, node bars, phase bars, sliders) | `--track: rgba(255,255,255,.07)` |
| Completed fill (done bars, done connectors) | `--done-fill: #3E5A78` |
| Text | `#E0E1E4` / `#A4A6AB` / `#6F7176` |
| Accent (only: primary action, selection/active, progress, focus) | `#5490DE` (text-on-dark `#7AA6E0`) |
| Success / warn / danger | `#5EA87D` / `#C9A35C` / `#CC6E62` (danger text `#D98A7F`) |
| Scope traces (desaturated, data-only) | fps `#6D9BD3` · gpu `#9C89C6` · cpu `#66AC9C` · vram `#C79B6A` · mem `#7FB2C9` · temp `#C57F76` |
| Type | exactly four sizes: UI 13px/1.5 base · 12px controls/titles/values · 11px meta/hints · 10px micro (uppercase +0.05em EN only — Japanese is never letterspaced); mono Cascadia stack, `tabular-nums` on every numeral; single sanctioned exception: 20px overall-% readout (≤22px cap) |
| Radius | 6px containers/controls · 4px nested cells · 3px micro badges |
| Heights | rows 28px · controls 26px · panel header 34px · label column 96px |
| Spacing | 4px grid (2px half-steps): 12/14px pane padding, 10/14px block gaps |
| Progress | flat solid `#5490DE`, 2–4px |
| Shadows | none on surfaces (popovers only — none needed in this state) |

## Presentation changes vs. current app (zero functional change)

- Panel titles become 10px uppercase micro-labels; STATUS/LIVE tabs are text tabs
  with a 2px blue underline (IDE tab language) instead of pill tabs.
- Monitor content is grouped into flat inset wells: 6 flow-node cards with 1px
  connectors, 3×2 scope grid, readout row (20px overall %, ≤22px cap), stage
  timeline strip, 2×2 phase grid, and a bordered 7-cell metrics strip.
- Inspector keeps the exact 96px label column and section/badge structure; badges are
  outlined mono chips (on/性器 on green-text, fast/native neutral). All controls are
  rendered in their true busy-disabled state at reduced opacity.
- Queue rows: 48×27 thumbnail, block title/meta with real ellipsis, outlined status
  badges, and the 46% bottom progress edge on the processing item.
- kbd chips removed; shortcuts live in tooltips. LED glows, scanlines and the radial
  canvas glow of the current app are gone entirely.

## 業務ソフト感の根拠

- **色の規律**: 青は「主要アクション・選択・進捗・フォーカス」のみ。画面上の青の
  総面積は数%に抑え、残りはすべて無彩色のグレー3段+ヘアライン。意味色（緑/黄/赤）
  も小さなドットとバッジテキストに限定。色が状態を"語る"だけで、装飾しない。
- **精度が品質**: 4pxグリッド、コントロール高26px・行高28px・ラベル列96pxの完全統一、
  1px境界の一貫、等幅数字（tabular-nums）による桁の整列。派手さで隠せない分、
  寸法の正確さがそのまま「製品の完成度」として見える。
- **実データの密度**: 実行中のジョブ状態（46%進捗、6,480/14,400フレーム、GPU 97%、
  ログ12行のtail、開始時刻とETA）が全ペインで矛盾なく一致している。スクリーンショット
  を見た人が「動いている業務ソフトの実画面」と誤認するレベルの整合性を持たせた。
- **IDEの文法**: タブ下線・折りたたみセクション・インセットのチャートウェル・
  ステータスバーの左右レイアウトなど、JetBrains/VS Codeユーザーが体で知っている
  構文だけで構成。新奇なUIメタファーはひとつもない=学習コストゼロの道具感。
- **動きの抑制**: アニメーションはステータスバーの実行中ドットの微小な明滅ひとつ。
  それ以外は完全に静止しており、長時間の監視作業でも疲れない。

## Distinctness from a generic dark theme

The identity is the *lightest-ground* stance: elevated grays with inset wells and a
single desaturated IDE blue, more air than the sibling variants (28px rows, 14px
gaps), and Fleet-style tab/section grammar. A generic dark theme colors chrome;
North removes chrome and lets alignment, hairlines and one disciplined accent carry
the entire design.
