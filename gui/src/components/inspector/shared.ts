import type {
  ClassPostprocessRule,
  OverlayPreset,
} from "../../../shared/types";


export const CUT_METHODS = [
  { value: "high_precision", label: "高精度（推奨・FFmpeg）" },
  { value: "frame_diff", label: "フレーム差分（OpenCV）" },
];

export const OVERLAY_PRESETS: ReadonlyArray<{
  value: OverlayPreset;
  label: string;
}> = [
  { value: "genital-detailed", label: "性器・詳細" },
  { value: "genital-simple", label: "性器・簡易" },
  { value: "face-detailed", label: "顔・詳細" },
  { value: "face-simple", label: "顔・簡易" },
  { value: "combined-detailed", label: "両方・詳細" },
  { value: "combined-simple", label: "両方・簡易" },
];

export const SIMPLE_POSTPROCESS_CLASSES: ReadonlyArray<
  ClassPostprocessRule["className"]
> = ["男性器", "女性器", "結合部分"];

export function lines(value: string[]): string {
  return value.join("\n");
}
export function parseLines(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}
