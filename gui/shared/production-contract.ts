import type { ShapeMode } from "./types";

/** GUI mirror of the deployed genital postprocess contract. */
export const PRODUCTION_POSTPROCESS = Object.freeze({
  shapeMode: "polygon" as ShapeMode,
  defaultKeyframeInterval: 6,
  polygonGapFillMaxFrames: 15,
});

export function effectivePostprocessMaxGap(
  shapeMode: ShapeMode,
  configured: number,
): number {
  return shapeMode === PRODUCTION_POSTPROCESS.shapeMode
    ? PRODUCTION_POSTPROCESS.polygonGapFillMaxFrames
    : configured;
}
