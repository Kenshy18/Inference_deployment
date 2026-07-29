import { describe, expect, it } from "vitest";
import { defaultDraft } from "./defaults";
import { plannedStages } from "./stages";

describe("plannedStages", () => {
  it("includes preset and compatibility overlays additively", () => {
    const draft = structuredClone(defaultDraft);
    draft.overlay.presets = ["combined-simple", "face-detailed"];
    draft.overlay.raw = true;
    draft.overlay.faces = true;

    expect(plannedStages(draft).map((stage) => stage.id)).toEqual([
      "inference",
      "postprocess",
      "overlay_combined_simple",
      "overlay_face_detailed",
      "overlay_raw",
      "overlay_faces",
    ]);
  });
});
