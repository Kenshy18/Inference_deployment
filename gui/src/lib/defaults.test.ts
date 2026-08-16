import { describe, expect, it } from "vitest";
import {
  defaultDraft,
  migrateProductionPostprocessDefaults,
} from "./defaults";

describe("default processing profile", () => {
  it("matches the adopted Inspector configuration without job-specific paths", () => {
    expect(defaultDraft.inputVideo).toBe("");
    expect(defaultDraft.outputRoot).toBe("");
    expect(defaultDraft.inference).toMatchObject({
      mode: "segmentation-face",
      segmentationModel: "dinov3_codino",
      segmentationBackend: "tensorrt-fast",
      faceModel: "face_dino_v2",
      faceBackend: "tensorrt-fast",
      device: "cuda:0",
      parallelModels: false,
      fastSqlite: false,
    });
    expect(defaultDraft.postprocess).toMatchObject({
      classPostprocessPolicySource: "editor",
      scoreMin: 0.6,
      keyframeInterval: 6,
      faceMaskTarget: "eyes",
      eyeMaskShape: "rectangle",
      faceDetectionScoreThreshold: 0.55,
      headDetectionScoreThreshold: 0.55,
    });
    expect(defaultDraft.postprocess.classPostprocessRules).toEqual([
      {
        className: "男性器",
        keyframeInterval: 6,
      },
      {
        className: "女性器",
        keyframeInterval: 6,
      },
      {
        className: "結合部分",
        keyframeInterval: 6,
      },
    ]);
    expect(defaultDraft.overlay).toMatchObject({
      executionMode: "fast",
      presets: ["combined-simple", "combined-detailed"],
      genitalSource: "final",
      faceMaskTarget: "eyes",
      eyeMaskShape: "rectangle",
      maskAlpha: 0.32,
      showLabels: true,
      codec: "h264_nvenc",
      workers: 6,
      targetBitrateMbps: 8,
      nvencPreset: "p1",
    });
  });
});

describe("Production postprocess draft migration", () => {
  it("moves former class defaults to the promoted profile", () => {
    const old = structuredClone(defaultDraft.postprocess);
    old.keyframeInterval = 2;
    old.classPostprocessRules = [
      { className: "男性器", keyframeInterval: 2 },
      { className: "女性器", keyframeInterval: 2 },
      { className: "結合部分", keyframeInterval: 2 },
      { className: "custom", keyframeInterval: 4 },
    ];
    const migrated = migrateProductionPostprocessDefaults("5", old);
    expect(migrated.keyframeInterval).toBe(6);
    expect(migrated.classPostprocessRules.slice(0, 3)).toEqual(
      defaultDraft.postprocess.classPostprocessRules,
    );
    expect(migrated.classPostprocessRules[3]).toEqual(old.classPostprocessRules[3]);
  });
});
