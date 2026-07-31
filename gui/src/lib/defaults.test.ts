import { describe, expect, it } from "vitest";
import { defaultDraft } from "./defaults";

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
      shapeMode: "polygon",
      classPostprocessPolicySource: "editor",
      scoreMin: 0.6,
      maxGap: 0,
      k2BatchSize: 128,
      k2PrepWorkers: 4,
      k2ForwardMode: "states_only",
      k2CudnnBenchmark: "on",
      k2Tf32: null,
      device: "cuda:0",
      faceMaskTarget: "eyes",
      eyeMaskShape: "rectangle",
    });
    expect(defaultDraft.postprocess.classPostprocessRules).toEqual([
      {
        className: "男性器",
        shapeMode: "polygon",
        keyframeInterval: 2,
        maxGap: 15,
      },
      {
        className: "女性器",
        shapeMode: "ellipse",
        keyframeInterval: 2,
        maxGap: 15,
      },
      {
        className: "結合部分",
        shapeMode: "ellipse",
        keyframeInterval: 2,
        maxGap: 15,
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
