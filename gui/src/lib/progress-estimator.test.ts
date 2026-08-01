import { describe, expect, it } from "vitest";
import type { JobSnapshot, QueueItem } from "../../shared/types";
import { defaultDraft, emptyJob } from "./defaults";
import { estimatePipelineProgress } from "./progress-estimator";

const VIDEO: Pick<
  QueueItem,
  "durationSeconds" | "width" | "height" | "fps" | "frameCount"
> = {
  durationSeconds: 176.51,
  width: 1920,
  height: 1080,
  fps: 30000 / 1001,
  frameCount: 5290,
};

function draft() {
  return structuredClone(defaultDraft);
}

function job(): JobSnapshot {
  return structuredClone(emptyJob);
}

describe("PC pipeline progress estimator", () => {
  it("predicts the measured 1080p v3-lite + Face V2 workflow scale", () => {
    const fastDraft = draft();
    fastDraft.inference.segmentationModel = "dinov3_codino_mh0";
    fastDraft.overlay.presets = ["combined-simple"];
    const value = estimatePipelineProgress(fastDraft, job(), VIDEO, 0);
    expect(value.estimatedTotalSeconds).toBeGreaterThan(95);
    expect(value.estimatedTotalSeconds).toBeLessThan(112);
    expect(value.confidence).toBe("pc-baseline");
  });

  it("assigns substantially more time to V3 than v3-lite", () => {
    const fastDraft = draft();
    fastDraft.inference.segmentationModel = "dinov3_codino_mh0";
    fastDraft.overlay.presets = ["combined-simple"];
    const fast = estimatePipelineProgress(fastDraft, job(), VIDEO, 0);
    const slowDraft = draft();
    slowDraft.inference.segmentationModel = "dinov3_codino";
    slowDraft.overlay.presets = ["combined-simple"];
    const slow = estimatePipelineProgress(slowDraft, job(), VIDEO, 0);
    expect(slow.estimatedTotalSeconds).toBeGreaterThan(300);
    expect(slow.estimatedTotalSeconds).toBeLessThan(350);
    expect(slow.estimatedTotalSeconds!).toBeGreaterThan(
      fast.estimatedTotalSeconds! * 2.5,
    );
  });

  it("uses live phase FPS to update ETA and global progress", () => {
    const currentJob = job();
    currentJob.status = "running";
    currentJob.telemetry.phases.segmentation_inference = {
      state: "running",
      completed: 2_645,
      total: 5_290,
      progress: 0.5,
      estimated: false,
      detail: "frames",
      fps: 150,
      activeElapsedSeconds: 12,
      updatedAtMs: Date.now(),
    };
    const value = estimatePipelineProgress(
      draft(),
      currentJob,
      VIDEO,
      25,
      new Date("2026-07-31T10:00:00+09:00"),
    );
    expect(value.overall).toBeGreaterThan(0);
    expect(value.overall).toBeLessThan(1);
    expect(value.remainingSeconds).toBeGreaterThan(0);
    expect(value.completionAt).not.toBeNull();
    expect(value.confidence).toBe("live");
  });

  it("does not invent startup progress or reset when frame progress begins", () => {
    const currentJob = job();
    currentJob.status = "running";
    currentJob.telemetry.phases.segmentation_inference = {
      ...currentJob.telemetry.phases.segmentation_inference,
      state: "running",
      detail: "model-loading",
      updatedAtMs: Date.now(),
    };
    const loading = estimatePipelineProgress(
      draft(),
      currentJob,
      VIDEO,
      12,
    );
    expect(loading.overall).toBe(0);

    currentJob.telemetry.phases.segmentation_inference = {
      ...currentJob.telemetry.phases.segmentation_inference,
      completed: 16,
      total: 5_290,
      progress: 16 / 5_290,
      detail: "frames",
      fps: 3,
    };
    const running = estimatePipelineProgress(
      draft(),
      currentJob,
      VIDEO,
      16,
    );
    expect(running.overall).toBeGreaterThanOrEqual(loading.overall!);
    expect(running.estimatedTotalSeconds).toBeLessThan(
      loading.estimatedTotalSeconds! + 10,
    );
  });

  it("accounts for the larger rendering cost of 4K", () => {
    const overlayOnly = draft();
    overlayOnly.inference.enabled = false;
    overlayOnly.postprocess.enabled = false;
    const hd = estimatePipelineProgress(overlayOnly, job(), VIDEO, 0);
    const uhd = estimatePipelineProgress(
      overlayOnly,
      job(),
      { ...VIDEO, width: 3840, height: 2160 },
      0,
    );
    expect(uhd.estimatedTotalSeconds).toBeGreaterThan(
      hd.estimatedTotalSeconds!,
    );
  });
});
