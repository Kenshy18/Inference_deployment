import { describe, expect, it } from "vitest";
import { emptyTelemetry, parseTelemetryLine } from "./telemetry";

describe("job telemetry parser", () => {
  it("tracks live inference progress using the configured frame limit", () => {
    const telemetry = parseTelemetryLine(
      emptyTelemetry(300),
      "[inference] [progress] processed=116/? detections=79 fps=15.630",
    );

    expect(telemetry.processedFrames).toBe(116);
    expect(telemetry.detections).toBe(79);
    expect(telemetry.fps).toBe(15.63);
    expect(telemetry.progress).toBeCloseTo(116 / 300);
  });

  it("captures completed throughput and aggregate counts", () => {
    let telemetry = emptyTelemetry(300);
    telemetry = parseTelemetryLine(
      telemetry,
      "[inference] processed 300 frames in 14.884s (20.156 fps)",
    );
    telemetry = parseTelemetryLine(
      telemetry,
      "[inference] measured compute throughput: 20.160 img/s",
    );
    telemetry = parseTelemetryLine(
      telemetry,
      "[inference] [orchestrator] frames=300 detections=1468 classifications=265 segmentations=265 face_observations=814 face_keypoints=4070",
    );

    expect(telemetry.elapsedSeconds).toBe(14.884);
    expect(telemetry.fps).toBe(20.156);
    expect(telemetry.computeFps).toBe(20.16);
    expect(telemetry.detections).toBe(1468);
    expect(telemetry.masks).toBe(265);
    expect(telemetry.faces).toBe(814);
    expect(telemetry.progress).toBe(1);
  });

  it("updates overlay frame and mask counters", () => {
    const telemetry = parseTelemetryLine(
      emptyTelemetry(300),
      "[overlay_final] [overlay] frames=200 source_frame=199 masks=166 faces=800",
    );

    expect(telemetry.processedFrames).toBe(200);
    expect(telemetry.masks).toBe(166);
    expect(telemetry.faces).toBe(800);
    expect(telemetry.progress).toBeCloseTo(2 / 3);
  });

  it("tracks four structured phases independently at sub-percent precision", () => {
    let telemetry = emptyTelemetry();
    telemetry = parseTelemetryLine(
      telemetry,
      '[inference] [phase-progress] {"phase":"segmentation_inference","state":"running","completed":1233,"total":5290,"detail":"frames","fps":145.3}',
    );
    telemetry = parseTelemetryLine(
      telemetry,
      '[inference] [phase-progress] {"phase":"face_inference","state":"running","completed":16,"total":5290,"detail":"frames","fps":170.2}',
    );
    telemetry = parseTelemetryLine(
      telemetry,
      '[postprocess] [phase-progress] {"phase":"postprocess","state":"running","completed":7,"total":30,"detail":"nms:input-validation","fps":null}',
    );

    expect(
      telemetry.phases.segmentation_inference.progress,
    ).toBeCloseTo(1233 / 5290);
    expect(telemetry.phases.segmentation_inference.fps).toBe(145.3);
    expect(telemetry.phases.face_inference.progress).toBeCloseTo(16 / 5290);
    expect(telemetry.phases.postprocess.progress).toBeCloseTo(7 / 30);
    expect(telemetry.phases.overlay.state).toBe("pending");
  });

  it("marks a phase complete even when its total was unavailable", () => {
    const telemetry = parseTelemetryLine(
      emptyTelemetry(),
      '[overlay] [phase-progress] {"phase":"overlay","state":"complete","completed":0,"total":null,"detail":"complete","fps":null}',
    );

    expect(telemetry.phases.overlay.progress).toBe(1);
    expect(telemetry.phases.overlay.state).toBe("complete");
  });

  it("keeps exact counts while accepting an explicitly estimated display value", () => {
    const telemetry = parseTelemetryLine(
      emptyTelemetry(),
      '[postprocess] [phase-progress] {"phase":"postprocess","state":"running","completed":2,"total":10,"display_progress":0.257,"estimated":true,"detail":"tracking:running","fps":null,"active_elapsed_seconds":8.4}',
    );

    expect(telemetry.phases.postprocess.completed).toBe(2);
    expect(telemetry.phases.postprocess.total).toBe(10);
    expect(telemetry.phases.postprocess.progress).toBe(0.257);
    expect(telemetry.phases.postprocess.estimated).toBe(true);
    expect(telemetry.phases.postprocess.activeElapsedSeconds).toBe(8.4);
    expect(telemetry.phases.postprocess.updatedAtMs).not.toBeNull();
  });

  it("never moves a running phase backward on a later estimate", () => {
    let telemetry = parseTelemetryLine(
      emptyTelemetry(),
      '[postprocess] [phase-progress] {"phase":"postprocess","state":"running","completed":2,"total":10,"display_progress":0.29,"estimated":true}',
    );
    telemetry = parseTelemetryLine(
      telemetry,
      '[postprocess] [phase-progress] {"phase":"postprocess","state":"running","completed":2,"total":10,"display_progress":0.25,"estimated":true}',
    );
    expect(telemetry.phases.postprocess.progress).toBe(0.29);
  });
});
