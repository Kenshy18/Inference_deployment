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
      "[inference] [orchestrator] frames=300 detections=1468 classifications=265 segmentations=265",
    );

    expect(telemetry.elapsedSeconds).toBe(14.884);
    expect(telemetry.fps).toBe(20.156);
    expect(telemetry.computeFps).toBe(20.16);
    expect(telemetry.detections).toBe(1468);
    expect(telemetry.masks).toBe(265);
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
});
