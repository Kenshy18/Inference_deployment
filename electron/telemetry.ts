import type { JobTelemetry } from "../shared/types";

export function emptyTelemetry(totalFrames: number | null = null): JobTelemetry {
  return {
    processedFrames: 0,
    totalFrames,
    fps: null,
    computeFps: null,
    detections: 0,
    masks: 0,
    faces: 0,
    elapsedSeconds: 0,
    progress: null,
  };
}

function withProgress(telemetry: JobTelemetry): JobTelemetry {
  const progress =
    telemetry.totalFrames && telemetry.totalFrames > 0
      ? Math.min(1, telemetry.processedFrames / telemetry.totalFrames)
      : null;
  return { ...telemetry, progress };
}

export function parseTelemetryLine(
  current: JobTelemetry,
  line: string,
): JobTelemetry {
  let next = { ...current };

  const progress = /\[progress]\s+processed=(\d+)\/(\d+|\?)\s+detections=(\d+)\s+fps=([\d.]+)/.exec(
    line,
  );
  if (progress) {
    next.processedFrames = Number(progress[1]);
    if (progress[2] !== "?") {
      next.totalFrames = Number(progress[2]);
    }
    next.detections = Number(progress[3]);
    next.fps = Number(progress[4]);
  }

  const completed =
    /processed\s+(\d+)\s+frames\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+fps\)/.exec(
      line,
    );
  if (completed) {
    next.processedFrames = Number(completed[1]);
    next.elapsedSeconds = Number(completed[2]);
    next.fps = Number(completed[3]);
  }

  const throughput =
    /measured compute throughput:\s+([\d.]+)\s+img\/s/.exec(line);
  if (throughput) {
    next.computeFps = Number(throughput[1]);
  }

  const orchestrator =
    /\[orchestrator]\s+frames=(\d+)\s+detections=(\d+).*segmentations=(\d+)/.exec(
      line,
    );
  if (orchestrator) {
    next.processedFrames = Number(orchestrator[1]);
    next.detections = Number(orchestrator[2]);
    next.masks = Number(orchestrator[3]);
  }

  const overlay =
    /\[overlay]\s+frames=(\d+).*masks=(\d+)\s+faces=(\d+)/.exec(line);
  if (overlay) {
    next.processedFrames = Number(overlay[1]);
    next.masks = Number(overlay[2]);
    next.faces = Number(overlay[3]);
  }

  return withProgress(next);
}
