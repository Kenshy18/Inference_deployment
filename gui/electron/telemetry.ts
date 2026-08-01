import type {
  JobTelemetry,
  PhaseProgress,
  ProgressPhase,
  ProgressPhaseState,
} from "../shared/types";

const PHASES: ProgressPhase[] = [
  "segmentation_inference",
  "face_inference",
  "postprocess",
  "overlay",
];

function emptyPhase(): PhaseProgress {
  return {
    state: "pending",
    completed: 0,
    total: null,
    progress: null,
    estimated: false,
    detail: "",
    fps: null,
    activeElapsedSeconds: null,
    updatedAtMs: null,
  };
}

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
    phases: {
      segmentation_inference: emptyPhase(),
      face_inference: emptyPhase(),
      postprocess: emptyPhase(),
      overlay: emptyPhase(),
    },
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

  const marker = "[phase-progress]";
  const markerIndex = line.indexOf(marker);
  if (markerIndex >= 0) {
    try {
      const payload = JSON.parse(
        line.slice(markerIndex + marker.length).trim(),
      ) as {
        phase?: string;
        state?: string;
        completed?: number;
        total?: number | null;
        display_progress?: number | null;
        estimated?: boolean;
        detail?: string;
        fps?: number | null;
        active_elapsed_seconds?: number | null;
      };
      if (
        PHASES.includes(payload.phase as ProgressPhase) &&
        ["pending", "running", "complete", "failed"].includes(
          payload.state ?? "",
        )
      ) {
        const phase = payload.phase as ProgressPhase;
        const state = payload.state as ProgressPhaseState;
        const completed = Math.max(0, Number(payload.completed ?? 0));
        const total =
          payload.total === null || payload.total === undefined
            ? null
            : Math.max(0, Number(payload.total));
        const exactProgress =
          total !== null && total > 0
            ? Math.min(1, completed / total)
            : state === "complete"
              ? 1
              : null;
        const displayProgress =
          state === "complete"
            ? 1
            : typeof payload.display_progress === "number" &&
                Number.isFinite(payload.display_progress)
              ? Math.min(
                  1,
                  Math.max(exactProgress ?? 0, payload.display_progress),
                )
              : exactProgress;
        const updated: PhaseProgress = {
          state,
          completed,
          total,
          progress: displayProgress,
          estimated: Boolean(payload.estimated),
          detail: String(payload.detail ?? ""),
          fps:
            payload.fps === null || payload.fps === undefined
              ? next.phases[phase].fps
              : Math.max(0, Number(payload.fps)),
          activeElapsedSeconds:
            payload.active_elapsed_seconds === null ||
            payload.active_elapsed_seconds === undefined
              ? null
              : Math.max(0, Number(payload.active_elapsed_seconds)),
          updatedAtMs: Date.now(),
        };
        const previousProgress = next.phases[phase].progress;
        if (
          updated.progress !== null &&
          previousProgress !== null &&
          state === "running"
        ) {
          updated.progress = Math.max(previousProgress, updated.progress);
        }
        next.phases = {
          ...next.phases,
          [phase]: updated,
        };
        next.processedFrames = completed;
        next.totalFrames = total;
        next.progress = displayProgress;
        if (updated.fps !== null) {
          next.fps = updated.fps;
        }
        return next;
      }
    } catch {
      // A malformed control event is ignored and remains visible in the log.
    }
  }

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

  // The unified inference writer reports faces separately from generic
  // detections. Fast native overlays do not necessarily print the legacy
  // `[overlay] ... faces=` summary, so this is the authoritative face count.
  const faceObservations = /face_observations=(\d+)/.exec(line);
  if (faceObservations) {
    next.faces = Number(faceObservations[1]);
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
