import type {
  JobSnapshot,
  PhaseProgress,
  PipelineDraft,
  QueueItem,
} from "../../shared/types";

export type PredictionConfidence = "live" | "pc-baseline" | "rough";

export interface PhaseTimeEstimate {
  id: "inference" | "postprocess" | "overlay" | "packaging";
  label: string;
  plannedSeconds: number;
  remainingSeconds: number;
}

export interface PipelineProgressEstimate {
  overall: number | null;
  remainingSeconds: number | null;
  estimatedTotalSeconds: number | null;
  completionAt: Date | null;
  confidence: PredictionConfidence;
  phases: PhaseTimeEstimate[];
}

export type VideoEstimateInput = Pick<
  QueueItem,
  "durationSeconds" | "width" | "height" | "fps" | "frameCount"
>;

const BASELINE_PIXELS = 1920 * 1080;

const SEGMENTATION_FPS: Record<
  PipelineDraft["inference"]["segmentationModel"],
  number
> = {
  dinov3_codino: 23.3,
  dinov3_codino_mh0: 160,
  dinov3_cascade: 18,
  eva02_cascade: 12,
};

const SEGMENTATION_STARTUP_SECONDS: Record<
  PipelineDraft["inference"]["segmentationModel"],
  number
> = {
  dinov3_codino: 30,
  dinov3_codino_mh0: 8,
  dinov3_cascade: 15,
  eva02_cascade: 15,
};

const FACE_FPS: Record<PipelineDraft["inference"]["faceModel"], number> = {
  face_dino_v2: 190,
  rtdetr_head_face: 42,
};

function finitePositive(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

function frameCount(
  draft: PipelineDraft,
  video: VideoEstimateInput | null,
  job: JobSnapshot,
): number | null {
  const phaseTotals = Object.values(job.telemetry.phases)
    .map((phase) => finitePositive(phase.total))
    .filter((value): value is number => value !== null);
  const observed = phaseTotals.length > 0 ? Math.max(...phaseTotals) : null;
  const probed =
    finitePositive(video?.frameCount) ??
    (finitePositive(video?.durationSeconds) && finitePositive(video?.fps)
      ? Math.round(
          finitePositive(video?.durationSeconds)! * finitePositive(video?.fps)!,
        )
      : null);
  const available = observed ?? probed;
  const limit = finitePositive(draft.inference.maxFrames);
  if (available === null) {
    return limit;
  }
  return limit === null ? available : Math.min(available, limit);
}

function resolutionFactor(
  video: VideoEstimateInput | null,
  exponent: number,
): number {
  const width = finitePositive(video?.width);
  const height = finitePositive(video?.height);
  if (width === null || height === null) {
    return 1;
  }
  const ratio = Math.max(0.25, (width * height) / BASELINE_PIXELS);
  return Math.min(3, Math.max(0.7, ratio ** exponent));
}

function backendFactor(backend: string): number {
  return backend === "pytorch" ? 0.34 : 1;
}

function phaseRemaining(
  phase: PhaseProgress,
  plannedSeconds: number,
  nominalFps?: number,
  tailSeconds = 0,
): number {
  if (phase.state === "complete") {
    return 0;
  }
  if (phase.state !== "running") {
    return plannedSeconds;
  }
  if (
    phase.total !== null &&
    phase.total > 0 &&
    phase.completed > 0 &&
    nominalFps !== undefined
  ) {
    const liveFps = finitePositive(phase.fps) ?? nominalFps;
    return (
      Math.max(0, phase.total - phase.completed) / liveFps + tailSeconds
    );
  }
  if (phase.progress !== null) {
    return plannedSeconds * Math.max(0, 1 - phase.progress);
  }
  return plannedSeconds;
}

function overlayCount(draft: PipelineDraft): number {
  return (
    draft.overlay.presets.length +
    [
      draft.overlay.raw,
      draft.overlay.tracked,
      draft.overlay.final,
      draft.overlay.faces,
    ].filter(Boolean).length
  );
}

/**
 * PC-specific wall-clock estimator. Baselines are measured end-to-end rates
 * from this workstation at 1080p; structured phase FPS replaces a baseline as
 * soon as a phase starts producing frames.
 */
export function estimatePipelineProgress(
  draft: PipelineDraft,
  job: JobSnapshot,
  video: VideoEstimateInput | null,
  elapsedSeconds: number,
  now = new Date(),
): PipelineProgressEstimate {
  const frames = frameCount(draft, video, job);
  if (frames === null) {
    return {
      overall: job.status === "completed" ? 1 : null,
      remainingSeconds: null,
      estimatedTotalSeconds: null,
      completionAt: null,
      confidence: "rough",
      phases: [],
    };
  }

  const phases = job.telemetry.phases;
  const inferenceParts: Array<{
    planned: number;
    remaining: number;
  }> = [];
  let hasLiveRate = false;

  if (draft.inference.enabled && draft.inference.mode !== "face") {
    const nominal =
      (SEGMENTATION_FPS[draft.inference.segmentationModel] *
        backendFactor(draft.inference.segmentationBackend)) /
      resolutionFactor(video, 0.12);
    const planned =
      frames / nominal +
      SEGMENTATION_STARTUP_SECONDS[draft.inference.segmentationModel];
    const phase = phases.segmentation_inference;
    hasLiveRate ||= phase.fps !== null;
    inferenceParts.push({
      planned,
      remaining: phaseRemaining(phase, planned, nominal, 2),
    });
  }

  if (draft.inference.enabled && draft.inference.mode !== "segmentation") {
    const nominal =
      (FACE_FPS[draft.inference.faceModel] *
        backendFactor(draft.inference.faceBackend)) /
      resolutionFactor(video, 0.15);
    const planned = frames / nominal + 6;
    const phase = phases.face_inference;
    hasLiveRate ||= phase.fps !== null;
    inferenceParts.push({
      planned,
      remaining: phaseRemaining(phase, planned, nominal, 2),
    });
  }

  let inferencePlanned = 0;
  let inferenceRemaining = 0;
  if (inferenceParts.length > 0) {
    if (draft.inference.parallelModels && inferenceParts.length > 1) {
      inferencePlanned =
        Math.max(...inferenceParts.map((part) => part.planned)) + 4;
      inferenceRemaining =
        Math.max(...inferenceParts.map((part) => part.remaining)) +
        (inferenceParts.every((part) => part.remaining === 0) ? 0 : 4);
    } else {
      inferencePlanned =
        inferenceParts.reduce((sum, part) => sum + part.planned, 0) + 4;
      inferenceRemaining =
        inferenceParts.reduce((sum, part) => sum + part.remaining, 0) +
        (inferenceParts.every((part) => part.remaining === 0) ? 0 : 4);
    }
  }
  const estimates: PhaseTimeEstimate[] = [];
  if (inferencePlanned > 0) {
    estimates.push({
      id: "inference",
      label: "推論",
      plannedSeconds: inferencePlanned,
      remainingSeconds: inferenceRemaining,
    });
  }

  if (draft.postprocess.enabled) {
    const hasEllipse =
      draft.postprocess.shapeMode === "ellipse" ||
      draft.postprocess.classPostprocessRules.some(
        (rule) => rule.shapeMode === "ellipse",
      );
    const geometryRate = hasEllipse ? 1_050 : 1_250;
    const serialCutSeconds =
      draft.postprocess.cutDetect &&
      !draft.postprocess.precomputeCutsDuringInference
        ? 1.5 + frames / 1_350
        : 0;
    const planned = 5 + frames / geometryRate + serialCutSeconds;
    estimates.push({
      id: "postprocess",
      label: "後処理",
      plannedSeconds: planned,
      remainingSeconds: phaseRemaining(phases.postprocess, planned),
    });
  }

  const overlays = draft.overlay.enabled ? overlayCount(draft) : 0;
  if (overlays > 0) {
    const overlayFrames = Math.max(
      0,
      Math.min(
        frames,
        draft.overlay.endFrame === null
          ? frames
          : draft.overlay.endFrame + 1,
      ) - Math.min(frames, draft.overlay.startFrame),
    );
    const baseRate =
      draft.overlay.executionMode === "fast"
        ? 1_500
        : draft.overlay.executionMode === "nvenc"
          ? 600
          : 260;
    const nominal = baseRate / resolutionFactor(video, 0.58);
    const planned = overlays * (1.5 + overlayFrames / nominal);
    const phase = phases.overlay;
    hasLiveRate ||= phase.fps !== null;
    estimates.push({
      id: "overlay",
      label: "オーバーレイ",
      plannedSeconds: planned,
      // Aggregate display_progress accounts for multiple overlay outputs.
      remainingSeconds: phaseRemaining(phase, planned),
    });
  }

  const packagingPlanned = 1.5 + frames / 4_500;
  estimates.push({
    id: "packaging",
    label: "SQLite検証・出力",
    plannedSeconds: packagingPlanned,
    remainingSeconds: job.status === "completed" ? 0 : packagingPlanned,
  });

  const plannedSeconds = estimates.reduce(
    (sum, estimate) => sum + estimate.plannedSeconds,
    0,
  );
  let remainingSeconds = estimates.reduce(
    (sum, estimate) => sum + estimate.remainingSeconds,
    0,
  );
  if (job.status === "completed") {
    remainingSeconds = 0;
  }

  const running = job.status === "running" || job.status === "cancelling";
  const overall =
    job.status === "completed"
      ? 1
      : running
        ? Math.min(
            0.999,
            Math.max(
              0,
              elapsedSeconds / Math.max(0.001, elapsedSeconds + remainingSeconds),
            ),
          )
        : null;
  const remaining = running ? remainingSeconds : null;
  const estimatedTotalSeconds = running
    ? elapsedSeconds + remainingSeconds
    : plannedSeconds;

  return {
    overall,
    remainingSeconds: remaining,
    estimatedTotalSeconds,
    completionAt:
      job.status === "completed"
        ? null
        : new Date(
            now.getTime() +
              (remaining ?? estimatedTotalSeconds) * 1_000,
          ),
    confidence: hasLiveRate
      ? "live"
      : finitePositive(video?.width) !== null &&
          finitePositive(video?.frameCount) !== null
        ? "pc-baseline"
        : "rough",
    phases: estimates,
  };
}
