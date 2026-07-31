import type {
  AppSettings,
  JobSnapshot,
  PipelineDraft,
} from "../../shared/types";
import { normalizeBackend, normalizeFaceBackend } from "./models";

export const DRAFT_STORAGE_VERSION = "4";

export const defaultDraft: PipelineDraft = {
  inputVideo: "",
  outputRoot: "",
  execution: {
    resume: false,
  },
  inference: {
    enabled: true,
    inputSqlite: "",
    mode: "segmentation-face",
    segmentationModel: "dinov3_codino",
    segmentationBackend: "tensorrt-fast",
    faceModel: "face_dino_v2",
    faceBackend: "tensorrt-fast",
    faceClasses: ["Face", "Head"],
    faceTrtBundle: "",
    device: "cuda:0",
    maxFrames: null,
    warmupFrames: 0,
    faceWarmupIterations: 3,
    parallelModels: false,
    parallelModelStaggerSeconds: 0,
    fastSqlite: false,
    extraArgs: [],
  },
  postprocess: {
    enabled: true,
    trackedSqlite: "",
    finalSqlite: "",
    shapeMode: "polygon",
    pipelineConfig: "",
    classPolicyJson: "",
    classPostprocessPolicySource: "editor",
    classPostprocessPolicyJson: "",
    classPostprocessRules: [
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
    ],
    scoreMin: 0.6,
    cutDetect: true,
    cutMethod: "high_precision",
    precomputeCutsDuringInference: true,
    removeShortTracksMaxFrames: 10,
    keyframeInterval: 2,
    maxGap: 0,
    modelRoot: "",
    k2RunDir: "",
    k2BatchSize: 128,
    k2PrepWorkers: 4,
    k2Precision: null,
    k2ForwardMode: "states_only",
    k2ProfileStages: false,
    k2CudnnBenchmark: "on",
    k2Tf32: null,
    device: "cuda:0",
    exportLegacySqlite: false,
    faceMaskTarget: "eyes",
    eyeMaskShape: "rectangle",
    minimumEyeConfidence: 0.35,
    faceTrackingMaxGapFrames: 5,
    faceTrackingHighScoreThreshold: 0.5,
    faceTrackingLowScoreThreshold: 0.05,
    faceShortTrackMaxHits: 2,
    faceShortTrackKeepScore: 0.9,
    faceInterpolationMaxGap: 3,
    extraArgs: [],
  },
  overlay: {
    enabled: true,
    executionMode: "fast",
    raw: false,
    tracked: false,
    final: false,
    faces: false,
    finalIncludeFaces: false,
    presets: ["combined-simple", "combined-detailed"],
    genitalSource: "final",
    // Privacy masks are already stored by postprocess. Overlay derivation is
    // an independent display override and must not silently alter a preset.
    faceMaskTarget: "eyes",
    eyeMaskShape: "rectangle",
    minimumEyeConfidence: 0.35,
    faceProbabilityMasks: true,
    faceKeypoints: true,
    faceEllipses: true,
    maskAlpha: 0.32,
    outlineThickness: 2,
    boxThickness: 2,
    showLabels: true,
    codec: "h264_nvenc",
    h264Crf: 18,
    h264Preset: "veryfast",
    ffmpegBin: "",
    nvencCq: 18,
    workers: 6,
    cpuWorkers: 0,
    copyAudio: false,
    targetBitrateMbps: 8,
    nvencPreset: "p1",
    nvencGpu: 0,
    faststart: false,
    startFrame: 0,
    endFrame: null,
    progressEvery: 300,
    extraArgs: [],
  },
};

export const emptyJob: JobSnapshot = {
  id: null,
  status: "idle",
  dryRun: false,
  stage: null,
  startedAt: null,
  completedAt: null,
  exitCode: null,
  error: null,
  logs: [],
  outputRoot: null,
  artifacts: {},
  telemetry: {
    processedFrames: 0,
    totalFrames: null,
    fps: null,
    computeFps: null,
    detections: 0,
    masks: 0,
    faces: 0,
    elapsedSeconds: 0,
    progress: null,
    phases: {
      segmentation_inference: {
        state: "pending",
        completed: 0,
        total: null,
        progress: null,
        estimated: false,
        detail: "",
        fps: null,
      },
      face_inference: {
        state: "pending",
        completed: 0,
        total: null,
        progress: null,
        estimated: false,
        detail: "",
        fps: null,
      },
      postprocess: {
        state: "pending",
        completed: 0,
        total: null,
        progress: null,
        estimated: false,
        detail: "",
        fps: null,
      },
      overlay: {
        state: "pending",
        completed: 0,
        total: null,
        progress: null,
        estimated: false,
        detail: "",
        fps: null,
      },
    },
  },
};

export const browserSettings: AppSettings = {
  backendMode: "native",
  backendRoot: "/home/kenshin/inference_backend2",
  runtimePython:
    "/home/kenshin/.local/share/video-mask-runtime/envs/production/bin/python3.10",
  wslDistro: "Ubuntu-24.04",
};

export function loadDraft(): PipelineDraft {
  try {
    const savedVersion = window.localStorage.getItem(
      "mask-studio-draft-version",
    );
    const saved = JSON.parse(
      window.localStorage.getItem("mask-studio-draft") ?? "null",
    ) as Partial<PipelineDraft> | null;
    if (!saved) {
      return defaultDraft;
    }
    const inference = { ...defaultDraft.inference, ...saved.inference };
    const normalizedInference = {
      ...inference,
      segmentationBackend: normalizeBackend(
        inference.segmentationModel,
        inference.segmentationBackend,
      ),
      faceBackend: normalizeFaceBackend(
        inference.faceModel,
        inference.faceBackend,
      ),
      parallelModels:
        inference.parallelModels &&
        inference.mode === "segmentation-face" &&
        inference.segmentationModel === "dinov3_codino_mh0" &&
        inference.faceModel === "face_dino_v2",
      parallelModelStaggerSeconds:
        inference.parallelModels &&
        inference.mode === "segmentation-face" &&
        inference.segmentationModel === "dinov3_codino_mh0" &&
        inference.faceModel === "face_dino_v2"
          ? inference.parallelModelStaggerSeconds
          : 0,
    };
    const postprocess = {
      ...defaultDraft.postprocess,
      ...saved.postprocess,
      classPostprocessRules:
        saved.postprocess?.classPostprocessRules
          ?.filter((rule) => rule.className.trim() !== "foreground")
          .map((rule) => ({
            ...rule,
          })) ?? defaultDraft.postprocess.classPostprocessRules.map((rule) => ({
          ...rule,
        })),
    };
    if (
      saved.postprocess?.classPostprocessPolicySource === undefined &&
      saved.postprocess?.classPostprocessPolicyJson
    ) {
      postprocess.classPostprocessPolicySource = "file";
    }
    if (
      normalizedInference.mode === "segmentation" ||
      normalizedInference.faceModel !== "face_dino_v2"
    ) {
      postprocess.faceMaskTarget = "none";
    }
    if (
      normalizedInference.mode === "face" &&
      postprocess.faceMaskTarget === "none"
    ) {
      postprocess.precomputeCutsDuringInference = false;
    }
    const mergedOverlay = {
      ...defaultDraft.overlay,
      ...saved.overlay,
    };
    const validPresets = mergedOverlay.presets.filter((preset) => {
      const needsFace =
        preset.startsWith("face-") || preset.startsWith("combined-");
      const needsSegmentation =
        preset.startsWith("genital-") || preset.startsWith("combined-");
      return (
        (!needsFace || normalizedInference.mode !== "segmentation") &&
        (!needsSegmentation || normalizedInference.mode !== "face")
      );
    });
    const presets =
      validPresets.length > 0
        ? validPresets
        : normalizedInference.mode === "face"
          ? (["face-simple"] as const)
          : normalizedInference.mode === "segmentation"
            ? (["genital-simple"] as const)
            : (["combined-simple"] as const);
    const overlay = {
      ...mergedOverlay,
      presets: [...presets],
      // Older builds silently ignored legacy stage flags whenever presets
      // existed. Clear those stale flags once before the two output families
      // become additive, while preserving an intentional legacy-only draft.
      raw:
        savedVersion === DRAFT_STORAGE_VERSION || presets.length === 0
          ? mergedOverlay.raw
          : false,
      tracked:
        savedVersion === DRAFT_STORAGE_VERSION || presets.length === 0
          ? mergedOverlay.tracked
          : false,
      final:
        savedVersion === DRAFT_STORAGE_VERSION || presets.length === 0
          ? mergedOverlay.final
          : false,
      faces:
        savedVersion === DRAFT_STORAGE_VERSION || presets.length === 0
          ? mergedOverlay.faces
          : false,
      finalIncludeFaces:
        savedVersion === DRAFT_STORAGE_VERSION || presets.length === 0
          ? mergedOverlay.finalIncludeFaces
          : false,
      faceMaskTarget:
        normalizedInference.mode !== "segmentation" &&
        normalizedInference.faceModel === "face_dino_v2"
          ? savedVersion === DRAFT_STORAGE_VERSION
            ? mergedOverlay.faceMaskTarget
            : ("none" as const)
          : ("none" as const),
      codec:
        mergedOverlay.executionMode === "cpu"
          ? ("h264" as const)
          : ("h264_nvenc" as const),
      targetBitrateMbps:
        mergedOverlay.executionMode === "fast"
          ? (mergedOverlay.targetBitrateMbps ?? 8)
          : mergedOverlay.targetBitrateMbps,
      cpuWorkers:
        mergedOverlay.executionMode === "fast"
          ? Math.min(mergedOverlay.cpuWorkers, mergedOverlay.workers)
          : 0,
      copyAudio:
        mergedOverlay.executionMode === "fast"
          ? mergedOverlay.copyAudio
          : false,
      faststart:
        mergedOverlay.executionMode === "fast"
          ? mergedOverlay.faststart
          : false,
    };
    return {
      ...defaultDraft,
      ...saved,
      execution: { ...defaultDraft.execution, ...saved.execution },
      inference: normalizedInference,
      postprocess,
      overlay,
    };
  } catch {
    return defaultDraft;
  }
}
