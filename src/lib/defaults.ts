import type {
  AppSettings,
  JobSnapshot,
  PipelineDraft,
} from "../../shared/types";

export const defaultDraft: PipelineDraft = {
  inputVideo: "",
  outputRoot: "",
  inference: {
    enabled: true,
    inputSqlite: "",
    mode: "segmentation-face",
    segmentationModel: "dinov3_codino",
    segmentationBackend: "tensorrt-fast",
    faceModel: "rtdetr_head_face",
    faceClasses: ["Face", "Head"],
    device: "cuda:0",
    maxFrames: null,
    warmupFrames: 0,
    faceWarmupIterations: 3,
    fastSqlite: false,
  },
  postprocess: {
    enabled: true,
    trackedSqlite: "",
    finalSqlite: "",
    shapeMode: "polygon",
    scoreMin: 0.35,
    cutDetect: true,
    cutMethod: "high_precision",
    removeShortTracksMaxFrames: 10,
    keyframeInterval: 3,
    device: "cpu",
  },
  overlay: {
    enabled: true,
    raw: true,
    tracked: true,
    final: true,
    faces: true,
    finalIncludeFaces: true,
    maskAlpha: 0.32,
    showLabels: true,
    codec: "mp4v",
    startFrame: 0,
    endFrame: null,
    progressEvery: 300,
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
    const saved = JSON.parse(
      window.localStorage.getItem("mask-studio-draft") ?? "null",
    ) as Partial<PipelineDraft> | null;
    if (!saved) {
      return defaultDraft;
    }
    return {
      ...defaultDraft,
      ...saved,
      inference: { ...defaultDraft.inference, ...saved.inference },
      postprocess: { ...defaultDraft.postprocess, ...saved.postprocess },
      overlay: { ...defaultDraft.overlay, ...saved.overlay },
    };
  } catch {
    return defaultDraft;
  }
}
