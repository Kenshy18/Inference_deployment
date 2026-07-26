export type BackendMode = "native" | "wsl";
export type InferenceMode = "segmentation" | "segmentation-face" | "face";
export type ShapeMode = "polygon" | "ellipse";
export type JobStatus =
  | "idle"
  | "validating"
  | "validated"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export interface AppSettings {
  backendMode: BackendMode;
  backendRoot: string;
  runtimePython: string;
  wslDistro: string;
}

export interface InferenceDraft {
  enabled: boolean;
  inputSqlite: string;
  mode: InferenceMode;
  segmentationModel:
    | "dinov3_codino"
    | "dinov3_cascade"
    | "eva02_cascade";
  segmentationBackend:
    | "auto"
    | "tensorrt-fast"
    | "tensorrt-backbone"
    | "pytorch";
  faceModel: "rtdetr_head_face";
  faceClasses: string[];
  device: string;
  maxFrames: number | null;
  warmupFrames: number;
  faceWarmupIterations: number;
  fastSqlite: boolean;
}

export interface PostprocessDraft {
  enabled: boolean;
  trackedSqlite: string;
  finalSqlite: string;
  shapeMode: ShapeMode;
  scoreMin: number;
  cutDetect: boolean;
  cutMethod: string;
  removeShortTracksMaxFrames: number;
  keyframeInterval: number;
  device: "cpu";
}

export interface OverlayDraft {
  enabled: boolean;
  raw: boolean;
  tracked: boolean;
  final: boolean;
  faces: boolean;
  finalIncludeFaces: boolean;
  maskAlpha: number;
  showLabels: boolean;
  codec: "mp4v" | "h264" | "h264_nvenc";
  startFrame: number;
  endFrame: number | null;
  progressEvery: number;
}

export interface PipelineDraft {
  inputVideo: string;
  outputRoot: string;
  inference: InferenceDraft;
  postprocess: PostprocessDraft;
  overlay: OverlayDraft;
}

export interface ArtifactMap {
  [name: string]: string;
}

export interface JobTelemetry {
  processedFrames: number;
  totalFrames: number | null;
  fps: number | null;
  computeFps: number | null;
  detections: number;
  masks: number;
  faces: number;
  elapsedSeconds: number;
  progress: number | null;
}

export interface JobSnapshot {
  id: string | null;
  status: JobStatus;
  dryRun: boolean;
  stage: string | null;
  startedAt: string | null;
  completedAt: string | null;
  exitCode: number | null;
  error: string | null;
  logs: string[];
  outputRoot: string | null;
  artifacts: ArtifactMap;
  telemetry: JobTelemetry;
}

export interface BootstrapData {
  platform: NodeJS.Platform;
  settings: AppSettings;
  job: JobSnapshot;
}

export type FilePickerKind = "video" | "sqlite" | "python";

export interface MaskStudioApi {
  bootstrap(): Promise<BootstrapData>;
  pickFile(kind: FilePickerKind): Promise<string | null>;
  pickDirectory(): Promise<string | null>;
  saveSettings(settings: AppSettings): Promise<AppSettings>;
  validateWorkflow(
    draft: PipelineDraft,
    settings: AppSettings,
  ): Promise<JobSnapshot>;
  startWorkflow(
    draft: PipelineDraft,
    settings: AppSettings,
  ): Promise<JobSnapshot>;
  cancelWorkflow(): Promise<JobSnapshot>;
  openOutput(path: string): Promise<string>;
  onJobUpdate(callback: (job: JobSnapshot) => void): () => void;
}
