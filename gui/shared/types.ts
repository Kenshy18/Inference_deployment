export type BackendMode = "native" | "wsl";
export type InferenceMode = "segmentation" | "segmentation-face" | "face";
export type SettingsView = "simple" | "advanced";
export type SegmentationModel =
  | "dinov3_codino"
  | "dinov3_codino_mh0"
  | "dinov3_cascade"
  | "eva02_cascade";
export type SegmentationBackend =
  | "auto"
  | "tensorrt-fast"
  | "tensorrt-backbone"
  | "pytorch";
export type FaceModel = "rtdetr_head_face" | "face_dino_v2";
export type FaceBackend = "tensorrt-fast" | "pytorch";
export type FaceMaskTarget = "none" | "face" | "eyes";
export type EyeMaskShape = "ellipse" | "rectangle";
export type OverlayExecutionMode = "cpu" | "nvenc" | "fast";
export type OverlayPreset =
  | "genital-detailed"
  | "genital-simple"
  | "face-detailed"
  | "face-simple"
  | "combined-detailed"
  | "combined-simple";
export type ClassPostprocessPolicySource = "global" | "editor" | "file";

export interface ClassPostprocessRule {
  className: string;
  keyframeInterval: number;
}
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
  segmentationModel: SegmentationModel;
  segmentationBackend: SegmentationBackend;
  faceModel: FaceModel;
  faceBackend: FaceBackend;
  faceClasses: string[];
  faceTrtBundle: string;
  device: string;
  maxFrames: number | null;
  warmupFrames: number;
  faceWarmupIterations: number;
  parallelModels: boolean;
  parallelModelStaggerSeconds: number;
  fastSqlite: boolean;
  extraArgs: string[];
}

export interface PostprocessDraft {
  enabled: boolean;
  trackedSqlite: string;
  finalSqlite: string;
  pipelineConfig: string;
  classPolicyJson: string;
  classPostprocessPolicySource: ClassPostprocessPolicySource;
  classPostprocessPolicyJson: string;
  classPostprocessRules: ClassPostprocessRule[];
  scoreMin: number | null;
  cutDetect: boolean;
  cutMethod: string;
  precomputeCutsDuringInference: boolean;
  removeShortTracksMaxFrames: number | null;
  keyframeInterval: number | null;
  exportLegacySqlite: boolean;
  faceMaskTarget: FaceMaskTarget;
  eyeMaskShape: EyeMaskShape;
  minimumEyeConfidence: number;
  faceDetectionScoreThreshold: number;
  headDetectionScoreThreshold: number;
  faceTrackingMaxGapFrames: number;
  faceTrackingHighScoreThreshold: number;
  faceTrackingLowScoreThreshold: number;
  faceShortTrackMaxHits: number;
  faceShortTrackKeepScore: number;
  faceInterpolationMaxGap: number;
  extraArgs: string[];
}

export interface OverlayDraft {
  enabled: boolean;
  executionMode: OverlayExecutionMode;
  raw: boolean;
  tracked: boolean;
  final: boolean;
  faces: boolean;
  finalIncludeFaces: boolean;
  presets: OverlayPreset[];
  genitalSource: "raw" | "final";
  faceMaskTarget: FaceMaskTarget;
  eyeMaskShape: EyeMaskShape;
  minimumEyeConfidence: number;
  faceProbabilityMasks: boolean;
  faceKeypoints: boolean;
  faceEllipses: boolean;
  maskAlpha: number;
  outlineThickness: number;
  boxThickness: number;
  showLabels: boolean;
  codec: "h264" | "h264_nvenc";
  h264Crf: number;
  h264Preset:
    | "ultrafast"
    | "superfast"
    | "veryfast"
    | "faster"
    | "fast"
    | "medium"
    | "slow"
    | "slower"
    | "veryslow";
  ffmpegBin: string;
  nvencCq: number;
  workers: number;
  cpuWorkers: number;
  copyAudio: boolean;
  targetBitrateMbps: number | null;
  nvencPreset: "p1" | "p2" | "p3" | "p4" | "p5" | "p6" | "p7";
  nvencGpu: number;
  faststart: boolean;
  startFrame: number;
  endFrame: number | null;
  progressEvery: number;
  extraArgs: string[];
}

export interface PipelineDraft {
  inputVideo: string;
  outputRoot: string;
  execution: {
    resume: boolean;
  };
  inference: InferenceDraft;
  postprocess: PostprocessDraft;
  overlay: OverlayDraft;
}

export interface ArtifactMap {
  [name: string]: string;
}

export type ProgressPhase =
  | "segmentation_inference"
  | "face_inference"
  | "postprocess"
  | "overlay";
export type ProgressPhaseState =
  | "pending"
  | "running"
  | "complete"
  | "failed";

export interface PhaseProgress {
  state: ProgressPhaseState;
  completed: number;
  total: number | null;
  progress: number | null;
  estimated: boolean;
  detail: string;
  fps: number | null;
  activeElapsedSeconds: number | null;
  updatedAtMs: number | null;
}

export type PhaseProgressMap = Record<ProgressPhase, PhaseProgress>;

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
  phases: PhaseProgressMap;
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

export interface VideoProbe {
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  frameCount: number | null;
  /** JPEG data URL, ~192px wide. null when ffmpeg is unavailable. */
  thumbnail: string | null;
}

export type QueueItemStatus = "pending" | "processing" | "done" | "failed";

export interface QueueOutput {
  id: string;
  outputDir: string;
  summary: string | null;
  completedAt: string | null;
  artifactCount: number | null;
}

export interface QueueItem {
  id: string;
  path: string;
  title: string;
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  frameCount: number | null;
  thumbnail: string | null;
  status: QueueItemStatus;
  /** Per-item job folder under the output repository, fixed at run start. */
  outputDir: string | null;
  /** Inference-settings summary captured when the run started. */
  summary: string | null;
  /** Completion metadata retained by the output queue. */
  completedAt: string | null;
  artifactCount: number | null;
  /** Immutable history: one entry per successfully completed run. */
  outputs: QueueOutput[];
  error: string | null;
}

/** A lightweight point-in-time hardware sample used by the live monitor. */
export interface HardwareMetrics {
  timestamp: number;
  cpuPercent: number | null;
  memoryPercent: number | null;
  memoryUsedBytes: number | null;
  memoryTotalBytes: number | null;
  gpuPercent: number | null;
  vramPercent: number | null;
  vramUsedMiB: number | null;
  vramTotalMiB: number | null;
  gpuTemperatureC: number | null;
}

export interface LivePreviewFrame {
  jobId: string | null;
  dataUrl: string;
  phase: string;
  frameIndex: number;
  timestampSeconds: number;
  model: string;
  stage?: string;
  status?: string;
  detail?: string;
  width: number;
  height: number;
  generatedAtMs: number;
  dropped: number;
}

export interface MaskStudioApi {
  bootstrap(): Promise<BootstrapData>;
  pickFile(kind: FilePickerKind): Promise<string | null>;
  pickVideos(): Promise<string[]>;
  pickDirectory(): Promise<string | null>;
  probeVideo(path: string, settings: AppSettings): Promise<VideoProbe>;
  /** Resolve the filesystem path of a dropped File (Electron only). */
  pathForFile(file: File): string | null;
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
  setPreviewEnabled(enabled: boolean): Promise<void>;
  sampleHardware(): Promise<HardwareMetrics>;
  openOutput(path: string): Promise<string>;
  onJobUpdate(callback: (job: JobSnapshot) => void): () => void;
  onPreviewUpdate(callback: (frame: LivePreviewFrame) => void): () => void;
}
