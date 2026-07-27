import path from "node:path";
import type {
  AppSettings,
  PipelineDraft,
} from "../shared/types";

export interface OrchestrationConfig {
  schema_version: 1;
  input_video: string;
  output_root: string;
  execution: {
    runtime_python: string;
    resume: boolean;
  };
  inference: Record<string, unknown>;
  postprocess: Record<string, unknown>;
  overlay: Record<string, unknown>;
}

export interface LaunchSpec {
  executable: string;
  args: string[];
  cwd?: string;
}

export function buildOrchestrationConfig(
  draft: PipelineDraft,
  settings: AppSettings,
): OrchestrationConfig {
  const runtimePath = (value: string): string =>
    settings.backendMode === "wsl" ? windowsToWslPath(value) : value;
  const inference: Record<string, unknown> = {
    enabled: draft.inference.enabled,
    mode: draft.inference.mode,
    segmentation_model: draft.inference.segmentationModel,
    segmentation_backend: draft.inference.segmentationBackend,
    face_model: draft.inference.faceModel,
    face_classes: draft.inference.faceClasses,
    device: draft.inference.device,
    max_frames: draft.inference.maxFrames,
    warmup_frames: draft.inference.warmupFrames,
    face_warmup_iterations: draft.inference.faceWarmupIterations,
    fast_sqlite: draft.inference.fastSqlite,
  };
  if (!draft.inference.enabled) {
    inference.input_sqlite = runtimePath(draft.inference.inputSqlite);
  }

  const postprocess: Record<string, unknown> = {
    enabled: draft.postprocess.enabled,
    shape_mode: draft.postprocess.shapeMode,
    score_min: draft.postprocess.scoreMin,
    cut_detect: draft.postprocess.cutDetect,
    cut_method: draft.postprocess.cutMethod,
    remove_short_tracks_max_frames:
      draft.postprocess.removeShortTracksMaxFrames,
    keyframe_interval: draft.postprocess.keyframeInterval,
    device: draft.postprocess.device,
  };
  if (!draft.postprocess.enabled) {
    postprocess.tracked_sqlite = draft.postprocess.trackedSqlite
      ? runtimePath(draft.postprocess.trackedSqlite)
      : null;
    postprocess.final_sqlite = draft.postprocess.finalSqlite
      ? runtimePath(draft.postprocess.finalSqlite)
      : null;
  }

  return {
    schema_version: 1,
    input_video: runtimePath(draft.inputVideo),
    output_root: runtimePath(draft.outputRoot),
    execution: {
      runtime_python: settings.runtimePython,
      resume: false,
    },
    inference,
    postprocess,
    overlay: {
      enabled: draft.overlay.enabled,
      raw: draft.overlay.raw,
      tracked: draft.overlay.tracked,
      final: draft.overlay.final,
      faces: draft.overlay.faces,
      final_include_faces: draft.overlay.finalIncludeFaces,
      mask_alpha: draft.overlay.maskAlpha,
      show_labels: draft.overlay.showLabels,
      codec: draft.overlay.codec,
      start_frame: draft.overlay.startFrame,
      end_frame: draft.overlay.endFrame,
      progress_every: draft.overlay.progressEvery,
    },
  };
}

export function windowsToWslPath(value: string): string {
  const uncMatch =
    /^\\\\wsl(?:\.localhost)?\\[^\\]+\\(.*)$/i.exec(value);
  if (uncMatch) {
    return `/${uncMatch[1].replaceAll("\\", "/")}`;
  }
  const match = /^([a-zA-Z]):[\\/](.*)$/.exec(value);
  if (!match) {
    return value.replaceAll("\\", "/");
  }
  return `/mnt/${match[1].toLowerCase()}/${match[2].replaceAll("\\", "/")}`;
}

export function buildLaunchSpec(
  settings: AppSettings,
  configPath: string,
  dryRun: boolean,
): LaunchSpec {
  const workflowArgs = [
    "-m",
    "orchestration",
    "--config",
    configPath,
    ...(dryRun ? ["--dry-run"] : []),
  ];
  if (settings.backendMode === "native") {
    return {
      executable: settings.runtimePython,
      args: workflowArgs,
      cwd: path.resolve(settings.backendRoot),
    };
  }

  const backendRoot = windowsToWslPath(settings.backendRoot);
  const wslConfig = windowsToWslPath(configPath);
  const args = [
    ...(settings.wslDistro ? ["-d", settings.wslDistro] : []),
    "--cd",
    backendRoot,
    "--",
    settings.runtimePython,
    "-m",
    "orchestration",
    "--config",
    wslConfig,
    ...(dryRun ? ["--dry-run"] : []),
  ];
  return { executable: "wsl.exe", args };
}
