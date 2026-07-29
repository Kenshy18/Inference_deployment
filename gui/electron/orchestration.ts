import path from "node:path";
import type {
  AppSettings,
  ClassPostprocessRule,
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

export interface ClassPostprocessPolicyFile {
  schema_version: 1;
  default: {
    shape_mode: PipelineDraft["postprocess"]["shapeMode"];
    keyframe_interval: number;
    max_gap: number;
  };
  classes: Record<
    string,
    {
      shape_mode: PipelineDraft["postprocess"]["shapeMode"];
      keyframe_interval: number;
      max_gap: number;
    }
  >;
}

function validateClassRule(rule: ClassPostprocessRule, index: number): string {
  const label = rule.className.trim();
  if (!label) {
    throw new Error(`クラス別後処理の${index + 1}行目にクラス名が必要です。`);
  }
  if (!Number.isInteger(rule.keyframeInterval) || rule.keyframeInterval < 1) {
    throw new Error(`${label}: キーフレーム間隔は1以上の整数が必要です。`);
  }
  if (!Number.isInteger(rule.maxGap) || rule.maxGap < 0) {
    throw new Error(`${label}: 補完上限は0以上の整数が必要です。`);
  }
  return label;
}

export function buildClassPostprocessPolicy(
  draft: PipelineDraft,
): ClassPostprocessPolicyFile | null {
  const settings = draft.postprocess;
  if (settings.classPostprocessPolicySource !== "editor") {
    return null;
  }
  if (
    settings.keyframeInterval === null ||
    !Number.isInteger(settings.keyframeInterval) ||
    settings.keyframeInterval < 1
  ) {
    throw new Error(
      "クラス別後処理では既定キーフレーム間隔に1以上の整数が必要です。",
    );
  }
  if (
    settings.maxGap === null ||
    !Number.isInteger(settings.maxGap) ||
    settings.maxGap < 0
  ) {
    throw new Error("クラス別後処理では既定補完上限に0以上の整数が必要です。");
  }
  const classes: ClassPostprocessPolicyFile["classes"] = {};
  settings.classPostprocessRules.forEach((rule, index) => {
    const label = validateClassRule(rule, index);
    if (Object.hasOwn(classes, label)) {
      throw new Error(`クラス別後処理のクラス名が重複しています: ${label}`);
    }
    classes[label] = {
      shape_mode: rule.shapeMode,
      keyframe_interval: rule.keyframeInterval,
      max_gap: rule.maxGap,
    };
  });
  return {
    schema_version: 1,
    default: {
      shape_mode: settings.shapeMode,
      keyframe_interval: settings.keyframeInterval,
      max_gap: settings.maxGap,
    },
    classes,
  };
}

export function buildOrchestrationConfig(
  draft: PipelineDraft,
  settings: AppSettings,
): OrchestrationConfig {
  if (
    draft.postprocess.classPostprocessPolicySource === "file" &&
    !draft.postprocess.classPostprocessPolicyJson.trim()
  ) {
    throw new Error(
      "クラス別JSONを選択した場合は、形状設定JSONのパスが必要です。",
    );
  }
  const runtimePath = (value: string): string =>
    settings.backendMode === "wsl" ? windowsToWslPath(value) : value;
  const optionalRuntimePath = (value?: string): string | null =>
    value?.trim() ? runtimePath(value.trim()) : null;
  const usesSegmentation = draft.inference.mode !== "face";
  const usesFaces = draft.inference.mode !== "segmentation";
  const inference: Record<string, unknown> = {
    enabled: draft.inference.enabled,
    mode: draft.inference.mode,
    device: draft.inference.device,
    max_frames: draft.inference.maxFrames,
    warmup_frames: draft.inference.warmupFrames,
    face_warmup_iterations: draft.inference.faceWarmupIterations,
    parallel_models: draft.inference.parallelModels,
    parallel_model_stagger_seconds:
      draft.inference.parallelModelStaggerSeconds,
    fast_sqlite: draft.inference.fastSqlite,
    extra_args: draft.inference.extraArgs,
  };
  if (usesSegmentation) {
    inference.segmentation_model = draft.inference.segmentationModel;
    inference.segmentation_backend = draft.inference.segmentationBackend;
  }
  if (usesFaces) {
    inference.face_model = draft.inference.faceModel;
    inference.face_backend = draft.inference.faceBackend;
    inference.face_classes = draft.inference.faceClasses;
    inference.face_trt_bundle = optionalRuntimePath(
      draft.inference.faceTrtBundle,
    );
  }
  if (!draft.inference.enabled) {
    inference.input_sqlite = optionalRuntimePath(draft.inference.inputSqlite);
  }

  const postprocess: Record<string, unknown> = {
    enabled: draft.postprocess.enabled,
    shape_mode: draft.postprocess.shapeMode,
    pipeline_config: optionalRuntimePath(draft.postprocess.pipelineConfig),
    class_policy_json: optionalRuntimePath(draft.postprocess.classPolicyJson),
    class_postprocess_policy_json:
      draft.postprocess.classPostprocessPolicySource === "global"
        ? null
        : optionalRuntimePath(draft.postprocess.classPostprocessPolicyJson),
    score_min: draft.postprocess.scoreMin,
    cut_detect: draft.postprocess.cutDetect,
    cut_method: draft.postprocess.cutMethod,
    precompute_cuts_during_inference:
      draft.postprocess.precomputeCutsDuringInference,
    remove_short_tracks_max_frames:
      draft.postprocess.removeShortTracksMaxFrames,
    keyframe_interval: draft.postprocess.keyframeInterval,
    max_gap: draft.postprocess.maxGap,
    model_root: optionalRuntimePath(draft.postprocess.modelRoot),
    k2_run_dir: optionalRuntimePath(draft.postprocess.k2RunDir),
    k2_batch_size: draft.postprocess.k2BatchSize,
    k2_prep_workers: draft.postprocess.k2PrepWorkers,
    k2_precision: draft.postprocess.k2Precision,
    k2_forward_mode: draft.postprocess.k2ForwardMode,
    k2_profile_stages: draft.postprocess.k2ProfileStages,
    k2_cudnn_benchmark: draft.postprocess.k2CudnnBenchmark,
    k2_tf32: draft.postprocess.k2Tf32,
    device: draft.postprocess.device,
    export_legacy_sqlite: draft.postprocess.exportLegacySqlite,
    face_mask_target: draft.postprocess.faceMaskTarget,
    eye_mask_shape: draft.postprocess.eyeMaskShape,
    minimum_eye_confidence: draft.postprocess.minimumEyeConfidence,
    face_tracking_max_gap_frames:
      draft.postprocess.faceTrackingMaxGapFrames,
    face_tracking_high_score_threshold:
      draft.postprocess.faceTrackingHighScoreThreshold,
    face_tracking_low_score_threshold:
      draft.postprocess.faceTrackingLowScoreThreshold,
    face_short_track_max_hits: draft.postprocess.faceShortTrackMaxHits,
    face_short_track_keep_score: draft.postprocess.faceShortTrackKeepScore,
    face_interpolation_max_gap:
      draft.postprocess.faceInterpolationMaxGap,
    extra_args: draft.postprocess.extraArgs,
  };
  if (!draft.postprocess.enabled) {
    postprocess.tracked_sqlite = optionalRuntimePath(
      draft.postprocess.trackedSqlite,
    );
    postprocess.final_sqlite = optionalRuntimePath(
      draft.postprocess.finalSqlite,
    );
  }

  return {
    schema_version: 1,
    input_video: runtimePath(draft.inputVideo),
    output_root: runtimePath(draft.outputRoot),
    execution: {
      runtime_python: runtimePath(settings.runtimePython),
      resume: draft.execution.resume,
    },
    inference,
    postprocess,
    overlay: {
      enabled: draft.overlay.enabled,
      execution_mode: draft.overlay.executionMode,
      raw: draft.overlay.raw,
      tracked: draft.overlay.tracked,
      final: draft.overlay.final,
      faces: draft.overlay.faces,
      final_include_faces: draft.overlay.finalIncludeFaces,
      presets: draft.overlay.presets,
      genital_source: draft.overlay.genitalSource,
      face_mask_target: draft.overlay.faceMaskTarget,
      eye_mask_shape: draft.overlay.eyeMaskShape,
      minimum_eye_confidence: draft.overlay.minimumEyeConfidence,
      face_probability_masks: draft.overlay.faceProbabilityMasks,
      face_keypoints: draft.overlay.faceKeypoints,
      face_ellipses: draft.overlay.faceEllipses,
      mask_alpha: draft.overlay.maskAlpha,
      outline_thickness: draft.overlay.outlineThickness,
      box_thickness: draft.overlay.boxThickness,
      show_labels: draft.overlay.showLabels,
      codec: draft.overlay.codec,
      h264_crf: draft.overlay.h264Crf,
      h264_preset: draft.overlay.h264Preset,
      ffmpeg_bin: optionalRuntimePath(draft.overlay.ffmpegBin),
      nvenc_cq: draft.overlay.nvencCq,
      workers: draft.overlay.workers,
      cpu_workers: draft.overlay.cpuWorkers,
      copy_audio: draft.overlay.copyAudio,
      target_bitrate_mbps: draft.overlay.targetBitrateMbps,
      nvenc_preset: draft.overlay.nvencPreset,
      nvenc_gpu: draft.overlay.nvencGpu,
      faststart: draft.overlay.faststart,
      start_frame: draft.overlay.startFrame,
      end_frame: draft.overlay.endFrame,
      progress_every: draft.overlay.progressEvery,
      extra_args: draft.overlay.extraArgs,
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
