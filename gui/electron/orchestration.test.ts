import { describe, expect, it } from "vitest";
import type { AppSettings, PipelineDraft } from "../shared/types";
import {
  buildClassPostprocessPolicy,
  buildLaunchSpec,
  buildOrchestrationConfig,
  windowsToWslPath,
} from "./orchestration";
import { defaultDraft } from "../src/lib/defaults";

const settings: AppSettings = {
  backendMode: "native",
  backendRoot: "/opt/inference_backend2",
  runtimePython: "/opt/runtime/bin/python",
  wslDistro: "Ubuntu",
};

const draft: PipelineDraft = {
  ...structuredClone(defaultDraft),
  inputVideo: "/video/input.mp4",
  outputRoot: "/runs/job-1",
  inference: { ...defaultDraft.inference, maxFrames: 300 },
};

describe("orchestration bridge", () => {
  it("builds the repository schema without GUI-only fields", () => {
    const config = buildOrchestrationConfig(draft, settings);
    expect(config.schema_version).toBe(1);
    expect(config.inference.segmentation_model).toBe("dinov3_codino");
    expect(config.inference.face_model).toBe("face_dino_v2");
    expect(config.inference.face_backend).toBe("tensorrt-fast");
    expect(config.postprocess.k2_batch_size).toBe(128);
    expect(config.postprocess.face_tracking_max_gap_frames).toBe(5);
    expect(config.postprocess.face_detection_score_threshold).toBe(0.55);
    expect(config.postprocess.head_detection_score_threshold).toBe(0.55);
    expect(config.overlay.execution_mode).toBe("fast");
    expect(config.overlay.presets).toEqual([
      "combined-simple",
      "combined-detailed",
    ]);
    expect(config.overlay.workers).toBe(6);
    expect(config.overlay.face_mask_target).toBe("eyes");
  });

  it("omits segmentation fields for a face-only workflow", () => {
    const config = buildOrchestrationConfig(
      {
        ...draft,
        inference: {
          ...draft.inference,
          mode: "face",
          parallelModels: false,
        },
        postprocess: {
          ...draft.postprocess,
          enabled: false,
        },
      },
      settings,
    );
    expect(config.inference.segmentation_model).toBeUndefined();
    expect(config.inference.segmentation_backend).toBeUndefined();
    expect(config.inference.face_model).toBe("face_dino_v2");
    expect(config.inference.face_backend).toBe("tensorrt-fast");
  });

  it("builds a native dry-run command", () => {
    const launch = buildLaunchSpec(settings, "/tmp/job.json", true);
    expect(launch.executable).toBe("/opt/runtime/bin/python");
    expect(launch.cwd).toBe("/opt/inference_backend2");
    expect(launch.args).toEqual([
      "-m",
      "orchestration",
      "--config",
      "/tmp/job.json",
      "--dry-run",
    ]);
  });

  it("maps Windows paths and builds a WSL command", () => {
    expect(windowsToWslPath("C:\\Users\\Editor\\job.json")).toBe(
      "/mnt/c/Users/Editor/job.json",
    );
    expect(
      windowsToWslPath(
        "\\\\wsl.localhost\\Ubuntu-24.04\\home\\kenshin\\input.mp4",
      ),
    ).toBe("/home/kenshin/input.mp4");
    const launch = buildLaunchSpec(
      {
        ...settings,
        backendMode: "wsl",
        backendRoot: "/home/editor/inference_backend2",
      },
      "C:\\Users\\Editor\\job.json",
      false,
    );
    expect(launch.executable).toBe("wsl.exe");
    expect(launch.args).toContain("/mnt/c/Users/Editor/job.json");
  });

  it("maps path-bearing advanced settings into WSL", () => {
    const config = buildOrchestrationConfig(
      {
        ...draft,
        inference: {
          ...draft.inference,
          faceTrtBundle: "C:\\models\\face.bundle",
        },
        postprocess: {
          ...draft.postprocess,
          classPostprocessPolicySource: "file",
          classPostprocessPolicyJson: "C:\\jobs\\classes.json",
          modelRoot: "C:\\models\\k2",
        },
        overlay: {
          ...draft.overlay,
          ffmpegBin: "C:\\tools\\ffmpeg.exe",
        },
      },
      { ...settings, backendMode: "wsl" },
    );
    expect(config.inference.face_trt_bundle).toBe(
      "/mnt/c/models/face.bundle",
    );
    expect(config.postprocess.class_postprocess_policy_json).toBe(
      "/mnt/c/jobs/classes.json",
    );
    expect(config.postprocess.model_root).toBe("/mnt/c/models/k2");
    expect(config.overlay.ffmpeg_bin).toBe("/mnt/c/tools/ffmpeg.exe");
  });

  it("builds a classwise policy from GUI rows", () => {
    const policy = buildClassPostprocessPolicy({
      ...draft,
      postprocess: {
        ...draft.postprocess,
        shapeMode: "polygon",
        keyframeInterval: 3,
        maxGap: 0,
        classPostprocessPolicySource: "editor",
        classPostprocessRules: [
          {
            className: "男性器",
            shapeMode: "ellipse",
            keyframeInterval: 2,
            maxGap: 30,
          },
          {
            className: "女性器",
            shapeMode: "polygon",
            keyframeInterval: 3,
            maxGap: 12,
          },
        ],
      },
    });
    expect(policy).toEqual({
      schema_version: 1,
      default: {
        shape_mode: "polygon",
        keyframe_interval: 3,
        max_gap: 0,
      },
      classes: {
        男性器: {
          shape_mode: "ellipse",
          keyframe_interval: 2,
          max_gap: 30,
        },
        女性器: {
          shape_mode: "polygon",
          keyframe_interval: 3,
          max_gap: 12,
        },
      },
    });
  });

  it("rejects duplicate GUI class names", () => {
    expect(() =>
      buildClassPostprocessPolicy({
        ...draft,
        postprocess: {
          ...draft.postprocess,
          maxGap: 0,
          classPostprocessPolicySource: "editor",
          classPostprocessRules: [
            {
              className: "男性器",
              shapeMode: "ellipse",
              keyframeInterval: 2,
              maxGap: 30,
            },
            {
              className: " 男性器 ",
              shapeMode: "polygon",
              keyframeInterval: 3,
              maxGap: 0,
            },
          ],
        },
      }),
    ).toThrow("重複");
  });

  it("requires a path when classwise JSON mode is selected", () => {
    expect(() =>
      buildOrchestrationConfig(
        {
          ...draft,
          postprocess: {
            ...draft.postprocess,
            classPostprocessPolicySource: "file",
            classPostprocessPolicyJson: "",
          },
        },
        settings,
      ),
    ).toThrow("形状設定JSON");
  });
});
