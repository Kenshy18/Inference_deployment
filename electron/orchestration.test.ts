import { describe, expect, it } from "vitest";
import type { AppSettings, PipelineDraft } from "../shared/types";
import {
  buildLaunchSpec,
  buildOrchestrationConfig,
  windowsToWslPath,
} from "./orchestration";

const settings: AppSettings = {
  backendMode: "native",
  backendRoot: "/opt/inference_backend2",
  runtimePython: "/opt/runtime/bin/python",
  wslDistro: "Ubuntu",
};

const draft: PipelineDraft = {
  inputVideo: "/video/input.mp4",
  outputRoot: "/runs/job-1",
  inference: {
    enabled: true,
    inputSqlite: "",
    mode: "segmentation-face",
    segmentationModel: "dinov3_codino",
    segmentationBackend: "tensorrt-fast",
    faceModel: "rtdetr_head_face",
    faceClasses: ["Face", "Head"],
    device: "cuda:0",
    maxFrames: 300,
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

describe("orchestration bridge", () => {
  it("builds the repository schema without GUI-only fields", () => {
    const config = buildOrchestrationConfig(draft, settings);
    expect(config.schema_version).toBe(1);
    expect(config.inference.segmentation_model).toBe("dinov3_codino");
    expect(config.postprocess.device).toBe("cpu");
    expect(config.overlay.final_include_faces).toBe(true);
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
});
