import type {
  AppSettings,
  HardwareMetrics,
  JobSnapshot,
  LivePreviewFrame,
  MaskStudioApi,
} from "../../shared/types";
import { browserSettings, emptyJob } from "./defaults";

/** Browser preview (`npm run dev:renderer`) has no Electron bridge.
 *  `?mock=running|done|failed` seeds a snapshot so the interface can be
 *  reviewed without a GPU run. */
function mockJob(): JobSnapshot {
  const kind = new URLSearchParams(window.location.search).get("mock");
  if (!kind) {
    return emptyJob;
  }
  const logs = [
    "$ /opt/runtime/bin/python -m orchestration --config /tmp/job/orchestration.json",
    "[inference] loading dinov3_codino tensorrt-fast engines",
    "[inference] measured compute throughput: 20.160 img/s",
    "[inference] [progress] processed=116/482 detections=79 fps=15.630",
    "[inference] [progress] processed=298/482 detections=1468 fps=20.156",
    "[inference] [orchestrator] frames=298 detections=1468 classifications=265 segmentations=265",
    "[postprocess] cut detection: high_precision, 4 cuts",
    "[postprocess] tracking 265 masks over 298 frames",
  ];
  const base: JobSnapshot = {
    ...emptyJob,
    id: "2026-07-26T13-04-11-882Z",
    status: "running",
    stage: "postprocess",
    startedAt: new Date(Date.now() - 252_000).toISOString(),
    outputRoot: "/home/kenshin/runs/job-041",
    logs,
    telemetry: {
      ...emptyJob.telemetry,
      processedFrames: 298,
      totalFrames: 482,
      fps: 20.156,
      computeFps: 20.16,
      detections: 1_468,
      masks: 265,
      faces: 812,
      elapsedSeconds: 252,
      progress: 298 / 482,
      phases: {
        segmentation_inference: {
          state: "complete",
          completed: 482,
          total: 482,
          progress: 1,
          estimated: false,
          detail: "complete",
          fps: 20.16,
          activeElapsedSeconds: null,
          updatedAtMs: Date.now(),
        },
        face_inference: {
          state: "complete",
          completed: 482,
          total: 482,
          progress: 1,
          estimated: false,
          detail: "complete",
          fps: 25.4,
          activeElapsedSeconds: null,
          updatedAtMs: Date.now(),
        },
        postprocess: {
          state: "running",
          completed: 14,
          total: 30,
          progress: 14 / 30,
          estimated: true,
          detail: "tracking:output-validation",
          fps: null,
          activeElapsedSeconds: 18.4,
          updatedAtMs: Date.now(),
        },
        overlay: {
          state: "pending",
          completed: 0,
          total: null,
          progress: null,
          estimated: false,
          detail: "",
          fps: null,
          activeElapsedSeconds: null,
          updatedAtMs: null,
        },
      },
    },
  };
  if (kind === "done") {
    return {
      ...base,
      status: "completed",
      stage: "overlay_final",
      exitCode: 0,
      completedAt: new Date().toISOString(),
      artifacts: {
        inference_sqlite:
          "/home/kenshin/runs/job-041/01_inference/inference.sqlite",
        tracked_sqlite:
          "/home/kenshin/runs/job-041/02_postprocess/tracked.sqlite",
        predictions_sqlite:
          "/home/kenshin/runs/job-041/02_postprocess/predictions.sqlite",
        overlay_raw: "/home/kenshin/runs/job-041/03_overlay/raw.mp4",
        overlay_tracked: "/home/kenshin/runs/job-041/03_overlay/tracked.mp4",
        overlay_final: "/home/kenshin/runs/job-041/03_overlay/final.mp4",
      },
    };
  }
  if (kind === "failed") {
    return {
      ...base,
      status: "failed",
      exitCode: 2,
      completedAt: new Date().toISOString(),
      error: "postprocess failed with exit code 1",
      logs: [
        ...logs,
        "[postprocess] RuntimeError: keyframe interval must be >= 1",
      ],
    };
  }
  return base;
}

/* Deterministic fake probe so the queue can be exercised without Electron. */
let mockPickCount = 0;

function mockThumbnail(seed: number): string {
  const hue = (seed * 47) % 360;
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='192' height='108'>` +
    `<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>` +
    `<stop offset='0' stop-color='hsl(${hue},45%,30%)'/>` +
    `<stop offset='1' stop-color='hsl(${(hue + 60) % 360},40%,14%)'/>` +
    `</linearGradient></defs>` +
    `<rect width='192' height='108' fill='url(%23g)'/>` +
    `<circle cx='96' cy='54' r='17' fill='rgba(255,255,255,0.25)'/>` +
    `<path d='M90 45 106 54 90 63Z' fill='white'/></svg>`;
  return `data:image/svg+xml;charset=utf-8,${svg}`;
}

const previewApi: MaskStudioApi = {
  bootstrap: async () => ({
    platform: "linux",
    settings: browserSettings,
    job: mockJob(),
  }),
  pickFile: async () => null,
  pickVideos: async () => {
    mockPickCount += 1;
    return [`/mock/videos/sample-${String(mockPickCount).padStart(2, "0")}.mp4`];
  },
  pickDirectory: async () => null,
  probeVideo: async (path: string) => {
    const seed = [...path].reduce((sum, char) => sum + char.charCodeAt(0), 0);
    const durationSeconds = 45 + (seed % 600);
    const fps = 30;
    return {
      durationSeconds,
      width: 1920,
      height: 1080,
      fps,
      frameCount: Math.round(durationSeconds * fps),
      thumbnail: mockThumbnail(seed),
    };
  },
  pathForFile: () => null,
  saveSettings: async (settings: AppSettings) => settings,
  validateWorkflow: async () => ({
    ...emptyJob,
    status: "validated",
    dryRun: true,
    completedAt: new Date().toISOString(),
    logs: ["ブラウザプレビューでは dry-run を実行しません。"],
  }),
  startWorkflow: async () => ({
    ...emptyJob,
    id: `preview-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    status: "failed",
    error: "ブラウザプレビューではジョブを実行できません。",
  }),
  cancelWorkflow: async () => emptyJob,
  setPreviewEnabled: async () => undefined,
  sampleHardware: async (): Promise<HardwareMetrics> => {
    const seconds = Date.now() / 1_000;
    return {
      timestamp: Date.now(),
      cpuPercent: 42 + Math.sin(seconds / 3.1) * 18,
      memoryPercent: 58 + Math.sin(seconds / 8.7) * 3,
      memoryUsedBytes: 37.1 * 1024 ** 3,
      memoryTotalBytes: 64 * 1024 ** 3,
      gpuPercent: 73 + Math.sin(seconds / 2.3) * 20,
      vramPercent: 64 + Math.sin(seconds / 5.2) * 6,
      vramUsedMiB: 15_728,
      vramTotalMiB: 24_576,
      gpuTemperatureC: 66 + Math.sin(seconds / 6.2) * 4,
    };
  },
  openOutput: async () => "",
  /* `?mock=running` also streams synthetic progress so the live monitor,
     console and throughput scope can be reviewed in the browser. */
  onJobUpdate: (callback: (job: JobSnapshot) => void) => {
    if (new URLSearchParams(window.location.search).get("mock") !== "running") {
      return () => undefined;
    }
    let snapshot = mockJob();
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      const processedFrames = Math.min(482, 298 + step * 6);
      const fps = 19.4 + Math.sin(step / 2.6) * 2.3 + (step % 7 === 0 ? -1.6 : 0);
      snapshot = {
        ...snapshot,
        telemetry: {
          ...snapshot.telemetry,
          processedFrames,
          fps,
          elapsedSeconds: 252 + step * 0.4,
          progress: processedFrames / 482,
        },
        logs: [
          ...snapshot.logs,
          `[postprocess] [progress] processed=${processedFrames}/482 detections=1468 fps=${fps.toFixed(3)}`,
        ].slice(-400),
      };
      callback(snapshot);
    }, 100);
    return () => window.clearInterval(timer);
  },
  onPreviewUpdate: (callback: (frame: LivePreviewFrame) => void) => {
    if (new URLSearchParams(window.location.search).get("mock") !== "running") {
      return () => undefined;
    }
    let frameIndex = 0;
    const timer = window.setInterval(() => {
      frameIndex += 5;
      callback({
        jobId: "2026-07-26T13-04-11-882Z",
        dataUrl: mockThumbnail(frameIndex),
        phase: frameIndex % 20 === 0 ? "face_inference" : "segmentation_inference",
        frameIndex,
        timestampSeconds: frameIndex / 30,
        model: "preview-model",
        width: 960,
        height: 540,
        generatedAtMs: Date.now(),
        dropped: 0,
      });
    }, 200);
    return () => window.clearInterval(timer);
  },
};

export const desktopApi = window.maskStudio ?? previewApi;
export const isElectron = window.maskStudio !== undefined;
