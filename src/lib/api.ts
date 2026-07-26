import type { AppSettings, JobSnapshot, MaskStudioApi } from "../../shared/types";
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
      processedFrames: 298,
      totalFrames: 482,
      fps: 20.156,
      computeFps: 20.16,
      detections: 1_468,
      masks: 265,
      faces: 812,
      elapsedSeconds: 252,
      progress: 298 / 482,
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

const previewApi: MaskStudioApi = {
  bootstrap: async () => ({
    platform: "linux",
    settings: browserSettings,
    job: mockJob(),
  }),
  pickFile: async () => null,
  pickDirectory: async () => null,
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
    status: "failed",
    error: "ブラウザプレビューではジョブを実行できません。",
  }),
  cancelWorkflow: async () => emptyJob,
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
    }, 320);
    return () => window.clearInterval(timer);
  },
};

export const desktopApi = window.maskStudio ?? previewApi;
export const isElectron = window.maskStudio !== undefined;
