import fs from "node:fs";
import path from "node:path";
import type { BrowserWindow } from "electron";

export interface QaE2eOptions {
  input: string;
  output: string;
  report: string;
  maxFrames: number;
  cancelAfterMs: number | null;
}

interface QaReport {
  status?: string;
  finalJob?: { status?: string; exitCode?: number | null; error?: string | null };
  preview?: { count?: number };
  [key: string]: unknown;
}

function waitForLoad(window: BrowserWindow): Promise<void> {
  if (!window.webContents.isLoadingMainFrame()) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    window.webContents.once("did-finish-load", () => resolve());
  });
}

export async function runQaE2e(
  window: BrowserWindow,
  options: QaE2eOptions,
): Promise<boolean> {
  await waitForLoad(window);
  fs.mkdirSync(path.dirname(options.report), { recursive: true });
  fs.mkdirSync(options.output, { recursive: true });
  const rendererErrors: string[] = [];
  window.webContents.on("console-message", (_event, level, message) => {
    if (level >= 3) {
      rendererErrors.push(message);
    }
  });

  const payload = JSON.stringify(options);
  let report: QaReport;
  try {
    report = (await window.webContents.executeJavaScript(`
      (async () => {
        const options = ${payload};
        const terminal = new Set(["validated", "completed", "failed", "cancelled"]);
        const startedAt = Date.now();
        while (!localStorage.getItem("mask-studio-draft")) {
          await new Promise((resolve) => setTimeout(resolve, 50));
        }
        const draft = JSON.parse(localStorage.getItem("mask-studio-draft"));
        draft.inputVideo = options.input;
        draft.outputRoot = options.output;
        draft.execution.resume = false;
        Object.assign(draft.inference, {
          enabled: true,
          mode: "segmentation-face",
          segmentationModel: "dinov3_codino_mh0",
          segmentationBackend: "tensorrt-fast",
          faceModel: "face_dino_v2",
          faceBackend: "tensorrt-fast",
          device: "cuda:0",
          maxFrames: options.maxFrames,
          parallelModels: true,
          parallelModelStaggerSeconds: 0,
          fastSqlite: true,
        });
        Object.assign(draft.postprocess, {
          enabled: true,
          classPostprocessPolicySource: "editor",
          shapeMode: "polygon",
          cutDetect: true,
          precomputeCutsDuringInference: true,
          device: "cuda:0",
          faceMaskTarget: "eyes",
          eyeMaskShape: "rectangle",
        });
        Object.assign(draft.overlay, {
          enabled: true,
          executionMode: "fast",
          raw: false,
          tracked: false,
          final: false,
          faces: false,
          finalIncludeFaces: false,
          presets: ["combined-simple"],
          workers: 6,
          cpuWorkers: 0,
          faceMaskTarget: "eyes",
          eyeMaskShape: "rectangle",
        });
        localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
        localStorage.setItem("mask-studio-draft-version", "4");

        const bootstrap = await window.maskStudio.bootstrap();
        const settings = {
          ...bootstrap.settings,
          backendMode: "wsl",
          // Deployment QA runs against a versioned distribution name. Keep
          // the backend coordinates seeded by the deployer instead of
          // silently falling back to the development distribution.
          wslDistro: bootstrap.settings.wslDistro,
          backendRoot: bootstrap.settings.backendRoot,
          runtimePython: bootstrap.settings.runtimePython,
        };
        await window.maskStudio.saveSettings(settings);
        await window.maskStudio.setPreviewEnabled(true);

        const updates = [];
        const progressStates = {
          segmentation_inference: new Set(),
          face_inference: new Set(),
          postprocess: new Set(),
          overlay: new Set(),
        };
        const preview = { count: 0, phases: {}, frames: [], droppedMax: 0 };
        const hardware = [];
        let lastJob = bootstrap.job;
        const stopJob = window.maskStudio.onJobUpdate((job) => {
          lastJob = job;
          updates.push({
            at: Date.now(),
            id: job.id,
            dryRun: job.dryRun,
            status: job.status,
            stage: job.stage,
            processedFrames: job.telemetry.processedFrames,
            progress: job.telemetry.progress,
            phases: structuredClone(job.telemetry.phases),
          });
          for (const [name, value] of Object.entries(job.telemetry.phases)) {
            progressStates[name].add(value.state);
          }
        });
        const stopPreview = window.maskStudio.onPreviewUpdate((frame) => {
          preview.count += 1;
          preview.phases[frame.phase] = (preview.phases[frame.phase] ?? 0) + 1;
          preview.droppedMax = Math.max(preview.droppedMax, frame.dropped ?? 0);
          if (preview.frames.length < 20) {
            preview.frames.push({
              phase: frame.phase,
              frameIndex: frame.frameIndex,
              width: frame.width,
              height: frame.height,
              hasJpeg: frame.dataUrl.startsWith("data:image/jpeg;base64,"),
            });
          }
        });
        const heartbeat = [];
        let heartbeatLast = performance.now();
        const heartbeatTimer = setInterval(() => {
          const now = performance.now();
          heartbeat.push(now - heartbeatLast);
          heartbeatLast = now;
        }, 100);
        const hardwareTimer = setInterval(async () => {
          try {
            hardware.push(await window.maskStudio.sampleHardware());
          } catch {}
        }, 1000);

        const waitForTerminal = async (id, timeoutMs) => {
          const deadline = Date.now() + timeoutMs;
          while (Date.now() < deadline) {
            const job = (await window.maskStudio.bootstrap()).job;
            if (job.id === id && terminal.has(job.status)) return job;
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
          throw new Error("workflow timeout: " + id);
        };

        let dryRun;
        let finalJob;
        let cancellationRequestedAt = null;
        try {
          const dryStart = await window.maskStudio.validateWorkflow(draft, settings);
          dryRun = await waitForTerminal(dryStart.id, 60000);
          if (dryRun.status !== "validated" || dryRun.exitCode !== 0) {
            throw new Error("Dry Run failed: " + JSON.stringify(dryRun));
          }
          const runStart = await window.maskStudio.startWorkflow(draft, settings);
          if (options.cancelAfterMs !== null) {
            await new Promise((resolve) => setTimeout(resolve, options.cancelAfterMs));
            cancellationRequestedAt = Date.now();
            await window.maskStudio.cancelWorkflow();
          }
          finalJob = await waitForTerminal(runStart.id, 15 * 60 * 1000);
        } finally {
          clearInterval(heartbeatTimer);
          clearInterval(hardwareTimer);
          stopJob();
          stopPreview();
          await window.maskStudio.setPreviewEnabled(false);
        }
        return {
          schemaVersion: 1,
          status: options.cancelAfterMs === null
            ? (finalJob?.status === "completed" && finalJob?.exitCode === 0
              ? "passed" : "failed")
            : (finalJob?.status === "cancelled" ? "passed" : "failed"),
          startedAt,
          elapsedSeconds: (Date.now() - startedAt) / 1000,
          input: options.input,
          output: options.output,
          maxFrames: options.maxFrames,
          cancelAfterMs: options.cancelAfterMs,
          cancellationRequestedAt,
          dryRun: {
            status: dryRun?.status,
            exitCode: dryRun?.exitCode,
            error: dryRun?.error,
          },
          finalJob,
          updates,
          progressStates: Object.fromEntries(
            Object.entries(progressStates).map(([name, states]) => [name, [...states]]),
          ),
          preview,
          hardware,
          heartbeat: {
            count: heartbeat.length,
            maxMs: heartbeat.length ? Math.max(...heartbeat) : null,
            over500Ms: heartbeat.filter((value) => value > 500).length,
          },
          lastJob,
        };
      })()
    `, true)) as QaReport;
  } catch (error) {
    report = {
      schemaVersion: 1,
      status: "harness_error",
      error: error instanceof Error ? error.stack ?? error.message : String(error),
    };
  }
  report.rendererErrors = rendererErrors;
  try {
    const image = await window.webContents.capturePage();
    const screenshot = options.report.replace(/\.json$/i, ".png");
    fs.writeFileSync(screenshot, image.toPNG());
    report.screenshot = screenshot;
  } catch (error) {
    report.screenshotError = error instanceof Error ? error.message : String(error);
  }
  fs.writeFileSync(options.report, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  const phases = [
    "segmentation_inference",
    "face_inference",
    "postprocess",
    "overlay",
  ];
  const progressStates = report.progressStates as Record<string, string[]> | undefined;
  const allPhasesComplete = phases.every((phase) =>
    progressStates?.[phase]?.includes("complete"),
  );
  const previewCount = report.preview?.count ?? 0;
  if (options.cancelAfterMs !== null) {
    return (
      report.status === "passed" &&
      report.finalJob?.status === "cancelled" &&
      rendererErrors.length === 0
    );
  }
  return (
    report.status === "passed" &&
    report.finalJob?.status === "completed" &&
    report.finalJob?.exitCode === 0 &&
    allPhasesComplete &&
    previewCount > 0 &&
    rendererErrors.length === 0
  );
}
