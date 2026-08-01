import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { _electron as electron } from "playwright";

const guiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(guiRoot, "..");
const matrixRoot = path.join(repositoryRoot, "output", "gui_matrix_20260801");
const runRoot = path.join(matrixRoot, "runs");
const artifactRoot = path.join(matrixRoot, "artifacts");
const runtimeLibraries = path.join(
  guiRoot,
  ".runtime-libs",
  "usr",
  "lib",
  "x86_64-linux-gnu",
);

fs.mkdirSync(runRoot, { recursive: true });
fs.mkdirSync(artifactRoot, { recursive: true });

const data = (...parts) => path.join(repositoryRoot, "data", ...parts);
const fixture = (name) => path.join(matrixRoot, "fixtures", name);

const mixedRules = [
  { className: "男性器", shapeMode: "polygon", keyframeInterval: 2, maxGap: 8 },
  { className: "女性器", shapeMode: "ellipse", keyframeInterval: 3, maxGap: 12 },
  { className: "結合部分", shapeMode: "polygon", keyframeInterval: 4, maxGap: 6 },
];

const cases = [
  {
    id: "01_batch_live_v3lite_facev2",
    videos: [
      data("codino_trt_3min_simple150_input.mp4"),
      fixture("real_720p24_45s.mp4"),
    ],
    live: true,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation-face",
        segmentationModel: "dinov3_codino_mh0",
        segmentationBackend: "tensorrt-fast",
        faceModel: "face_dino_v2",
        faceBackend: "tensorrt-fast",
        maxFrames: 900,
        parallelModels: true,
        parallelModelStaggerSeconds: 0,
        fastSqlite: true,
      },
      postprocess: {
        enabled: true,
        shapeMode: "polygon",
        classPostprocessRules: mixedRules,
        faceMaskTarget: "eyes",
        eyeMaskShape: "rectangle",
        cutDetect: true,
        precomputeCutsDuringInference: true,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["combined-simple", "combined-detailed"],
        faceMaskTarget: "eyes",
        eyeMaskShape: "rectangle",
        workers: 6,
        cpuWorkers: 0,
      },
    },
  },
  {
    id: "02_v3_facev2_mixed_720p",
    videos: [fixture("real_720p24_45s.mp4")],
    live: false,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation-face",
        segmentationModel: "dinov3_codino",
        segmentationBackend: "tensorrt-fast",
        faceModel: "face_dino_v2",
        faceBackend: "tensorrt-fast",
        maxFrames: 1080,
        parallelModels: false,
        fastSqlite: false,
      },
      postprocess: {
        enabled: true,
        shapeMode: "polygon",
        classPostprocessRules: mixedRules,
        faceMaskTarget: "eyes",
        eyeMaskShape: "ellipse",
        cutDetect: true,
        precomputeCutsDuringInference: true,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["combined-detailed"],
        faceMaskTarget: "eyes",
        eyeMaskShape: "ellipse",
      },
    },
  },
  {
    id: "03_v2_segmentation_portrait_mkv",
    videos: [fixture("real_portrait_720x1280_20s.mkv")],
    live: true,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation",
        segmentationModel: "dinov3_cascade",
        segmentationBackend: "tensorrt-backbone",
        maxFrames: 480,
        parallelModels: false,
        fastSqlite: false,
      },
      postprocess: {
        enabled: true,
        shapeMode: "polygon",
        classPostprocessRules: mixedRules.map((rule) => ({
          ...rule,
          shapeMode: "polygon",
        })),
        faceMaskTarget: "none",
        precomputeCutsDuringInference: true,
        device: "cpu",
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["genital-simple", "genital-detailed"],
        faceMaskTarget: "none",
      },
    },
  },
  {
    id: "04_v1_segmentation_vfr_nvenc",
    videos: [fixture("real_vfr_pts_gap_30s.mp4")],
    live: false,
    timeoutMs: 10 * 60_000,
    patch: {
      inference: {
        mode: "segmentation",
        segmentationModel: "eva02_cascade",
        segmentationBackend: "tensorrt-backbone",
        maxFrames: 720,
        parallelModels: false,
        fastSqlite: false,
      },
      postprocess: {
        enabled: true,
        shapeMode: "ellipse",
        classPostprocessRules: mixedRules.map((rule) => ({
          ...rule,
          shapeMode: "ellipse",
        })),
        faceMaskTarget: "none",
        precomputeCutsDuringInference: false,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "nvenc",
        presets: ["genital-detailed"],
        faceMaskTarget: "none",
      },
    },
  },
  {
    id: "05_facev1_only_nonzero_start",
    videos: [
      data(
        "新しいフォルダー",
        "HEYZO-3549 浜田希 はまたのそみ 激しめイラマか好き - 無修正アタルト動画 HEYZO -.mp4",
      ),
    ],
    live: true,
    timeoutMs: 10 * 60_000,
    patch: {
      inference: {
        mode: "face",
        faceModel: "rtdetr_head_face",
        faceBackend: "pytorch",
        maxFrames: 720,
        parallelModels: false,
        fastSqlite: false,
      },
      postprocess: {
        enabled: false,
        classPostprocessPolicySource: "global",
        faceMaskTarget: "none",
        precomputeCutsDuringInference: false,
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["face-simple", "face-detailed"],
        faceMaskTarget: "none",
      },
    },
  },
  {
    id: "06_v3lite_facev2_4k_noaudio",
    videos: [fixture("real_4k24_20s_noaudio.mp4")],
    live: false,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation-face",
        segmentationModel: "dinov3_codino_mh0",
        segmentationBackend: "tensorrt-fast",
        faceModel: "face_dino_v2",
        faceBackend: "tensorrt-fast",
        maxFrames: 480,
        parallelModels: false,
        fastSqlite: true,
      },
      postprocess: {
        enabled: true,
        shapeMode: "polygon",
        classPostprocessRules: mixedRules,
        faceMaskTarget: "face",
        eyeMaskShape: "ellipse",
        precomputeCutsDuringInference: true,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["combined-simple", "combined-detailed"],
        faceMaskTarget: "face",
        workers: 6,
        cpuWorkers: 0,
      },
    },
  },
  {
    id: "07_v3_facev2_ten_minute_live_stress",
    videos: [data("新しいフォルダー", "HEYZO-3545_30分-45分.mp4")],
    live: true,
    timeoutMs: 25 * 60_000,
    patch: {
      inference: {
        mode: "segmentation-face",
        segmentationModel: "dinov3_codino",
        segmentationBackend: "tensorrt-fast",
        faceModel: "face_dino_v2",
        faceBackend: "tensorrt-fast",
        maxFrames: 14_400,
        parallelModels: false,
        fastSqlite: true,
      },
      postprocess: {
        enabled: true,
        shapeMode: "polygon",
        classPostprocessRules: mixedRules,
        faceMaskTarget: "eyes",
        eyeMaskShape: "rectangle",
        precomputeCutsDuringInference: true,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["combined-detailed"],
        faceMaskTarget: "eyes",
        eyeMaskShape: "rectangle",
        workers: 6,
        cpuWorkers: 0,
      },
    },
  },
  {
    id: "08_facev1_positive_short",
    videos: [fixture("real_720p24_45s.mp4")],
    live: false,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "face",
        faceModel: "rtdetr_head_face",
        faceBackend: "pytorch",
        maxFrames: 480,
        parallelModels: false,
        fastSqlite: true,
      },
      postprocess: {
        enabled: false,
        classPostprocessPolicySource: "global",
        faceMaskTarget: "none",
        precomputeCutsDuringInference: false,
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["face-detailed"],
        faceMaskTarget: "none",
        workers: 6,
        cpuWorkers: 0,
      },
    },
  },
  {
    id: "09_v3lite_cpu_overlay",
    videos: [fixture("real_portrait_720x1280_20s.mkv")],
    live: false,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation",
        segmentationModel: "dinov3_codino_mh0",
        segmentationBackend: "tensorrt-fast",
        maxFrames: 240,
        fastSqlite: false,
      },
      postprocess: {
        enabled: true,
        shapeMode: "ellipse",
        classPostprocessRules: mixedRules.map((rule) => ({
          ...rule,
          shapeMode: "ellipse",
        })),
        faceMaskTarget: "none",
        precomputeCutsDuringInference: false,
        device: "cuda:0",
      },
      overlay: {
        enabled: true,
        executionMode: "cpu",
        presets: ["genital-simple"],
        faceMaskTarget: "none",
      },
    },
  },
  {
    id: "10_v3lite_raw_without_postprocess",
    videos: [fixture("real_720p24_45s.mp4")],
    live: false,
    timeoutMs: 8 * 60_000,
    patch: {
      inference: {
        mode: "segmentation",
        segmentationModel: "dinov3_codino_mh0",
        segmentationBackend: "tensorrt-fast",
        maxFrames: 240,
        fastSqlite: true,
      },
      postprocess: {
        enabled: false,
        classPostprocessPolicySource: "global",
        faceMaskTarget: "none",
        precomputeCutsDuringInference: false,
      },
      overlay: {
        enabled: true,
        executionMode: "fast",
        presets: ["genital-detailed"],
        genitalSource: "raw",
        faceMaskTarget: "none",
        workers: 6,
        cpuWorkers: 0,
      },
    },
  },
];

function merge(base, patch) {
  if (Array.isArray(patch)) {
    return patch.map((value) =>
      value && typeof value === "object" ? structuredClone(value) : value,
    );
  }
  if (patch === null || typeof patch !== "object") {
    return patch;
  }
  const output = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    output[key] =
      value && typeof value === "object" && !Array.isArray(value)
        ? merge(base?.[key] ?? {}, value)
        : merge(undefined, value);
  }
  return output;
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.floor(fraction * ordered.length))];
}

function summarizeNumbers(values) {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance =
    finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    finite.length;
  return {
    count: finite.length,
    min: Math.min(...finite),
    max: Math.max(...finite),
    mean,
    standardDeviation: Math.sqrt(variance),
    p50: percentile(finite, 0.5),
    p95: percentile(finite, 0.95),
    p99: percentile(finite, 0.99),
  };
}

function copyJobInputs(userData, destination) {
  const jobs = path.join(userData, "jobs");
  if (!fs.existsSync(jobs)) return [];
  fs.mkdirSync(destination, { recursive: true });
  const copied = [];
  for (const job of fs.readdirSync(jobs)) {
    const source = path.join(jobs, job);
    for (const name of ["orchestration.json", "class_postprocess_policy.json"]) {
      const file = path.join(source, name);
      if (!fs.existsSync(file)) continue;
      const target = path.join(destination, `${job}-${name}`);
      fs.copyFileSync(file, target);
      copied.push(target);
    }
  }
  return copied;
}

async function runCase(specification) {
  for (const video of specification.videos) {
    if (!fs.existsSync(video)) throw new Error(`missing input: ${video}`);
  }
  const caseOutput = path.join(runRoot, specification.id);
  const caseArtifacts = path.join(artifactRoot, specification.id);
  fs.mkdirSync(caseOutput, { recursive: true });
  fs.mkdirSync(caseArtifacts, { recursive: true });
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), `mask-studio-${specification.id}-`));
  const rendererErrors = [];
  const hardware = [];
  const jobSamples = [];
  const evaluateLatenciesMs = [];
  let app;
  const startedAt = Date.now();
  let timedOut = false;
  let dryRun = null;
  try {
    app = await electron.launch({
      args: [
        ".",
        "--software-rendering",
        ...specification.videos.map((video) => `--automation-video=${video}`),
        `--automation-output=${caseOutput}`,
        `--user-data-dir=${userData}`,
      ],
      cwd: guiRoot,
      env: {
        ...process.env,
        MASK_STUDIO_AUTOMATION_NO_EXTERNAL: "1",
        LD_LIBRARY_PATH: [runtimeLibraries, process.env.LD_LIBRARY_PATH]
          .filter(Boolean)
          .join(":"),
      },
    });
    const window = await app.firstWindow();
    window.on("pageerror", (error) => rendererErrors.push(`page: ${error.message}`));
    window.on("console", (message) => {
      if (message.type() === "error") rendererErrors.push(`console: ${message.text()}`);
    });
    await window.waitForLoadState("domcontentloaded");
    await window.waitForFunction(() => localStorage.getItem("mask-studio-draft") !== null);
    await window.evaluate(
      ({ patch, outputRoot }) => {
        const current = JSON.parse(localStorage.getItem("mask-studio-draft"));
        const mergeDraft = (base, values) => {
          if (Array.isArray(values)) return structuredClone(values);
          if (values === null || typeof values !== "object") return values;
          const result = { ...base };
          for (const [key, value] of Object.entries(values)) {
            result[key] =
              value && typeof value === "object" && !Array.isArray(value)
                ? mergeDraft(base?.[key] ?? {}, value)
                : structuredClone(value);
          }
          return result;
        };
        const next = mergeDraft(current, patch);
        next.outputRoot = outputRoot;
        localStorage.setItem("mask-studio-draft", JSON.stringify(next));
        localStorage.setItem("mask-studio-draft-version", "4");
        localStorage.setItem("mask-studio-queue", "[]");
        localStorage.setItem("mask-studio-settings-view", "advanced");
      },
      { patch: specification.patch, outputRoot: caseOutput },
    );
    await window.reload();
    await window.waitForLoadState("domcontentloaded");
    const sourcePanel = window.locator(".panel").filter({ hasText: "Source" });
    await sourcePanel.getByRole("button", { name: "参照" }).first().click();
    await sourcePanel.getByRole("button", { name: /追加/ }).click();
    await window.waitForFunction(
      (expected) => {
        const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
        return (
          queue.length === expected &&
          queue.every((item) =>
            [item.width, item.height, item.fps, item.frameCount].every(
              (value) => value !== null,
            ),
          )
        );
      },
      specification.videos.length,
      { timeout: 30_000 },
    );
    await window.screenshot({
      path: path.join(caseArtifacts, "01-ready.png"),
      fullPage: true,
    });

    await window.getByRole("button", { name: /Dry Run/ }).click();
    await window.waitForFunction(
      async () => (await window.maskStudio.bootstrap()).job.status === "validated",
      null,
      { timeout: 45_000 },
    );
    dryRun = await window.evaluate(async () => (await window.maskStudio.bootstrap()).job);

    await window.evaluate(() => {
      window.__qaHeartbeat = { last: performance.now(), deltas: [] };
      window.__qaHeartbeatTimer = setInterval(() => {
        const now = performance.now();
        window.__qaHeartbeat.deltas.push(now - window.__qaHeartbeat.last);
        window.__qaHeartbeat.last = now;
        if (window.__qaHeartbeat.deltas.length > 20_000) {
          window.__qaHeartbeat.deltas.splice(0, 10_000);
        }
      }, 100);
      window.__qaPreview = { count: 0, phases: {}, droppedMax: 0 };
      window.__qaPreviewStop = window.maskStudio.onPreviewUpdate((frame) => {
        window.__qaPreview.count += 1;
        window.__qaPreview.phases[frame.phase] =
          (window.__qaPreview.phases[frame.phase] ?? 0) + 1;
        window.__qaPreview.droppedMax = Math.max(
          window.__qaPreview.droppedMax,
          frame.dropped ?? 0,
        );
      });
    });

    // A completed Dry Run updates the main-process snapshot slightly before
    // React has necessarily committed the next enabled transport button.  A
    // role-only click could therefore land during that transition and leave
    // the queue pending without throwing.  Wait for the real primary button,
    // click it, and prove that a non-Dry-Run job was actually created before
    // collecting performance samples.
    const runButton = window.locator(".topbar .btn--primary");
    await runButton.waitFor({ state: "visible" });
    await window.waitForFunction(
      () => {
        const button = document.querySelector(".topbar .btn--primary");
        return button instanceof HTMLButtonElement && !button.disabled;
      },
      null,
      { timeout: 30_000 },
    );
    await runButton.click();
    let startedJob = null;
    const startDeadline = Date.now() + 30_000;
    while (Date.now() < startDeadline) {
      const candidate = await window.evaluate(async () =>
        (await window.maskStudio.bootstrap()).job,
      );
      if (candidate.id !== dryRun.id && !candidate.dryRun) {
        startedJob = candidate;
        break;
      }
      await window.waitForTimeout(100);
    }
    if (startedJob === null) {
      const diagnostics = await window.evaluate(() => ({
        primaryDisabled: document.querySelector(".topbar .btn--primary")?.disabled,
        queue: JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]"),
        body: document.body.innerText.slice(0, 4_000),
      }));
      throw new Error(
        `real workflow did not start after Dry Run: ${JSON.stringify(diagnostics)}`,
      );
    }
    if (specification.live) {
      await window.getByRole("tab", { name: "LIVE" }).click();
    }
    const deadline = Date.now() + specification.timeoutMs;
    while (Date.now() < deadline) {
      const evaluateStarted = Date.now();
      const snapshot = await window.evaluate(async () => {
        const bootstrap = await window.maskStudio.bootstrap();
        const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
        return {
          job: bootstrap.job,
          queue: queue.map((item) => ({
            title: item.title,
            status: item.status,
            outputDir: item.outputDir,
            error: item.error,
            width: item.width,
            height: item.height,
            fps: item.fps,
            frameCount: item.frameCount,
          })),
          scrollY: window.scrollY,
          bodyTextLength: document.body.innerText.length,
          appVisible: Boolean(document.querySelector(".app")),
        };
      });
      evaluateLatenciesMs.push(Date.now() - evaluateStarted);
      jobSamples.push({
        timestamp: Date.now(),
        status: snapshot.job.status,
        stage: snapshot.job.stage,
        fps: snapshot.job.telemetry.fps,
        processedFrames: snapshot.job.telemetry.processedFrames,
        phases: snapshot.job.telemetry.phases,
        scrollY: snapshot.scrollY,
        bodyTextLength: snapshot.bodyTextLength,
        appVisible: snapshot.appVisible,
      });
      try {
        hardware.push(
          await window.evaluate(async () => window.maskStudio.sampleHardware()),
        );
      } catch {
        // Metrics are diagnostic and must not change the workflow result.
      }
      const unfinished = snapshot.queue.some((item) =>
        ["pending", "processing"].includes(item.status),
      );
      if (!unfinished) break;
      await window.waitForTimeout(500);
    }
    const queue = await window.evaluate(() =>
      JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]"),
    );
    if (queue.some((item) => ["pending", "processing"].includes(item.status))) {
      timedOut = true;
      const stop = window.getByRole("button", { name: /^停止/ });
      if (await stop.isVisible()) await stop.click();
      await window.waitForTimeout(2_000);
    }
    const pageDiagnostics = await window.evaluate(() => {
      clearInterval(window.__qaHeartbeatTimer);
      window.__qaPreviewStop?.();
      return {
        heartbeatDeltasMs: window.__qaHeartbeat?.deltas ?? [],
        preview: window.__qaPreview ?? null,
        scrollY: window.scrollY,
        bodyTextLength: document.body.innerText.length,
        appVisible: Boolean(document.querySelector(".app")),
      };
    });
    await window.getByRole("tab", { name: "STATUS" }).click().catch(() => undefined);
    await window.screenshot({
      path: path.join(caseArtifacts, "02-finished.png"),
      fullPage: true,
    });
    const finalJob = await window.evaluate(async () => (await window.maskStudio.bootstrap()).job);
    const generatedInputs = copyJobInputs(userData, path.join(caseArtifacts, "job-inputs"));
    return {
      id: specification.id,
      status:
        timedOut
          ? "timed_out"
          : queue.every((item) => item.status === "done")
            ? "passed"
            : "failed",
      live: specification.live,
      videos: specification.videos,
      patch: specification.patch,
      elapsedSeconds: (Date.now() - startedAt) / 1000,
      dryRun: dryRun
        ? { status: dryRun.status, exitCode: dryRun.exitCode, error: dryRun.error }
        : null,
      finalJob: {
        status: finalJob.status,
        stage: finalJob.stage,
        exitCode: finalJob.exitCode,
        error: finalJob.error,
        outputRoot: finalJob.outputRoot,
        artifacts: finalJob.artifacts,
      },
      queue,
      rendererErrors,
      pageDiagnostics: {
        ...pageDiagnostics,
        heartbeat: summarizeNumbers(pageDiagnostics.heartbeatDeltasMs),
      },
      evaluateLatencyMs: summarizeNumbers(evaluateLatenciesMs),
      hardware: {
        samples: hardware.length,
        cpuPercent: summarizeNumbers(hardware.map((item) => item.cpuPercent)),
        memoryPercent: summarizeNumbers(hardware.map((item) => item.memoryPercent)),
        gpuPercent: summarizeNumbers(hardware.map((item) => item.gpuPercent)),
        vramPercent: summarizeNumbers(hardware.map((item) => item.vramPercent)),
        gpuTemperatureC: summarizeNumbers(
          hardware.map((item) => item.gpuTemperatureC),
        ),
      },
      telemetryFps: summarizeNumbers(
        jobSamples.map((sample) => sample.fps).filter((value) => value !== null),
      ),
      jobSamples,
      generatedInputs,
    };
  } catch (error) {
    copyJobInputs(userData, path.join(caseArtifacts, "job-inputs"));
    return {
      id: specification.id,
      status: "harness_error",
      live: specification.live,
      videos: specification.videos,
      elapsedSeconds: (Date.now() - startedAt) / 1000,
      error: error instanceof Error ? error.stack ?? error.message : String(error),
      rendererErrors,
      jobSamples,
      hardware,
    };
  } finally {
    await app?.close();
    fs.rmSync(userData, { recursive: true, force: true });
  }
}

const report = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  cases: [],
};
const selectedCase = process.env.GUI_MATRIX_CASE?.trim();
const requestedCases = new Set(
  selectedCase?.split(",").map((value) => value.trim()).filter(Boolean) ?? [],
);
const selectedCases = selectedCase
  ? cases.filter((specification) => requestedCases.has(specification.id))
  : cases;
if (selectedCases.length === 0) {
  throw new Error(`unknown GUI_MATRIX_CASE: ${selectedCase}`);
}
for (const specification of selectedCases) {
  console.log(`[gui-matrix] starting ${specification.id}`);
  const result = await runCase(specification);
  report.cases.push(result);
  fs.writeFileSync(
    path.join(artifactRoot, "gui-matrix-report.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  console.log(
    `[gui-matrix] ${specification.id}: ${result.status} ` +
      `${result.elapsedSeconds.toFixed(1)}s`,
  );
}
report.completedAt = new Date().toISOString();
report.elapsedSeconds = report.cases.reduce(
  (sum, result) => sum + result.elapsedSeconds,
  0,
);
fs.writeFileSync(
  path.join(artifactRoot, "gui-matrix-report.json"),
  `${JSON.stringify(report, null, 2)}\n`,
);
console.log(JSON.stringify(report, null, 2));
