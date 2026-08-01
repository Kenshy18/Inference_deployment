import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

function option(name, fallback = null) {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const endpoint = option("endpoint", "http://127.0.0.1:9321");
const reportPath = option("report");
const screenshotPath = option("screenshot");
const outputRoot = option("output-root");
const expectedQueue = Number(option("expected-queue", "1"));
const execute = process.argv.includes("--execute");
const maxFrames = Number(option("max-frames", "120"));

if (!reportPath || !screenshotPath || !outputRoot) {
  throw new Error("--report, --screenshot and --output-root are required");
}

const report = {
  schemaVersion: 1,
  endpoint,
  startedAt: new Date().toISOString(),
  execute,
  checks: {},
  rendererErrors: [],
};

let browser;
try {
  browser = await chromium.connectOverCDP(endpoint);
  const context = browser.contexts()[0];
  const page = context?.pages()[0];
  if (!page) throw new Error("installed Electron window was not found");
  page.on("pageerror", (error) => report.rendererErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") report.rendererErrors.push(`console: ${message.text()}`);
  });
  await page.waitForLoadState("domcontentloaded");
  await page.locator(".app").waitFor({ state: "visible", timeout: 30_000 });
  report.checks.window = {
    title: await page.title(),
    url: page.url(),
    bodyTextLength: await page.locator("body").innerText().then((value) => value.length),
  };

  await page.waitForFunction(() => localStorage.getItem("mask-studio-draft") !== null);
  await page.evaluate(
    ({ outputRootValue, maxFramesValue }) => {
      const draft = JSON.parse(localStorage.getItem("mask-studio-draft"));
      draft.outputRoot = outputRootValue;
      draft.inference.mode = "segmentation-face";
      draft.inference.segmentationModel = "dinov3_codino_mh0";
      draft.inference.segmentationBackend = "tensorrt-fast";
      draft.inference.faceModel = "face_dino_v2";
      draft.inference.faceBackend = "tensorrt-fast";
      draft.inference.maxFrames = maxFramesValue;
      draft.postprocess.enabled = true;
      draft.overlay.enabled = true;
      draft.overlay.executionMode = "fast";
      draft.overlay.presets = ["combined-simple"];
      localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
      localStorage.setItem("mask-studio-queue", "[]");
      localStorage.setItem("mask-studio-settings-view", "advanced");
    },
    { outputRootValue: outputRoot, maxFramesValue: maxFrames },
  );
  await page.reload();
  await page.waitForLoadState("domcontentloaded");
  const source = page.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  await page.waitForFunction(
    (count) => {
      const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
      return queue.length === count && queue.every((item) => item.frameCount !== null);
    },
    expectedQueue,
    { timeout: 60_000 },
  );
  report.checks.queue = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
      title: item.title,
      width: item.width,
      height: item.height,
      fps: item.fps,
      frameCount: item.frameCount,
      status: item.status,
    })),
  );

  await page.getByRole("button", { name: /Dry Run/ }).click();
  await page.waitForFunction(
    async () => (await window.maskStudio.bootstrap()).job.status === "validated",
    null,
    { timeout: 90_000 },
  );
  report.checks.dryRun = await page.evaluate(async () => {
    const job = (await window.maskStudio.bootstrap()).job;
    return { id: job.id, status: job.status, exitCode: job.exitCode, error: job.error };
  });

  if (execute) {
    const previousId = report.checks.dryRun.id;
    const runButton = page.locator(".topbar .btn--primary");
    await page.waitForFunction(() => {
      const button = document.querySelector(".topbar .btn--primary");
      return button instanceof HTMLButtonElement && !button.disabled;
    });
    await runButton.click();
    await page.waitForFunction(
      async (oldId) => {
        const job = (await window.maskStudio.bootstrap()).job;
        return job.id !== oldId && !job.dryRun;
      },
      previousId,
      { timeout: 30_000 },
    );
    await page.waitForFunction(
      () => {
        const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
        return queue.length > 0 && queue.every((item) => !["pending", "processing"].includes(item.status));
      },
      null,
      { timeout: 15 * 60_000 },
    );
    report.checks.finalQueue = await page.evaluate(() =>
      JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
        title: item.title,
        status: item.status,
        error: item.error,
        outputDir: item.outputDir,
      })),
    );
    report.checks.finalJob = await page.evaluate(async () => {
      const job = (await window.maskStudio.bootstrap()).job;
      return {
        status: job.status,
        stage: job.stage,
        exitCode: job.exitCode,
        error: job.error,
        outputRoot: job.outputRoot,
        artifacts: job.artifacts,
      };
    });
  }

  await page.screenshot({ path: screenshotPath, fullPage: true });
  if (report.rendererErrors.length > 0) {
    throw new Error(`renderer errors: ${report.rendererErrors.join(" | ")}`);
  }
  if (execute && report.checks.finalQueue.some((item) => item.status !== "done")) {
    throw new Error(`workflow did not complete: ${JSON.stringify(report.checks.finalQueue)}`);
  }
  report.status = "passed";
} catch (error) {
  report.status = "failed";
  report.error = error instanceof Error ? error.stack ?? error.message : String(error);
  process.exitCode = 1;
} finally {
  report.finishedAt = new Date().toISOString();
  report.elapsedSeconds =
    (Date.parse(report.finishedAt) - Date.parse(report.startedAt)) / 1000;
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  await browser?.close().catch(() => undefined);
}
