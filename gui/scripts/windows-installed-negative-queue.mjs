import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const get = (name, fallback = null) =>
  process.argv.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const endpoint = get("endpoint", "http://127.0.0.1:9330");
const reportPath = get("report");
const outputRoot = get("output-root");
if (!reportPath || !outputRoot) throw new Error("--report and --output-root are required");

const report = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  rendererErrors: [],
  status: "running",
};
let browser;
try {
  browser = await chromium.connectOverCDP(endpoint);
  const page = browser.contexts()[0]?.pages()[0];
  if (!page) throw new Error("installed Electron renderer was not found");
  page.on("pageerror", (error) => report.rendererErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") report.rendererErrors.push(`console: ${message.text()}`);
  });
  await page.waitForLoadState("domcontentloaded");
  await page.waitForFunction(() => localStorage.getItem("mask-studio-draft") !== null);
  await page.evaluate((root) => {
    const draft = JSON.parse(localStorage.getItem("mask-studio-draft"));
    draft.outputRoot = root;
    draft.inference.mode = "segmentation";
    draft.inference.segmentationModel = "dinov3_codino_mh0";
    draft.inference.segmentationBackend = "tensorrt-fast";
    draft.inference.maxFrames = 60;
    draft.inference.fastSqlite = true;
    draft.postprocess.enabled = false;
    draft.overlay.enabled = false;
    localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
    localStorage.setItem("mask-studio-queue", "[]");
    localStorage.setItem("mask-studio-settings-view", "advanced");
  }, outputRoot);
  await page.reload();
  const source = page.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  await page.waitForFunction(
    () => JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").length === 2,
    null,
    { timeout: 30_000 },
  );
  await page.waitForTimeout(3_000);
  report.probedQueue = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
      title: item.title,
      status: item.status,
      width: item.width,
      height: item.height,
      fps: item.fps,
      frameCount: item.frameCount,
    })),
  );
  const button = page.locator(".topbar .btn--primary");
  await button.click();
  await page.waitForFunction(
    () => {
      const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
      return queue.length === 2 && queue.every((item) => !["pending", "processing"].includes(item.status));
    },
    null,
    { timeout: 8 * 60_000 },
  );
  report.finalQueue = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
      title: item.title,
      status: item.status,
      outputDir: item.outputDir,
      error: item.error,
      outputs: item.outputs,
    })),
  );
  report.outputQueueCount = await page.locator(".output-queue .qitem--output").count();
  report.appVisible = await page.locator(".app").isVisible();
  const statuses = report.finalQueue.map((item) => item.status);
  if (statuses[0] !== "failed" || statuses[1] !== "done") {
    throw new Error(`expected failed then done, got ${JSON.stringify(statuses)}`);
  }
  if (!report.finalQueue[0].error) throw new Error("invalid input has no readable error");
  if (report.outputQueueCount !== 1) {
    throw new Error(`expected one published output, got ${report.outputQueueCount}`);
  }
  if (report.rendererErrors.length) throw new Error(report.rendererErrors.join(" | "));
  report.status = "passed";
} catch (error) {
  report.status = "failed";
  report.error = error instanceof Error ? error.stack ?? error.message : String(error);
  process.exitCode = 1;
} finally {
  report.completedAt = new Date().toISOString();
  report.elapsedSeconds = (Date.parse(report.completedAt) - Date.parse(report.startedAt)) / 1000;
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  await browser?.close().catch(() => undefined);
}
