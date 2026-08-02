import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const get = (name, fallback = null) =>
  process.argv.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const endpoint = get("endpoint", "http://127.0.0.1:9331");
const reportPath = get("report");
const outputRoot = get("output-root");
if (!reportPath || !outputRoot) throw new Error("--report and --output-root are required");

const report = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  rendererErrors: [],
  runs: [],
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
  }, outputRoot);
  await page.reload();
  const source = page.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  await page.waitForFunction(
    () => {
      const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
      return queue.length === 1 && queue[0].frameCount !== null;
    },
    null,
    { timeout: 60_000 },
  );

  for (let run = 0; run < 3; run += 1) {
    if (run > 0) {
      const completed = page.locator(".queue:not(.output-queue) .qitem.is-done");
      await completed.click({ button: "right" });
      await page.getByRole("button", { name: "再処理（未処理に戻す）" }).click();
      await page.locator(".queue:not(.output-queue) .qitem.is-pending").waitFor();
    }
    const priorId = await page.evaluate(async () => (await window.maskStudio.bootstrap()).job.id);
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
      priorId,
      { timeout: 30_000 },
    );
    await page.waitForFunction(
      () => JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]")[0]?.status === "done",
      null,
      { timeout: 8 * 60_000 },
    );
    report.runs.push(
      await page.evaluate(async () => {
        const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]")[0];
        const job = (await window.maskStudio.bootstrap()).job;
        return {
          jobStatus: job.status,
          jobId: job.id,
          currentOutputDir: queue.outputDir,
          outputHistory: queue.outputs.map((item) => item.outputDir),
        };
      }),
    );
  }

  const final = report.runs.at(-1);
  const paths = final.outputHistory;
  report.outputQueueCount = await page.locator(".output-queue .qitem--output").count();
  report.inputQueueCount = await page.locator(".queue:not(.output-queue) .qitem").count();
  for (const entry of await page.locator(".output-queue .qitem--output").all()) {
    await entry.click();
  }
  const basenames = paths.map((value) => path.win32.basename(value));
  report.outputBasenames = basenames;
  if (new Set(paths).size !== 3 || report.outputQueueCount !== 3 || report.inputQueueCount !== 1) {
    throw new Error(`history mismatch: ${JSON.stringify({ paths, output: report.outputQueueCount, input: report.inputQueueCount })}`);
  }
  if (!basenames[1].endsWith("_2") || !basenames[2].endsWith("_3")) {
    throw new Error(`output suffix mismatch: ${JSON.stringify(basenames)}`);
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
