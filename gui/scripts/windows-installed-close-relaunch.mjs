import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawn, spawnSync } from "node:child_process";
import { chromium } from "playwright";

const get = (name, fallback = null) =>
  process.argv.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const installedExe = get("exe");
const video = get("video");
const outputRoot = get("output-root");
const reportPath = get("report");
const distro = get("distro", "Ubuntu-24.04");
if (!installedExe || !video || !outputRoot || !reportPath) {
  throw new Error("--exe, --video, --output-root and --report are required");
}

const report = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  status: "running",
  rendererErrors: [],
};
const userData = fs.mkdtempSync(path.join(os.tmpdir(), "mask-studio-close-relaunch-"));

function waitForExit(child, timeoutMs) {
  if (child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      child.off("exit", done);
      resolve(false);
    }, timeoutMs);
    const done = () => {
      clearTimeout(timer);
      resolve(true);
    };
    child.once("exit", done);
  });
}

async function launch(port) {
  const child = spawn(
    installedExe,
    [
      `--automation-port=${port}`,
      `--automation-video=${video}`,
      `--automation-output=${outputRoot}`,
      `--user-data-dir=${userData}`,
    ],
    { env: { ...process.env, MASK_STUDIO_AUTOMATION_NO_EXTERNAL: "1" }, stdio: "ignore" },
  );
  const deadline = Date.now() + 45_000;
  let browser;
  while (Date.now() < deadline && child.exitCode === null) {
    try {
      browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`);
      break;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  if (!browser) throw new Error(`Electron failed to expose CDP on ${port}`);
  const page = browser.contexts()[0]?.pages()[0];
  if (!page) throw new Error("Electron renderer missing");
  page.on("pageerror", (error) => report.rendererErrors.push(`page: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") report.rendererErrors.push(`console: ${message.text()}`);
  });
  return { child, browser, page };
}

function linuxProcesses(jobId) {
  const check = spawnSync(
    "wsl.exe",
    ["-d", distro, "--", "/usr/bin/pgrep", "-af", jobId],
    { encoding: "utf8", windowsHide: true },
  );
  return check.status === 0 ? check.stdout.trim().split(/\r?\n/).filter(Boolean) : [];
}

async function closeGracefully(app) {
  await app.page.close();
  let exited = await waitForExit(app.child, 15_000);
  if (!exited) {
    app.child.kill();
    exited = await waitForExit(app.child, 5_000);
  }
  await app.browser.close().catch(() => undefined);
  return exited;
}

let first;
let second;
try {
  first = await launch(9332);
  await first.page.waitForLoadState("domcontentloaded");
  await first.page.waitForFunction(() => localStorage.getItem("mask-studio-draft") !== null);
  await first.page.evaluate((root) => {
    const draft = JSON.parse(localStorage.getItem("mask-studio-draft"));
    draft.outputRoot = root;
    draft.inference.mode = "segmentation";
    draft.inference.segmentationModel = "dinov3_codino_mh0";
    draft.inference.segmentationBackend = "tensorrt-fast";
    draft.inference.maxFrames = 3000;
    draft.inference.fastSqlite = true;
    draft.postprocess.enabled = false;
    draft.overlay.enabled = false;
    localStorage.setItem("mask-studio-draft", JSON.stringify(draft));
    localStorage.setItem("mask-studio-queue", "[]");
  }, outputRoot);
  await first.page.reload();
  const source = first.page.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  await first.page.waitForFunction(
    () => JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]")[0]?.frameCount !== null,
    null,
    { timeout: 60_000 },
  );
  await first.page.locator(".topbar .btn--primary").click();
  const running = await first.page.evaluate(async () => {
    const deadline = Date.now() + 90_000;
    while (Date.now() < deadline) {
      const job = (await window.maskStudio.bootstrap()).job;
      if (
        job.status === "running" &&
        job.telemetry.phases.segmentation_inference.state === "running" &&
        job.telemetry.phases.segmentation_inference.completed > 0
      ) {
        return { id: job.id, status: job.status, completed: job.telemetry.phases.segmentation_inference.completed };
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error("first run did not enter segmentation inference");
  });
  report.beforeClose = running;
  report.firstWindowExited = await closeGracefully(first);
  first = null;
  const cleanupDeadline = Date.now() + 60_000;
  let remaining = linuxProcesses(running.id);
  while (remaining.length && Date.now() < cleanupDeadline) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    remaining = linuxProcesses(running.id);
  }
  report.processesAfterClose = remaining;
  if (remaining.length) throw new Error(`WSL children survived normal close: ${remaining.join(" | ")}`);

  second = await launch(9333);
  await second.page.waitForLoadState("domcontentloaded");
  await second.page.waitForFunction(
    () => JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]")[0]?.status === "pending",
    null,
    { timeout: 30_000 },
  );
  report.queueAfterRelaunch = await second.page.evaluate(() =>
    JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]").map((item) => ({
      title: item.title,
      status: item.status,
      outputDir: item.outputDir,
      outputs: item.outputs,
    })),
  );
  await second.page.locator(".topbar .btn--primary").click();
  await second.page.waitForFunction(
    () => JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]")[0]?.status === "done",
    null,
    { timeout: 8 * 60_000 },
  );
  report.finalQueue = await second.page.evaluate(() =>
    JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]"),
  );
  report.finalJob = await second.page.evaluate(async () => (await window.maskStudio.bootstrap()).job);
  if (report.rendererErrors.length) throw new Error(report.rendererErrors.join(" | "));
  report.status = "passed";
} catch (error) {
  report.status = "failed";
  report.error = error instanceof Error ? error.stack ?? error.message : String(error);
  process.exitCode = 1;
} finally {
  if (first) await closeGracefully(first).catch(() => undefined);
  if (second) await closeGracefully(second).catch(() => undefined);
  fs.rmSync(userData, { recursive: true, force: true });
  report.completedAt = new Date().toISOString();
  report.elapsedSeconds = (Date.parse(report.completedAt) - Date.parse(report.startedAt)) / 1000;
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
}
