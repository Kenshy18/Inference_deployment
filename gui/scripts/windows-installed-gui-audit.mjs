import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

function option(name, fallback = null) {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const endpoint = option("endpoint", "http://127.0.0.1:9322");
const reportPath = option("report");
const screenshotPath = option("screenshot");
if (!reportPath || !screenshotPath) {
  throw new Error("--report and --screenshot are required");
}

const report = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  endpoint,
  status: "running",
  rendererErrors: [],
  actions: [],
  healthChecks: [],
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

  const assertHealthy = async (name) => {
    await page.waitForTimeout(30);
    const state = await page.evaluate(() => ({
      appCount: document.querySelectorAll(".app").length,
      appVisible: (() => {
        const app = document.querySelector(".app");
        if (!(app instanceof HTMLElement)) return false;
        const rect = app.getBoundingClientRect();
        const style = getComputedStyle(app);
        return rect.width > 0 && rect.height > 0 && style.display !== "none";
      })(),
      bodyLength: document.body.innerText.length,
      rootScrollY: window.scrollY,
      replacementCharacters: (document.body.innerText.match(/\uFFFD/g) ?? []).length,
    }));
    report.healthChecks.push({ name, ...state });
    if (
      state.appCount !== 1 ||
      !state.appVisible ||
      state.bodyLength < 100 ||
      state.rootScrollY !== 0 ||
      state.replacementCharacters !== 0
    ) {
      throw new Error(`GUI health check failed after ${name}: ${JSON.stringify(state)}`);
    }
  };

  await assertHealthy("startup");
  for (const tab of ["LIVE", "STATUS"]) {
    await page.getByRole("tab", { name: tab, exact: true }).click();
    report.actions.push(`monitor:${tab}`);
    await assertHealthy(`monitor:${tab}`);
  }
  await page.locator(".panel").filter({ hasText: "Console" }).waitFor({ state: "visible" });
  report.actions.push("console:visible");
  await assertHealthy("console:visible");

  const inspector = page.locator(".panel--inspector");
  await inspector.getByRole("button", { name: "簡単", exact: true }).click();
  const simpleRules = inspector.locator(".simple-policy-editor__rule");
  if ((await simpleRules.count()) !== 3) {
    throw new Error(`simple class policy expected 3 rules, got ${await simpleRules.count()}`);
  }
  for (let index = 0; index < 3; index += 1) {
    const rule = simpleRules.nth(index);
    const originalShape = (await rule.locator(".seg .is-on").innerText()).trim();
    const alternateShape = originalShape === "楕円" ? "ポリゴン" : "楕円";
    await rule.getByRole("button", { name: alternateShape, exact: true }).click();
    await assertHealthy(`simple-shape:${index}:${alternateShape}`);
    await rule.getByRole("button", { name: originalShape, exact: true }).click();
    const keyframe = rule.locator('input[type="number"]');
    const originalInterval = await keyframe.inputValue();
    await keyframe.fill(String(Number(originalInterval) + 1));
    await keyframe.press("Tab");
    await assertHealthy(`simple-keyframe:${index}`);
    await keyframe.fill(originalInterval);
    await keyframe.press("Tab");
    report.actions.push(`simple-rule:${index}`);
  }
  for (const view of ["簡単", "詳細", "簡単", "詳細"]) {
    await inspector.getByRole("button", { name: view, exact: true }).click();
    report.actions.push(`settings:${view}`);
    await assertHealthy(`settings:${view}`);
  }

  const modeRow = inspector.locator(".row").filter({ hasText: "処理モード" });
  for (const mode of ["性器", "顔", "両方"]) {
    await modeRow.getByRole("button", { name: mode, exact: true }).click();
    report.actions.push(`mode:${mode}`);
    await assertHealthy(`mode:${mode}`);
  }

  // Expand every closed Inspector section, then exercise real pointer clicks
  // on every header. This reproduces the historic document-root scroll/black
  // window regression without changing pipeline data.
  const sectionHeads = inspector.locator(".sec__head");
  for (let index = 0; index < (await sectionHeads.count()); index += 1) {
    const head = sectionHeads.nth(index);
    await head.scrollIntoViewIfNeeded();
    const name = (await head.locator(".sec__name").innerText()).trim();
    const wasOpen = (await head.getAttribute("aria-expanded")) === "true";
    await head.click();
    await assertHealthy(`section-toggle:${name}`);
    await head.click();
    if (!wasOpen) await head.click();
    report.actions.push(`section:${name}`);
  }

  // Exercise all currently rendered enabled custom dropdowns. Pick a
  // different option and restore the original value so this remains a UI
  // binding audit rather than a configuration mutation.
  let selectIndex = 0;
  while (selectIndex < (await inspector.locator(".select__btn:not(:disabled)").count())) {
    const button = inspector.locator(".select__btn:not(:disabled)").nth(selectIndex);
    await button.scrollIntoViewIfNeeded();
    const original = (await button.innerText()).trim();
    await button.click();
    const options = page.getByRole("option");
    const count = await options.count();
    if (count > 1) {
      let replacement = null;
      for (let optionIndex = 0; optionIndex < count; optionIndex += 1) {
        const candidate = (await options.nth(optionIndex).innerText()).trim();
        if (candidate !== original) {
          replacement = candidate;
          await options.nth(optionIndex).click();
          break;
        }
      }
      if (replacement !== null) {
        await assertHealthy(`select:${original}->${replacement}`);
        const currentButtons = inspector.locator(".select__btn:not(:disabled)");
        const restoreButton = currentButtons.nth(
          Math.min(selectIndex, Math.max(0, (await currentButtons.count()) - 1)),
        );
        await restoreButton.scrollIntoViewIfNeeded();
        await restoreButton.click();
        const restoreOption = page.getByRole("option", { name: original, exact: true });
        if ((await restoreOption.count()) > 0) await restoreOption.first().click();
        else await page.keyboard.press("Escape");
      }
    } else {
      await page.keyboard.press("Escape");
    }
    report.actions.push(`select:${original}`);
    selectIndex += 1;
  }

  const checks = inspector.locator('input[type="checkbox"]:not(:disabled)');
  const checkCount = await checks.count();
  for (let index = 0; index < checkCount; index += 1) {
    const current = inspector.locator('input[type="checkbox"]:not(:disabled)').nth(index);
    if ((await current.count()) === 0) break;
    await current.scrollIntoViewIfNeeded();
    const original = await current.isChecked();
    await current.locator("xpath=..").click();
    await assertHealthy(`checkbox:${index}`);
    const restore = inspector.locator('input[type="checkbox"]:not(:disabled)').nth(index);
    if ((await restore.count()) > 0 && (await restore.isChecked()) !== original) {
      await restore.locator("xpath=..").click();
    }
    report.actions.push(`checkbox:${index}`);
  }

  const numbers = inspector.locator('input[type="number"]:not(:disabled)');
  for (let index = 0; index < (await numbers.count()); index += 1) {
    const input = numbers.nth(index);
    await input.scrollIntoViewIfNeeded();
    const original = await input.inputValue();
    const min = Number(await input.getAttribute("min"));
    const max = Number(await input.getAttribute("max"));
    const step = Number(await input.getAttribute("step")) || 1;
    const value = Number(original || 0);
    const candidate = Number.isFinite(max) && value + step > max ? Math.max(min, value - step) : value + step;
    await input.fill(String(candidate));
    await input.press("Tab");
    await assertHealthy(`number:${index}`);
    await input.fill(original);
    await input.press("Tab");
    report.actions.push(`number:${index}`);
  }

  const ranges = inspector.locator('input[type="range"]:not(:disabled)');
  for (let index = 0; index < (await ranges.count()); index += 1) {
    const input = ranges.nth(index);
    await input.scrollIntoViewIfNeeded();
    const original = await input.inputValue();
    await input.focus();
    await input.press("ArrowRight");
    await assertHealthy(`range:${index}`);
    await input.fill(original);
    report.actions.push(`range:${index}`);
  }

  // Ensure a real selected input can be probed and Dry Run can traverse the
  // Windows-to-WSL bridge from the installed application.
  const source = page.locator(".panel").filter({ hasText: "Source" });
  await source.getByRole("button", { name: "参照" }).first().click();
  await source.getByRole("button", { name: /追加/ }).click();
  await page.waitForFunction(() => {
    const queue = JSON.parse(localStorage.getItem("mask-studio-queue") ?? "[]");
    return queue.length > 0 && queue.every((item) => item.frameCount !== null);
  }, null, { timeout: 60_000 });
  const previousJobId = await page.evaluate(async () =>
    (await window.maskStudio.bootstrap()).job.id,
  );
  await page.getByRole("button", { name: /Dry Run/ }).click();
  report.dryRun = await page.evaluate(async (oldId) => {
    const deadline = Date.now() + 90_000;
    while (Date.now() < deadline) {
      const job = (await window.maskStudio.bootstrap()).job;
      if (job.id !== oldId && job.status === "validated") {
        return { id: job.id, status: job.status, exitCode: job.exitCode, error: job.error };
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error("Dry Run did not reach validated with a new job ID");
  }, previousJobId);
  await assertHealthy("dry-run");

  await page.screenshot({ path: screenshotPath, fullPage: true });
  if (report.rendererErrors.length > 0) {
    throw new Error(`renderer errors: ${report.rendererErrors.join(" | ")}`);
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
